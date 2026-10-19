#!/usr/bin/env python3
"""
Robust GDELT daily events downloader.

Downloads https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip for a
date range into a folder, with:
  - exponential backoff + jitter (honours Retry-After on 429/503)
  - atomic writes (.part file -> rename), so no half-written files are kept
  - zip integrity validation (corrupt downloads are retried)
  - resume: files already downloaded and valid are skipped
  - a log file plus failed_files.txt / permanent_missing.txt reports
  - small thread pool (default 4 workers) to be polite to the server
  - optional unzip

Only the Python standard library is used (3.8+).

Examples:
  python gdelt_download.py
  python gdelt_download.py --start 2025-01-01 --end 2025-06-30 --out gdelt_data
  python gdelt_download.py --workers 2 --unzip
  python gdelt_download.py --retry-failed        # re-attempt only previous failures
"""

import argparse
import logging
import random
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

BASE_URL = "https://data.gdeltproject.org/events/{ymd}.export.CSV.zip"
USER_AGENT = "gdelt-bulk-downloader/1.0 (research use)"

# HTTP codes worth retrying; everything else 4xx is treated as permanent.
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None and len(zf.namelist()) > 0
    except (zipfile.BadZipFile, OSError):
        return False


def backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter."""
    return random.uniform(0, min(cap, base * (2 ** (attempt - 1))))


def download_one(d: date, out_dir: Path, args, log: logging.Logger):
    """
    Returns (date_str, status, detail)
    status in: ok, skipped, missing (404, permanent), failed
    """
    ymd = d.strftime("%Y%m%d")
    url = BASE_URL.format(ymd=ymd)
    dest = out_dir / f"{ymd}.export.CSV.zip"
    part = out_dir / f"{ymd}.export.CSV.zip.part"

    if dest.exists():
        if is_valid_zip(dest):
            log.info("SKIP   %s (already downloaded and valid)", ymd)
            return ymd, "skipped", ""
        log.warning("%s exists but is corrupt; re-downloading", dest.name)
        dest.unlink(missing_ok=True)

    last_err = ""
    for attempt in range(1, args.retries + 1):
        retry_after = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                expected = resp.headers.get("Content-Length")
                written = 0
                with open(part, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)

            if expected is not None and written != int(expected):
                raise IOError(f"truncated download: got {written} of {expected} bytes")
            if not is_valid_zip(part):
                raise IOError("downloaded file is not a valid zip")

            part.replace(dest)  # atomic on same filesystem
            if args.unzip:
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(out_dir)
            log.info("OK     %s (%.1f MB, attempt %d)", ymd, written / 1e6, attempt)
            return ymd, "ok", ""

        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code} {e.reason}"
            if e.code == 404:
                log.error("MISSING %s -> 404 (not published on server): %s", ymd, url)
                part.unlink(missing_ok=True)
                return ymd, "missing", last_err
            if e.code not in RETRYABLE_HTTP:
                log.error("FAILED %s -> %s (non-retryable)", ymd, last_err)
                part.unlink(missing_ok=True)
                return ymd, "failed", last_err
            ra = e.headers.get("Retry-After") if e.headers else None
            if ra and ra.isdigit():
                retry_after = float(ra)
        except (urllib.error.URLError, socket.timeout, TimeoutError,
                ConnectionError, IOError, zipfile.BadZipFile) as e:
            last_err = f"{type(e).__name__}: {e}"
        except Exception as e:  # never let one date kill the whole run
            last_err = f"unexpected {type(e).__name__}: {e}"

        part.unlink(missing_ok=True)
        if attempt < args.retries:
            delay = retry_after if retry_after is not None else \
                backoff_delay(attempt, args.backoff_base, args.backoff_cap)
            log.warning("RETRY  %s attempt %d/%d failed (%s); sleeping %.1fs",
                        ymd, attempt, args.retries, last_err, delay)
            time.sleep(delay)

    log.error("FAILED %s after %d attempts: %s", ymd, args.retries, last_err)
    return ymd, "failed", last_err


def read_previous_failures(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip().split("\t")[0]
        if len(line) == 8 and line.isdigit():
            out.append(datetime.strptime(line, "%Y%m%d").date())
    return out


def main():
    p = argparse.ArgumentParser(description="Robust GDELT daily events downloader")
    p.add_argument("--start", type=parse_date, default=date(2025, 1, 1))
    p.add_argument("--end", type=parse_date, default=date(2025, 6, 30))
    p.add_argument("--out", type=Path, default=Path("gdelt_data"))
    p.add_argument("--workers", type=int, default=4, help="parallel downloads (be polite)")
    p.add_argument("--retries", type=int, default=6, help="max attempts per file")
    p.add_argument("--backoff-base", type=float, default=2.0, help="seconds")
    p.add_argument("--backoff-cap", type=float, default=120.0, help="max sleep seconds")
    p.add_argument("--timeout", type=float, default=60.0, help="socket timeout seconds")
    p.add_argument("--unzip", action="store_true", help="extract CSV after download")
    p.add_argument("--retry-failed", action="store_true",
                   help="only re-attempt dates listed in failed_files.txt")
    args = p.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("gdelt")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(out_dir / "download.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)

    failed_path = out_dir / "failed_files.txt"
    missing_path = out_dir / "permanent_missing.txt"

    if args.retry_failed:
        dates = read_previous_failures(failed_path)
        if not dates:
            log.info("No previous failures recorded in %s", failed_path)
            return 0
    else:
        if args.end < args.start:
            p.error("--end must be >= --start")
        dates = list(daterange(args.start, args.end))

    log.info("Downloading %d file(s) to %s with %d worker(s)",
             len(dates), out_dir.resolve(), args.workers)

    results = {"ok": [], "skipped": [], "missing": [], "failed": []}
    details = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(download_one, d, out_dir, args, log): d for d in dates}
            for fut in as_completed(futs):
                ymd, status, detail = fut.result()
                results[status].append(ymd)
                details[ymd] = detail
    except KeyboardInterrupt:
        log.warning("Interrupted. Re-run the script to resume; finished files are kept.")
        return 130

    # Reports (overwrite on each run so they reflect the latest state)
    failed_lines = [f"{y}\t{details[y]}" for y in sorted(results["failed"])]
    missing_lines = [f"{y}\t{details[y]}" for y in sorted(results["missing"])]
    failed_path.write_text("\n".join(failed_lines) + ("\n" if failed_lines else ""))
    missing_path.write_text("\n".join(missing_lines) + ("\n" if missing_lines else ""))

    log.info("=" * 60)
    log.info("SUMMARY: %d downloaded, %d skipped (already had), %d missing on server, %d failed",
             len(results["ok"]), len(results["skipped"]),
             len(results["missing"]), len(results["failed"]))
    if results["missing"]:
        log.info("Missing (404) dates: %s -> %s", ", ".join(sorted(results["missing"])), missing_path)
    if results["failed"]:
        log.info("Failed dates: %s -> %s", ", ".join(sorted(results["failed"])), failed_path)
        log.info("Re-run with --retry-failed (or just re-run normally) to try again.")
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())