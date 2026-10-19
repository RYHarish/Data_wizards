#!/usr/bin/env python3
"""
Combine GDELT 1.0 daily event zips into one large CSV.

GDELT's "*.export.CSV.zip" files are actually TAB-separated, have NO header row,
and have 58 columns. This script:
  - streams each zip member directly (no need to extract to disk, low memory)
  - writes a single header row followed by all rows, in date order
  - converts to a true comma-separated CSV (use --delimiter tab to keep TSV)
  - skips corrupt zips and logs them; malformed rows go to a rejects file
  - writes to a .part file and renames at the end (no half-written output)
  - optionally also extracts the raw files (--extract-dir)

Standard library only.

Examples:
  python gdelt_combine.py
  python gdelt_combine.py --input gdelt_data --output gdelt_2025_H1.csv
  python gdelt_combine.py --start 2025-01-01 --end 2025-06-13
  python gdelt_combine.py --delimiter tab --output combined.tsv
  python gdelt_combine.py --extract-dir gdelt_extracted
"""

import argparse
import csv
import io
import logging
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# GDELT 1.0 Event Database column names (58 columns).
ACTOR_FIELDS = ["Code", "Name", "CountryCode", "KnownGroupCode", "EthnicCode",
                "Religion1Code", "Religion2Code", "Type1Code", "Type2Code", "Type3Code"]
GEO_FIELDS = ["Type", "FullName", "CountryCode", "ADM1Code", "Lat", "Long", "FeatureID"]

COLUMNS = (
    ["GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate"]
    + [f"Actor1{f}" for f in ACTOR_FIELDS]
    + [f"Actor2{f}" for f in ACTOR_FIELDS]
    + ["IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
       "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone"]
    + [f"Actor1Geo_{f}" for f in GEO_FIELDS]
    + [f"Actor2Geo_{f}" for f in GEO_FIELDS]
    + [f"ActionGeo_{f}" for f in GEO_FIELDS]
    + ["DATEADDED", "SOURCEURL"]
)
NCOLS = len(COLUMNS)  # 58

NAME_RE = re.compile(r"^(\d{8})\.export\.CSV\.zip$")


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def find_zips(folder: Path, start, end):
    items = []
    for p in folder.glob("*.export.CSV.zip"):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y%m%d").date()
        if start and d < start:
            continue
        if end and d > end:
            continue
        items.append((d, p))
    items.sort()
    return items


def main():
    ap = argparse.ArgumentParser(description="Combine GDELT daily zips into one CSV")
    ap.add_argument("--input", type=Path, default=Path("gdelt_data"), help="folder with the zip files")
    ap.add_argument("--output", type=Path, default=Path("gdelt_combined.csv"))
    ap.add_argument("--start", type=parse_date, help="YYYY-MM-DD (optional filter)")
    ap.add_argument("--end", type=parse_date, help="YYYY-MM-DD (optional filter)")
    ap.add_argument("--delimiter", choices=["comma", "tab"], default="comma")
    ap.add_argument("--no-header", action="store_true", help="do not write the header row")
    ap.add_argument("--extract-dir", type=Path, help="also extract the raw files here")
    ap.add_argument("--encoding", default="utf-8", help="input encoding (errors are replaced)")
    args = ap.parse_args()

    log = logging.getLogger("combine")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)

    zips = find_zips(args.input, args.start, args.end)
    if not zips:
        log.error("No *.export.CSV.zip files found in %s", args.input.resolve())
        return 1
    log.info("Found %d zip file(s) in %s", len(zips), args.input.resolve())

    if args.extract_dir:
        args.extract_dir.mkdir(parents=True, exist_ok=True)

    delim = "," if args.delimiter == "comma" else "\t"
    part = args.output.with_name(args.output.name + ".part")
    rejects_path = args.output.with_name(args.output.stem + "_rejects.tsv")

    total_rows = total_rejected = 0
    bad_files = []
    ok_files = 0
    rejects_f = None

    try:
        with open(part, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f, delimiter=delim, lineterminator="\n")
            if not args.no_header:
                writer.writerow(COLUMNS)

            for d, zpath in zips:
                file_rows = file_rejected = 0
                try:
                    with zipfile.ZipFile(zpath) as zf:
                        bad = zf.testzip()
                        if bad is not None:
                            raise zipfile.BadZipFile(f"corrupt member {bad}")
                        if args.extract_dir:
                            zf.extractall(args.extract_dir)
                        for name in zf.namelist():
                            if name.endswith("/"):
                                continue
                            with zf.open(name) as raw:
                                text = io.TextIOWrapper(raw, encoding=args.encoding,
                                                        errors="replace", newline="")
                                # QUOTE_NONE: GDELT fields are not quoted; stray quote
                                # characters inside values must not merge lines.
                                reader = csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE)
                                for row in reader:
                                    if not row:
                                        continue
                                    if len(row) != NCOLS:
                                        if rejects_f is None:
                                            rejects_f = open(rejects_path, "w", encoding="utf-8")
                                        rejects_f.write(f"{zpath.name}\t{len(row)}cols\t" + "\t".join(row) + "\n")
                                        file_rejected += 1
                                        continue
                                    writer.writerow(row)
                                    file_rows += 1
                except (zipfile.BadZipFile, OSError, EOFError, RuntimeError) as e:
                    # Rows from a partially-read file may already be written; flag clearly.
                    log.error("BAD ZIP %s: %s (rows written before error: %d)", zpath.name, e, file_rows)
                    bad_files.append(zpath.name)
                    continue

                ok_files += 1
                total_rows += file_rows
                total_rejected += file_rejected
                log.info("OK     %s  rows=%d rejected=%d", zpath.name, file_rows, file_rejected)

        if rejects_f:
            rejects_f.close()
        part.replace(args.output)
    except KeyboardInterrupt:
        log.warning("Interrupted; partial output left at %s (not renamed)", part)
        return 130

    log.info("=" * 60)
    log.info("DONE: %d files combined, %d rows, %d rejected rows -> %s (%.2f GB)",
             ok_files, total_rows, total_rejected, args.output,
             args.output.stat().st_size / 1e9)
    if total_rejected:
        log.warning("Rejected rows (wrong column count) saved to %s", rejects_path)
    if bad_files:
        log.error("Skipped %d bad zip(s): %s", len(bad_files), ", ".join(bad_files))
        log.error("Re-download them with gdelt_download.py, then re-run this script.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
