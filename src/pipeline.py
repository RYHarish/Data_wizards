from pathlib import Path
import argparse
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_cleaning import clean_gdelt


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_DIR / "gdelt_2025_H1.csv"
PROCESSED_DIR = PROJECT_DIR / "gdelt_data" / "processed"

OUTPUT_DIR = PROCESSED_DIR / "gdelt_cleaned_reproduced"
CSV_SAMPLE_FILE = PROCESSED_DIR / "gdelt_reproduced_sample.csv"

CHUNK_SIZE = 100_000

# GDELT reference files
CAMEO_URL = (
    "https://www.gdeltproject.org/data/lookups/CAMEO.eventcodes.txt"
)

FIPS_URL = (
    "https://www.gdeltproject.org/data/lookups/FIPS.country.txt"
)


def load_references():
    """Load the CAMEO and FIPS reference codes."""

    cameo = pd.read_csv(
        CAMEO_URL,
        sep="\t",
        header=None,
        names=["Code", "Description"],
        dtype="string"
    )

    fips = pd.read_csv(
        FIPS_URL,
        sep="\t",
        header=None,
        dtype="string"
    )

    valid_cameo = set(
        cameo["Code"].str.strip().dropna()
    )

    # Extend reference with verified CAMEO codes
    valid_cameo.update(["106", "1213"])

    valid_fips = set(
        fips.iloc[:, 0].str.strip().dropna()
    )

    return valid_cameo, valid_fips


def get_dtype_map():
    """Define string columns, preserving CAMEO leading zeros."""

    sample = pd.read_csv(
        RAW_FILE,
        nrows=10_000,
        low_memory=False
    )

    string_columns = [
        col for col in sample.columns
        if pd.api.types.is_string_dtype(sample[col])
    ]

    string_columns += [
        "EventCode",
        "EventBaseCode",
        "EventRootCode"
    ]

    return {
        col: "string"
        for col in set(string_columns)
    }


def run_pipeline(test_mode=False):
    """Clean the GDELT dataset and save monthly Parquet files."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(RAW_FILE)

    print("Loading reference files...")

    valid_cameo, valid_fips = load_references()
    dtype_map = get_dtype_map()

    # Test mode processes only 100,000 records
    if test_mode:
        chunk = pd.read_csv(
            RAW_FILE,
            nrows=CHUNK_SIZE,
            dtype=dtype_map,
            low_memory=False
        )

        cleaned = clean_gdelt(
            chunk,
            dtype_map,
            valid_fips,
            valid_cameo
        )

        print("Test rows:", len(cleaned))
        print("Test columns:", len(cleaned.columns))
        print("Quality flags:", sum(
            col.startswith("Flag_")
            for col in cleaned.columns
        ))

        return

    # Avoid overwriting existing datasets
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {OUTPUT_DIR}"
        )

    if CSV_SAMPLE_FILE.exists():
        raise FileExistsError(CSV_SAMPLE_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    writers = {}
    schema = None
    total_rows = 0
    total_chunks = 0
    start_time = time.time()

    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                RAW_FILE,
                chunksize=CHUNK_SIZE,
                dtype=dtype_map,
                low_memory=False
            ),
            start=1
        ):

            cleaned = clean_gdelt(
                chunk,
                dtype_map,
                valid_fips,
                valid_cameo
            )

            # Define a consistent Parquet schema
            if schema is None:
                schema = pa.Table.from_pandas(
                    cleaned,
                    preserve_index=False
                ).schema

                # Save CSV sample
                cleaned.head(1000).to_csv(
                    CSV_SAMPLE_FILE,
                    index=False
                )

            years = cleaned["DateAdded"].dt.strftime("%Y").fillna("unknown")
            months = cleaned["DateAdded"].dt.strftime("%m").fillna("unknown")

            for (year, month), indices in cleaned.groupby(
                [years, months]
            ).groups.items():

                key = (year, month)

                partition_dir = (
                    OUTPUT_DIR
                    / f"DateAddedYear={year}"
                    / f"DateAddedMonth={month}"
                )

                partition_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                if key not in writers:
                    writers[key] = pq.ParquetWriter(
                        partition_dir / "part-0001.parquet",
                        schema,
                        compression="snappy"
                    )

                table = pa.Table.from_pandas(
                    cleaned.loc[indices],
                    schema=schema,
                    preserve_index=False,
                    safe=True
                )

                writers[key].write_table(table)

            total_rows += len(cleaned)
            total_chunks += 1

            if chunk_number % 10 == 0:
                print(
                    f"Chunks: {chunk_number} | "
                    f"Rows: {total_rows:,}"
                )

    finally:
        for writer in writers.values():
            writer.close()

    (OUTPUT_DIR / "_SUCCESS").touch()

    elapsed = (time.time() - start_time) / 60

    print("\nPipeline completed.")
    print("Total chunks:", total_chunks)
    print("Total rows:", f"{total_rows:,}")
    print("Partitions:", len(writers))
    print("Processing time (minutes):", round(elapsed, 2))
    print("Output:", OUTPUT_DIR)

from feature_engineering import run_feature_pipeline



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GDELT cleaning and feature engineering pipeline"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the existing cleaning pipeline in test mode"
    )

    parser.add_argument(
        "--features",
        action="store_true",
        help="Run daily geopolitical feature engineering"
    )

    args = parser.parse_args()

    if args.features:
        if args.test:
            parser.error("--test and --features cannot be used together")

        project_dir = Path(__file__).resolve().parent.parent

        cleaned_dir = (
            project_dir
            / "gdelt_data"
            / "processed"
            / "gdelt_cleaned_by_date_v2"
        )

        feature_output_dir = (
            project_dir / "data" / "processed"
        )

        run_feature_pipeline(
            input_dir=cleaned_dir,
            output_dir=feature_output_dir,
            batch_size=100_000
        )
    else:
        run_pipeline(test_mode=args.test)
