from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
YEARS = (2024, 2025)
TAXI_TYPES = ("yellow", "green")

TAXI_CONFIGS = {
    "yellow": {
        "pickup_col": "tpep_pickup_datetime",
        "dropoff_col": "tpep_dropoff_datetime",
        "sample_per_file": 30_000,
    },
    "green": {
        "pickup_col": "lpep_pickup_datetime",
        "dropoff_col": "lpep_dropoff_datetime",
        "sample_per_file": 12_000,
    },
}

COMMON_COLUMNS = [
    "VendorID",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
    "airport_fee",
    "cbd_congestion_fee",
    "trip_type",
    "ehail_fee",
]

FINAL_COLUMNS = [
    "taxi_type",
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_year",
    "pickup_month",
    "pickup_year_month",
    "pickup_day",
    "pickup_hour",
    "pickup_weekday",
    "is_weekend",
    "daypart",
    "time_split",
    "trip_duration_minutes",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
    "total_amount",
    "VendorID",
    "vendor_id",
    "passenger_count",
    "passenger_bucket",
    "RatecodeID",
    "ratecode",
    "store_and_fwd_flag",
    "payment_type",
    "trip_type",
    "ehail_fee",
    "PULocationID",
    "DOLocationID",
    "pickup_borough",
    "pickup_zone",
    "pickup_service_zone",
    "dropoff_borough",
    "dropoff_zone",
    "dropoff_service_zone",
    "tip_amount",
    "tip_given",
    "log_tip_amount",
    "source_file",
    "source_url",
]

MODEL_FEATURES = [
    "taxi_type",
    "pickup_year",
    "pickup_month",
    "pickup_hour",
    "pickup_weekday",
    "is_weekend",
    "daypart",
    "trip_duration_minutes",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
    "VendorID",
    "passenger_bucket",
    "ratecode",
    "store_and_fwd_flag",
    "trip_type",
    "PULocationID",
    "DOLocationID",
    "pickup_borough",
    "pickup_zone",
    "pickup_service_zone",
    "dropoff_borough",
    "dropoff_zone",
    "dropoff_service_zone",
]


def source_url(taxi_type: str, year: int, month: int) -> str:
    return f"{BASE_URL}/{taxi_type}_tripdata_{year}-{month:02d}.parquet"


def source_path(taxi_type: str, year: int, month: int) -> Path:
    return ROOT_DIR / f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"


def parse_year_month(path: Path) -> tuple[int, int]:
    match = re.search(r"_(\d{4})-(\d{2})\.parquet$", path.name)
    if not match:
        raise ValueError(f"Could not parse year/month from {path.name}")
    return int(match.group(1)), int(match.group(2))


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def split_for(year: int, month: int) -> str:
    if year == 2024 and month <= 9:
        return "train"
    if year == 2024:
        return "valid"
    return "test"


def daypart_for(hour: pd.Series) -> pd.Series:
    labels = np.select(
        [
            hour.between(5, 10),
            hour.between(11, 15),
            hour.between(16, 20),
            hour.between(21, 23) | hour.between(0, 4),
        ],
        ["morning", "midday", "evening", "night"],
        default="unknown",
    )
    return pd.Series(labels, index=hour.index, dtype="string")


def load_zone_lookup() -> pd.DataFrame:
    lookup = pd.read_csv(ROOT_DIR / "taxi_zone_lookup.csv")
    return lookup.rename(
        columns={
            "LocationID": "location_id",
            "Borough": "borough",
            "Zone": "zone",
            "service_zone": "service_zone",
        }
    )


def read_month(taxi_type: str, path: Path, zone_lookup: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    year, month = parse_year_month(path)
    config = TAXI_CONFIGS[taxi_type]
    pickup_col = config["pickup_col"]
    dropoff_col = config["dropoff_col"]

    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema.names)
    columns = [c for c in COMMON_COLUMNS + [pickup_col, dropoff_col] if c in available_columns]
    table = pq.read_table(path, columns=columns)
    raw_rows = int(table.num_rows)

    mask = pc.equal(table["payment_type"], 1)
    mask = pc.and_(mask, pc.greater(table["fare_amount"], 0))
    mask = pc.and_(mask, pc.greater(table["trip_distance"], 0))
    table = table.filter(mask)

    df = table.to_pandas()
    df = df.rename(columns={pickup_col: "pickup_datetime", dropoff_col: "dropoff_datetime"})

    for column in COMMON_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    if "Airport_fee" in df.columns and "airport_fee" in df.columns:
        df["airport_fee"] = df["airport_fee"].fillna(df["Airport_fee"])
    elif "Airport_fee" in df.columns:
        df["airport_fee"] = df["Airport_fee"]
    else:
        df["airport_fee"] = np.nan

    pickup = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
    duration_minutes = (dropoff - pickup).dt.total_seconds() / 60.0
    in_source_month = (pickup.dt.year == year) & (pickup.dt.month == month)
    df = df.loc[(duration_minutes > 0) & in_source_month].copy()
    pickup = pickup.loc[df.index]
    dropoff = dropoff.loc[df.index]
    duration_minutes = duration_minutes.loc[df.index].clip(upper=180)

    filtered_rows = int(len(df))
    sample_n = min(filtered_rows, int(config["sample_per_file"]))
    if filtered_rows > sample_n:
        df = df.sample(n=sample_n, random_state=2026 + year * 100 + month + (0 if taxi_type == "yellow" else 10_000))
        pickup = pickup.loc[df.index]
        dropoff = dropoff.loc[df.index]
        duration_minutes = duration_minutes.loc[df.index]

    passenger_count = clean_number(df["passenger_count"])
    passenger_count_positive = passenger_count.where(passenger_count > 0)

    df["taxi_type"] = taxi_type
    df["pickup_datetime"] = pickup
    df["dropoff_datetime"] = dropoff
    df["pickup_year"] = pickup.dt.year.astype("Int64")
    df["pickup_month"] = pickup.dt.month.astype("Int64")
    df["pickup_year_month"] = pickup.dt.strftime("%Y-%m")
    df["pickup_day"] = pickup.dt.day.astype("Int64")
    df["pickup_hour"] = pickup.dt.hour.astype("Int64")
    df["pickup_weekday"] = pickup.dt.dayofweek.astype("Int64")
    df["is_weekend"] = df["pickup_weekday"].isin([5, 6]).astype("int8")
    df["daypart"] = daypart_for(df["pickup_hour"].astype(int))
    df["time_split"] = split_for(year, month)
    df["trip_duration_minutes"] = duration_minutes
    df["vendor_id"] = clean_number(df["VendorID"]).fillna(-1).astype("int16").astype(str)
    df["ratecode"] = clean_number(df["RatecodeID"]).fillna(99).astype("int16").astype(str)
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna("Unknown").astype(str).replace({"": "Unknown"})
    df["passenger_bucket"] = (
        passenger_count_positive.fillna(-1)
        .clip(lower=-1, upper=6)
        .astype("int16")
        .map({-1: "Unknown", 0: "Unknown", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6+"})
        .fillna("6+")
    )
    df["tip_amount"] = clean_number(df["tip_amount"]).clip(lower=0)
    df["tip_given"] = (df["tip_amount"] > 0).astype("int8")
    df["log_tip_amount"] = np.log1p(df["tip_amount"])
    df["source_file"] = path.name
    df["source_url"] = source_url(taxi_type, year, month)

    for numeric_column in [
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tolls_amount",
        "improvement_surcharge",
        "congestion_surcharge",
        "airport_fee",
        "cbd_congestion_fee",
        "total_amount",
        "ehail_fee",
    ]:
        df[numeric_column] = clean_number(df[numeric_column])

    df = (
        df.merge(
            zone_lookup.add_prefix("pickup_"),
            left_on="PULocationID",
            right_on="pickup_location_id",
            how="left",
        )
        .merge(
            zone_lookup.add_prefix("dropoff_"),
            left_on="DOLocationID",
            right_on="dropoff_location_id",
            how="left",
        )
        .drop(columns=["pickup_location_id", "dropoff_location_id"])
    )

    for column in [
        "pickup_borough",
        "pickup_zone",
        "pickup_service_zone",
        "dropoff_borough",
        "dropoff_zone",
        "dropoff_service_zone",
    ]:
        df[column] = df[column].fillna("Unknown")

    for column in FINAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    metadata = {
        "taxi_type": taxi_type,
        "year": year,
        "month": month,
        "source_file": path.name,
        "source_url": source_url(taxi_type, year, month),
        "raw_rows": raw_rows,
        "filtered_credit_card_positive_fare_distance_duration_rows": filtered_rows,
        "sampled_rows": int(len(df)),
        "time_split": split_for(year, month),
    }
    return df[FINAL_COLUMNS].reset_index(drop=True), metadata


def build_data_dictionary() -> pd.DataFrame:
    definitions = {
        "taxi_type": ("feature", "Yellow or green TLC taxi fleet."),
        "pickup_datetime": ("raw_reference", "Original pickup timestamp."),
        "dropoff_datetime": ("raw_reference", "Original dropoff timestamp."),
        "pickup_year": ("feature", "Pickup calendar year."),
        "pickup_month": ("feature", "Pickup calendar month."),
        "pickup_year_month": ("feature", "YYYY-MM pickup month label."),
        "pickup_day": ("feature", "Pickup day of month."),
        "pickup_hour": ("feature", "Pickup hour from 0 to 23."),
        "pickup_weekday": ("feature", "Pickup weekday where Monday is 0 and Sunday is 6."),
        "is_weekend": ("feature", "1 when pickup weekday is Saturday or Sunday, else 0."),
        "daypart": ("feature", "Morning, midday, evening, or night bucket derived from pickup hour."),
        "time_split": ("split", "train: 2024 Jan-Sep; valid: 2024 Oct-Dec; test: all 2025."),
        "trip_duration_minutes": ("feature", "Dropoff minus pickup in minutes, clipped at 180."),
        "trip_distance": ("feature", "Trip distance in miles."),
        "fare_amount": ("feature", "Metered fare amount before tip."),
        "extra": ("feature", "TLC extra surcharge field."),
        "mta_tax": ("feature", "MTA tax field."),
        "tolls_amount": ("feature", "Tolls amount field."),
        "improvement_surcharge": ("feature", "Improvement surcharge field."),
        "congestion_surcharge": ("feature", "Congestion surcharge field when present."),
        "airport_fee": ("feature", "Airport fee field when present."),
        "cbd_congestion_fee": ("feature", "CBD congestion fee field when present."),
        "total_amount": ("raw_reference", "Total charged amount; exclude from modeling because it includes tip information."),
        "VendorID": ("feature", "Original TLC vendor ID."),
        "vendor_id": ("feature", "String version of VendorID for categorical modeling."),
        "passenger_count": ("feature", "Passenger count reported by TLC."),
        "passenger_bucket": ("feature", "Passenger count bucket: 1, 2, 3, 4, 5, 6+, or Unknown."),
        "RatecodeID": ("feature", "Original TLC rate code."),
        "ratecode": ("feature", "String version of RatecodeID for categorical modeling."),
        "store_and_fwd_flag": ("feature", "TLC store-and-forward flag."),
        "payment_type": ("filter_reference", "Payment type retained as 1 only; cash tips are not recorded by TLC tip_amount."),
        "trip_type": ("feature", "Green taxi trip type when present."),
        "ehail_fee": ("feature", "Green taxi e-hail fee when present."),
        "PULocationID": ("feature", "Pickup taxi zone ID."),
        "DOLocationID": ("feature", "Dropoff taxi zone ID."),
        "pickup_borough": ("feature", "Pickup taxi-zone borough."),
        "pickup_zone": ("feature", "Pickup taxi-zone name."),
        "pickup_service_zone": ("feature", "Pickup TLC service zone."),
        "dropoff_borough": ("feature", "Dropoff taxi-zone borough."),
        "dropoff_zone": ("feature", "Dropoff taxi-zone name."),
        "dropoff_service_zone": ("feature", "Dropoff TLC service zone."),
        "tip_amount": ("target", "Recorded electronic tip amount in dollars."),
        "tip_given": ("target", "Stage 1 label: 1 if tip_amount > 0, else 0."),
        "log_tip_amount": ("target", "Stage 2 label: log(1 + tip_amount). Use on tipped rows for conditional amount modeling."),
        "source_file": ("provenance", "Raw TLC parquet file used to create the row."),
        "source_url": ("provenance", "Official TLC CloudFront source URL."),
    }
    return pd.DataFrame(
        [
            {
                "column": column,
                "role": definitions[column][0],
                "use_as_model_feature": column in MODEL_FEATURES,
                "description": definitions[column][1],
            }
            for column in FINAL_COLUMNS
        ]
    )


def build_profiles(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    monthly = (
        df.groupby(["taxi_type", "pickup_year", "pickup_month", "time_split"], observed=True)
        .agg(
            rows=("tip_given", "size"),
            tip_rate=("tip_given", "mean"),
            avg_tip_amount=("tip_amount", "mean"),
            median_tip_amount=("tip_amount", "median"),
            avg_fare_amount=("fare_amount", "mean"),
            avg_trip_distance=("trip_distance", "mean"),
            avg_trip_duration_minutes=("trip_duration_minutes", "mean"),
        )
        .reset_index()
        .sort_values(["taxi_type", "pickup_year", "pickup_month"])
    )
    splits = (
        df.groupby(["time_split", "taxi_type"], observed=True)
        .agg(
            rows=("tip_given", "size"),
            tip_rate=("tip_given", "mean"),
            avg_tip_amount=("tip_amount", "mean"),
            positive_tip_rows=("tip_given", "sum"),
        )
        .reset_index()
        .sort_values(["time_split", "taxi_type"])
    )
    hourly = (
        df.groupby(["taxi_type", "daypart", "pickup_hour"], observed=True)
        .agg(
            rows=("tip_given", "size"),
            tip_rate=("tip_given", "mean"),
            avg_tip_amount=("tip_amount", "mean"),
            avg_fare_amount=("fare_amount", "mean"),
        )
        .reset_index()
        .sort_values(["taxi_type", "pickup_hour"])
    )
    zones = (
        df.groupby(["taxi_type", "pickup_borough", "pickup_zone"], observed=True)
        .agg(
            rows=("tip_given", "size"),
            tip_rate=("tip_given", "mean"),
            avg_tip_amount=("tip_amount", "mean"),
            avg_fare_amount=("fare_amount", "mean"),
            avg_trip_distance=("trip_distance", "mean"),
        )
        .reset_index()
        .sort_values(["taxi_type", "tip_rate", "rows"], ascending=[True, False, False])
    )
    return {
        "monthly_profile.csv": monthly,
        "split_profile.csv": splits,
        "hourly_profile.csv": hourly,
        "zone_profile.csv": zones,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset_card(df: pd.DataFrame, manifest: pd.DataFrame, package_files: list[Path]) -> None:
    rows = len(df)
    tip_rate = df["tip_given"].mean()
    train_rows = int((df["time_split"] == "train").sum())
    valid_rows = int((df["time_split"] == "valid").sum())
    test_rows = int((df["time_split"] == "test").sum())
    lines = [
        "# Tip or Skip Final Dataset",
        "",
        "This is the model-ready dataset for the NYC taxi tipping project described in `Proposal.pdf`.",
        "",
        "## Scope",
        "",
        "- Source: official NYC TLC Yellow and Green Taxi Trip Record parquet files.",
        "- Coverage: monthly files for 2024 and 2025.",
        "- Fleets: yellow and green taxis.",
        "- Rows retained: credit-card trips only (`payment_type == 1`), because TLC `tip_amount` does not record cash tips.",
        "- Cleaning: removed rows with nonpositive fare, nonpositive trip distance, nonpositive trip duration, or pickup timestamps outside the source file's year-month.",
        "- Sampling: fixed reproducible sample per fleet-month: 30,000 yellow rows and 12,000 green rows where available.",
        "- Zone join: pickup/dropoff location IDs joined to `taxi_zone_lookup.csv`.",
        "",
        "## Labels",
        "",
        "- Stage 1 label: `tip_given = 1[tip_amount > 0]`.",
        "- Stage 2 label: `log_tip_amount = log(1 + tip_amount)`, used on rows where `tip_given == 1`.",
        "",
        "## Split Policy",
        "",
        "- `train`: January 2024 through September 2024.",
        "- `valid`: October 2024 through December 2024.",
        "- `test`: all 2025 rows.",
        "",
        "This supports the proposal's time-shift experiment: train on early 2024, validate on later 2024, and test on 2025.",
        "",
        "## Row Counts",
        "",
        f"- Total rows: {rows:,}",
        f"- Train rows: {train_rows:,}",
        f"- Validation rows: {valid_rows:,}",
        f"- Test rows: {test_rows:,}",
        f"- Overall recorded electronic tip rate: {tip_rate:.3%}",
        "",
        "## Files",
        "",
        "- `tip_or_skip_final_dataset.parquet`: full model-ready dataset.",
        "- `tip_or_skip_final_dataset.csv.gz`: full model-ready dataset in compressed CSV format.",
        "- `tip_or_skip_preview_10000.csv`: small preview for quick inspection.",
        "- `data_dictionary.csv`: column roles, feature flags, and descriptions.",
        "- `source_manifest.csv`: raw files, official source URLs, raw row counts, filtered row counts, and sampled row counts.",
        "- `monthly_profile.csv`, `split_profile.csv`, `hourly_profile.csv`, `zone_profile.csv`: summary tables for EDA and report figures.",
        "- `taxi_zone_lookup.csv`: TLC taxi-zone lookup used in the join.",
        "- `generate_tip_or_skip_dataset.py`: reproducible generator.",
        "",
        "## Modeling Notes",
        "",
        "- Do not use `tip_amount`, `tip_given`, or `log_tip_amount` as input features.",
        "- Do not use `total_amount` as an input feature because it includes the tip and leaks target information.",
        "- `payment_type` is retained only as provenance/filter documentation; it is constant in this dataset.",
        "- Positive tips here mean recorded electronic tips, not all real-world tips.",
        "",
        "## Source File Summary",
        "",
        f"- Raw source files processed: {len(manifest):,}",
        f"- Raw rows scanned: {int(manifest['raw_rows'].sum()):,}",
        f"- Rows after cleaning before sampling: {int(manifest['filtered_credit_card_positive_fare_distance_duration_rows'].sum()):,}",
        "",
        "## Checksums",
        "",
    ]
    for file_path in package_files:
        lines.append(f"- `{file_path.name}`: `{sha256(file_path)}`")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip(files: list[Path]) -> Path:
    zip_path = OUTPUT_DIR / "tip_or_skip_final_dataset_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in files:
            zf.write(file_path, arcname=file_path.name)
    return zip_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zone_lookup = load_zone_lookup()

    frames = []
    manifest_rows = []
    for taxi_type in TAXI_TYPES:
        for year in YEARS:
            for month in range(1, 13):
                path = source_path(taxi_type, year, month)
                if not path.exists():
                    raise FileNotFoundError(f"Missing source file: {path}")
                print(f"Processing {path.name}...")
                month_df, metadata = read_month(taxi_type, path, zone_lookup)
                frames.append(month_df)
                manifest_rows.append(metadata)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["taxi_type", "pickup_year", "pickup_month", "pickup_datetime"]).reset_index(drop=True)

    dataset_parquet = OUTPUT_DIR / "tip_or_skip_final_dataset.parquet"
    dataset_csv_gz = OUTPUT_DIR / "tip_or_skip_final_dataset.csv.gz"
    preview_csv = OUTPUT_DIR / "tip_or_skip_preview_10000.csv"
    dictionary_csv = OUTPUT_DIR / "data_dictionary.csv"
    manifest_csv = OUTPUT_DIR / "source_manifest.csv"
    feature_columns_json = OUTPUT_DIR / "model_feature_columns.json"
    zone_lookup_copy = OUTPUT_DIR / "taxi_zone_lookup.csv"

    df.to_parquet(dataset_parquet, index=False, compression="zstd")
    df.to_csv(dataset_csv_gz, index=False, compression="gzip")
    df.sample(n=min(10_000, len(df)), random_state=42).to_csv(preview_csv, index=False)

    data_dictionary = build_data_dictionary()
    data_dictionary.to_csv(dictionary_csv, index=False)

    manifest = pd.DataFrame(manifest_rows).sort_values(["taxi_type", "year", "month"])
    manifest.to_csv(manifest_csv, index=False)

    profiles = build_profiles(df)
    profile_paths = []
    for filename, profile in profiles.items():
        path = OUTPUT_DIR / filename
        profile.to_csv(path, index=False)
        profile_paths.append(path)

    feature_columns_json.write_text(json.dumps({"model_features": MODEL_FEATURES}, indent=2), encoding="utf-8")
    shutil.copy2(ROOT_DIR / "taxi_zone_lookup.csv", zone_lookup_copy)

    package_files = [
        dataset_parquet,
        dataset_csv_gz,
        preview_csv,
        dictionary_csv,
        manifest_csv,
        *profile_paths,
        feature_columns_json,
        zone_lookup_copy,
        Path(__file__),
    ]
    write_dataset_card(df, manifest, package_files)
    package_files.append(OUTPUT_DIR / "README.md")
    zip_path = make_zip(package_files)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "tip_rate": float(df["tip_given"].mean()),
        "split_counts": df["time_split"].value_counts().sort_index().to_dict(),
        "taxi_type_counts": df["taxi_type"].value_counts().sort_index().to_dict(),
        "package": str(zip_path.relative_to(ROOT_DIR)),
        "package_sha256": sha256(zip_path),
    }
    (OUTPUT_DIR / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
