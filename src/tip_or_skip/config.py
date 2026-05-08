from __future__ import annotations

import json
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
HF_SPACE_DIR = SRC_DIR.parent
DATA_DIR = HF_SPACE_DIR / "data"
FINAL_DATASET_DIR = DATA_DIR / "final_dataset"
FINAL_DATASET_PATH = FINAL_DATASET_DIR / "tip_or_skip_final_dataset.parquet"
FINAL_PACKAGE_PATH = FINAL_DATASET_DIR / "tip_or_skip_final_dataset_package.zip"
ARTIFACT_DIR = HF_SPACE_DIR / "artifacts"
FINAL_ARTIFACT_DIR = ARTIFACT_DIR / "final"
REPORT_DIR = ARTIFACT_DIR / "report_data"
FIGURE_DIR = HF_SPACE_DIR / "docs" / "figures"
SUBMISSION_DIR = HF_SPACE_DIR / "submission"

TARGET_COLUMNS = ["tip_amount", "tip_given", "log_tip_amount"]
LEAKAGE_COLUMNS = TARGET_COLUMNS + ["total_amount", "payment_type", "source_file", "source_url"]

DEFAULT_MODEL_FEATURES = [
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

NUMERIC_FEATURES = [
    "pickup_year",
    "pickup_month",
    "pickup_hour",
    "pickup_weekday",
    "is_weekend",
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
    "PULocationID",
    "DOLocationID",
]

CATEGORICAL_FEATURES = [feature for feature in DEFAULT_MODEL_FEATURES if feature not in NUMERIC_FEATURES]
REQUIRED_DATA_COLUMNS = sorted(
    set(
        [
            "taxi_type",
            "time_split",
            "payment_type",
            "fare_amount",
            "trip_distance",
            "trip_duration_minutes",
            *TARGET_COLUMNS,
        ]
    )
)


def model_features() -> list[str]:
    feature_file = FINAL_DATASET_DIR / "model_feature_columns.json"
    if feature_file.exists():
        data = json.loads(feature_file.read_text(encoding="utf-8"))
        return list(data.get("model_features", DEFAULT_MODEL_FEATURES))
    return list(DEFAULT_MODEL_FEATURES)


def ensure_directories() -> None:
    for path in (FINAL_ARTIFACT_DIR, REPORT_DIR, FIGURE_DIR, SUBMISSION_DIR):
        path.mkdir(parents=True, exist_ok=True)
