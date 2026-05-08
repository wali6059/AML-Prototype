from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT_DIR.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"

TAXI_CONFIGS: Dict[str, Dict[str, object]] = {
    "yellow": {
        "pattern": "yellow_tripdata_2025-*.parquet",
        "pickup_col": "tpep_pickup_datetime",
        "dropoff_col": "tpep_dropoff_datetime",
        "extra_columns": ["Airport_fee"],
        "sample_per_file": 30000,
    },
    "green": {
        "pattern": "green_tripdata_2025-*.parquet",
        "pickup_col": "lpep_pickup_datetime",
        "dropoff_col": "lpep_dropoff_datetime",
        "extra_columns": ["trip_type", "ehail_fee"],
        "sample_per_file": 12000,
    },
}

NUMERIC_FEATURES = [
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "trip_distance",
    "fare_amount",
    "trip_duration_minutes",
]

CATEGORICAL_FEATURES = [
    "vendor_id",
    "passenger_bucket",
    "ratecode",
    "store_and_fwd_flag",
    "pickup_borough",
    "pickup_zone",
    "dropoff_borough",
    "dropoff_zone",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

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
    "tip_amount",
    "total_amount",
]

TRAIN_MONTHS = set(range(1, 10))
VALID_MONTHS = {10}
TEST_MONTHS = {11, 12}


def load_zone_lookup(base_dir: Path | None = None) -> pd.DataFrame:
    base_dir = base_dir or RAW_DATA_DIR
    lookup = pd.read_csv(base_dir / "taxi_zone_lookup.csv")
    lookup = lookup.rename(
        columns={
            "LocationID": "location_id",
            "Borough": "borough",
            "Zone": "zone",
            "service_zone": "service_zone",
        }
    )
    return lookup


def _iter_raw_files(taxi_type: str, base_dir: Path | None = None) -> Iterable[Path]:
    base_dir = base_dir or RAW_DATA_DIR
    pattern = TAXI_CONFIGS[taxi_type]["pattern"]
    return sorted(base_dir.glob(pattern))


def _build_filter_mask(table) -> object:
    mask = pc.equal(table["payment_type"], 1)
    mask = pc.and_(mask, pc.greater(table["fare_amount"], 0))
    mask = pc.and_(mask, pc.greater(table["trip_distance"], 0))
    return mask


def _month_split(month: int) -> str:
    if month in TRAIN_MONTHS:
        return "train"
    if month in VALID_MONTHS:
        return "valid"
    return "test"


def _add_engineered_features(df: pd.DataFrame, taxi_type: str, month: int) -> pd.DataFrame:
    pickup = pd.to_datetime(df["pickup_datetime"])
    dropoff = pd.to_datetime(df["dropoff_datetime"])
    duration_minutes = (dropoff - pickup).dt.total_seconds() / 60.0

    df = df.loc[duration_minutes > 0].copy()
    duration_minutes = duration_minutes.loc[df.index]

    passenger_count = pd.to_numeric(df["passenger_count"], errors="coerce")
    passenger_count = passenger_count.where(passenger_count > 0)

    df["taxi_type"] = taxi_type
    df["pickup_month"] = month
    df["pickup_hour"] = pickup.loc[df.index].dt.hour.astype(int)
    df["pickup_weekday"] = pickup.loc[df.index].dt.dayofweek.astype(int)
    df["trip_duration_minutes"] = duration_minutes.clip(upper=180)
    df["vendor_id"] = df["VendorID"].fillna(-1).astype(int).astype(str)
    df["ratecode"] = df["RatecodeID"].fillna(99).astype(int).astype(str)
    df["store_and_fwd_flag"] = (
        df["store_and_fwd_flag"].fillna("Unknown").astype(str).replace({"": "Unknown"})
    )
    df["passenger_bucket"] = (
        passenger_count.fillna(-1)
        .clip(lower=-1, upper=6)
        .astype(int)
        .map(
            {
                -1: "Unknown",
                0: "Unknown",
                1: "1",
                2: "2",
                3: "3",
                4: "4",
                5: "5",
                6: "6+",
            }
        )
        .fillna("6+")
    )
    df["tip_given"] = (df["tip_amount"] > 0).astype(int)
    df["log_tip_amount"] = np.log1p(df["tip_amount"].clip(lower=0))
    df["month_split"] = df["pickup_month"].map(_month_split)
    return df


def sample_taxi_data(
    taxi_type: str,
    base_dir: Path | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    base_dir = base_dir or RAW_DATA_DIR
    config = TAXI_CONFIGS[taxi_type]
    lookup = load_zone_lookup(base_dir)
    frames = []
    pickup_col = config["pickup_col"]
    dropoff_col = config["dropoff_col"]
    columns = COMMON_COLUMNS + [pickup_col, dropoff_col] + list(config["extra_columns"])

    for month_index, parquet_path in enumerate(_iter_raw_files(taxi_type, base_dir), start=1):
        table = pq.read_table(parquet_path, columns=columns)
        table = table.filter(_build_filter_mask(table))
        df = table.to_pandas()
        df = df.rename(columns={pickup_col: "pickup_datetime", dropoff_col: "dropoff_datetime"})
        df = _add_engineered_features(df, taxi_type=taxi_type, month=month_index)

        if len(df) > config["sample_per_file"]:
            df = df.sample(
                n=int(config["sample_per_file"]),
                random_state=random_seed + month_index,
            )

        df = (
            df.merge(
                lookup.add_prefix("pickup_"),
                left_on="PULocationID",
                right_on="pickup_location_id",
                how="left",
            )
            .merge(
                lookup.add_prefix("dropoff_"),
                left_on="DOLocationID",
                right_on="dropoff_location_id",
                how="left",
            )
            .drop(columns=["pickup_location_id", "dropoff_location_id"])
        )

        keep_columns = list(dict.fromkeys(MODEL_FEATURES + [
            "taxi_type",
            "pickup_month",
            "month_split",
            "PULocationID",
            "DOLocationID",
            "tip_amount",
            "tip_given",
            "log_tip_amount",
        ]))
        frames.append(df[keep_columns].reset_index(drop=True))

    combined = pd.concat(frames, ignore_index=True)
    combined["pickup_borough"] = combined["pickup_borough"].fillna("Unknown")
    combined["pickup_zone"] = combined["pickup_zone"].fillna("Unknown")
    combined["dropoff_borough"] = combined["dropoff_borough"].fillna("Unknown")
    combined["dropoff_zone"] = combined["dropoff_zone"].fillna("Unknown")
    return combined


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

def train_models(df: pd.DataFrame) -> Tuple[Pipeline, Pipeline, Dict[str, float]]:
    train_mask = df["month_split"].isin(["train", "valid"])
    test_mask = df["month_split"] == "test"

    X_train = df.loc[train_mask, MODEL_FEATURES]
    y_train = df.loc[train_mask, "tip_given"]
    X_test = df.loc[test_mask, MODEL_FEATURES]
    y_test = df.loc[test_mask, "tip_given"]

    classifier = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    max_iter=150,
                    learning_rate=0.1,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ]
    )
    classifier.fit(X_train, y_train)

    class_probs = classifier.predict_proba(X_test)[:, 1]
    class_preds = (class_probs >= 0.5).astype(int)

    tipped_train = df.loc[train_mask & (df["tip_given"] == 1)]
    tipped_test = df.loc[test_mask & (df["tip_given"] == 1)]

    regressor = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model", 
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=150, 
                    learning_rate=0.1,
                    random_state=42,
                    early_stopping=True,
                )
            ),
        ]
    )
    regressor.fit(tipped_train[MODEL_FEATURES], tipped_train["log_tip_amount"])

    reg_preds_log = regressor.predict(tipped_test[MODEL_FEATURES])

    metrics = {
        "samples_train_valid": int(train_mask.sum()),
        "samples_test": int(test_mask.sum()),
        "tip_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, class_probs)),
        "precision_at_0_5": float(precision_score(y_test, class_preds, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_test, class_preds, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_test, class_preds, zero_division=0)),
        "rmse_log_tip": float(np.sqrt(mean_squared_error(tipped_test["log_tip_amount"], reg_preds_log))),
        "mae_log_tip": float(mean_absolute_error(tipped_test["log_tip_amount"], reg_preds_log)),
    }
    return classifier, regressor, metrics


def save_model_bundle(
    classifier: Pipeline,
    regressor: Pipeline,
    taxi_type: str,
    output_dir: Path | None = None,
) -> Path:
    output_dir = output_dir or ARTIFACT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "classifier": classifier,
        "regressor": regressor,
        "features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "taxi_type": taxi_type,
        "target_note": "Models are trained on credit-card trips only because TLC tip_amount excludes cash tips.",
    }
    output_path = output_dir / f"{taxi_type}_model_bundle.joblib"
    joblib.dump(bundle, output_path)
    return output_path


def build_summary_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    monthly = (
        df.groupby(["taxi_type", "pickup_month"], as_index=False)
        .agg(
            trips=("tip_given", "size"),
            tip_rate=("tip_given", "mean"),
            avg_tip_amount=("tip_amount", "mean"),
            avg_fare_amount=("fare_amount", "mean"),
        )
        .sort_values(["taxi_type", "pickup_month"])
    )

    hourly = (
        df.groupby(["taxi_type", "pickup_hour"], as_index=False)
        .agg(
            trips=("tip_given", "size"),
            tip_rate=("tip_given", "mean"),
            avg_tip_amount=("tip_amount", "mean"),
        )
        .sort_values(["taxi_type", "pickup_hour"])
    )

    zones = (
        df.groupby(["taxi_type", "pickup_borough", "pickup_zone"], as_index=False)
        .agg(
            trips=("tip_given", "size"),
            tip_rate=("tip_given", "mean"),
            avg_tip_amount=("tip_amount", "mean"),
        )
        .sort_values(["taxi_type", "tip_rate", "trips"], ascending=[True, False, False])
    )
    return {"monthly_summary.csv": monthly, "hourly_summary.csv": hourly, "zone_summary.csv": zones}


def save_summary_tables(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    output_dir = output_dir or ARTIFACT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in build_summary_tables(df).items():
        table.to_csv(output_dir / filename, index=False)

    zone_options = (
        df[["pickup_zone", "pickup_borough"]]
        .drop_duplicates()
        .sort_values(["pickup_borough", "pickup_zone"])
        .rename(columns={"pickup_zone": "zone", "pickup_borough": "borough"})
    )
    zone_options.to_csv(output_dir / "zone_options.csv", index=False)

    sample_rows = df.sample(n=min(500, len(df)), random_state=42).copy()
    sample_rows.to_csv(output_dir / "sample_rows.csv", index=False)


def save_metrics(all_metrics: Dict[str, Dict[str, float]], output_dir: Path | None = None) -> None:
    output_dir = output_dir or ARTIFACT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)


def predict_tip(
    bundle: Dict[str, object],
    feature_values: Dict[str, object],
) -> Dict[str, float]:
    row = pd.DataFrame([feature_values], columns=bundle["features"])
    tip_probability = float(bundle["classifier"].predict_proba(row)[:, 1][0])
    conditional_log_tip = float(bundle["regressor"].predict(row)[0])
    conditional_tip = max(0.0, float(np.expm1(conditional_log_tip)))
    expected_tip = tip_probability * conditional_tip
    return {
        "tip_probability": tip_probability,
        "conditional_tip": conditional_tip,
        "expected_tip": expected_tip,
    }

