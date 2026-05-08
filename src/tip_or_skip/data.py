from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import FINAL_DATASET_PATH, REQUIRED_DATA_COLUMNS


def load_dataset(columns: Iterable[str] | None = None, path: Path | None = None) -> pd.DataFrame:
    dataset_path = path or FINAL_DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Final dataset not found at {dataset_path}. "
            "Generate or restore it in data/final_dataset first."
        )
    return pd.read_parquet(dataset_path, columns=list(columns) if columns is not None else None)


def load_splits(columns: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    requested = list(columns) if columns is not None else None
    if requested is not None and "time_split" not in requested:
        requested = requested + ["time_split"]
    df = load_dataset(columns=requested)
    train = df[df["time_split"] == "train"].copy()
    valid = df[df["time_split"] == "valid"].copy()
    test = df[df["time_split"] == "test"].copy()
    return train, valid, test


def validate_dataset_contract(df: pd.DataFrame) -> dict[str, object]:
    missing = [column for column in REQUIRED_DATA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    report = {
        "rows": int(len(df)),
        "split_counts": df["time_split"].value_counts().sort_index().to_dict(),
        "taxi_type_counts": df["taxi_type"].value_counts().sort_index().to_dict()
        if "taxi_type" in df.columns
        else {},
        "non_credit_rows": int((df["payment_type"] != 1).sum()) if "payment_type" in df.columns else 0,
        "nonpositive_fare_rows": int((df["fare_amount"] <= 0).sum()) if "fare_amount" in df.columns else 0,
        "nonpositive_distance_rows": int((df["trip_distance"] <= 0).sum())
        if "trip_distance" in df.columns
        else 0,
        "nonpositive_duration_rows": int((df["trip_duration_minutes"] <= 0).sum())
        if "trip_duration_minutes" in df.columns
        else 0,
        "null_target_rows": int(df[["tip_amount", "tip_given", "log_tip_amount"]].isna().any(axis=1).sum()),
    }
    return report


def sample_for_development(df: pd.DataFrame, max_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
