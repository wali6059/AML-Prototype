from pathlib import Path

import pandas as pd

from tip_or_skip.data import load_dataset, load_splits, validate_dataset_contract


def test_final_dataset_contract_matches_frozen_package():
    df = load_dataset(
        columns=[
            "taxi_type",
            "time_split",
            "payment_type",
            "fare_amount",
            "trip_distance",
            "trip_duration_minutes",
            "tip_amount",
            "tip_given",
            "log_tip_amount",
        ]
    )

    report = validate_dataset_contract(df)

    assert report["rows"] == 1_008_000
    assert report["split_counts"] == {"test": 504_000, "train": 378_000, "valid": 126_000}
    assert report["taxi_type_counts"] == {"green": 288_000, "yellow": 720_000}
    assert report["non_credit_rows"] == 0
    assert report["nonpositive_fare_rows"] == 0
    assert report["nonpositive_distance_rows"] == 0
    assert report["nonpositive_duration_rows"] == 0
    assert report["null_target_rows"] == 0


def test_load_splits_returns_expected_time_windows():
    train, valid, test = load_splits(
        columns=["taxi_type", "pickup_year", "pickup_month", "time_split", "tip_given"]
    )

    assert len(train) == 378_000
    assert len(valid) == 126_000
    assert len(test) == 504_000
    assert set(train["pickup_year"]) == {2024}
    assert set(valid["pickup_year"]) == {2024}
    assert set(test["pickup_year"]) == {2025}
    assert train["pickup_month"].max() == 9
    assert valid["pickup_month"].min() == 10

