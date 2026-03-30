from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from prototype_pipeline import (
    ARTIFACT_DIR,
    ROOT_DIR,
    RAW_DATA_DIR,
    load_zone_lookup,
    sample_taxi_data,
    save_metrics,
    save_model_bundle,
    save_summary_tables,
    train_models,
)


def copy_blog_background() -> None:
    source = ROOT_DIR / "blog_background.md"
    target = ARTIFACT_DIR / "blog_background.md"
    shutil.copyfile(source, target)


def write_dataset_card() -> None:
    zone_lookup = load_zone_lookup(RAW_DATA_DIR)
    lines = [
        "# Dataset Notes",
        "",
        "This Space ships compact artifacts generated from the NYC TLC 2025 taxi trip data stored locally during development.",
        "",
        "- Raw data directory during development: the parent `Prototype/` folder.",
        "- Source tables: 12 monthly yellow taxi parquet files and 12 monthly green taxi parquet files.",
        f"- Taxi zones available: {len(zone_lookup)} location IDs.",
        "- Training scope: credit-card trips only, because TLC `tip_amount` excludes cash tips.",
        "- Cleaning rules: dropped rows with nonpositive fare, nonpositive trip distance, and nonpositive trip duration.",
        "- Split policy: January-September train, October validation, November-December test.",
        "",
        "The app reads only saved artifacts and does not require the raw parquet files at runtime.",
    ]
    (ARTIFACT_DIR / "dataset_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    combined_frames = []

    for taxi_type in ("yellow", "green"):
        print(f"Loading and sampling {taxi_type} taxi data...")
        df = sample_taxi_data(taxi_type=taxi_type, base_dir=RAW_DATA_DIR)
        combined_frames.append(df)
        print(f"Training models for {taxi_type} taxi data on {len(df):,} sampled rows...")
        classifier, regressor, metrics = train_models(df)
        save_model_bundle(classifier, regressor, taxi_type)
        all_metrics[taxi_type] = metrics
        print(f"Saved {taxi_type} model bundle.")

    combined = pd.concat(combined_frames, ignore_index=True)
    save_summary_tables(combined)
    save_metrics(all_metrics)
    copy_blog_background()
    write_dataset_card()
    print(f"Artifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
