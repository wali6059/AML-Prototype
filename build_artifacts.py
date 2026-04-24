from __future__ import annotations

import shutil
import json
from pathlib import Path
import pandas as pd
import torch

from prototype_pipeline import (
    ARTIFACT_DIR,
    ROOT_DIR,
    RAW_DATA_DIR,
    load_zone_lookup,
    sample_taxi_data,
    train_models,
    save_model_bundle,
    save_summary_tables,
)

def save_metrics(all_metrics: Dict[str, Dict[str, float]], output_dir: Path | None = None) -> None:
    output_dir = output_dir or ARTIFACT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)

def copy_blog_background() -> None:
    source = ROOT_DIR / "blog_background.md"
    target = ARTIFACT_DIR / "blog_background.md"
    if source.exists():
        shutil.copyfile(source, target)
    else:
        print(f"Warning: {source} not found, skipping copy.")

def write_dataset_card() -> None:
    zone_lookup = load_zone_lookup(RAW_DATA_DIR)
    lines = [
        "# Dataset Notes",
        "",
        "This Space ships compact artifacts generated from the NYC TLC 2025 taxi trip data.",
        "",
        "## Technical Specifications",
        f"- **Model Architecture**: Tabular Transformer with Mixture Density Network (MDN) Head.",
        "- **Training Scope**: Credit-card trips only (recorded electronic tips).",
        "- **Cleaning**: Dropped nonpositive fare, distance, and duration.",
        f"- **Taxi Zones**: {len(zone_lookup)} location IDs integrated.",
        "- **Split Policy**: Jan-Sep Train, Oct Valid, Nov-Dec Test.",
        "",
        "The application utilizes pre-trained PyTorch weights and does not require raw Parquet files at runtime.",
    ]
    (ARTIFACT_DIR / "dataset_notes.md").write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    print(f"Initializing artifact generation at {ARTIFACT_DIR}...")
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {}
    combined_frames = []

    for taxi_type in ("yellow", "green"):
        print(f"\n--- Processing {taxi_type.upper()} Taxi Data ---")

        print(f"Loading and sampling {taxi_type} records...")
        df = sample_taxi_data(taxi_type=taxi_type, base_dir=RAW_DATA_DIR)
        combined_frames.append(df)

        print(f"Training Deep Learning model on {len(df):,} sampled rows...")
        model, preprocessor, metrics = train_models(df)

        save_model_bundle(model, preprocessor, taxi_type)
        all_metrics[taxi_type] = metrics
        print(f"Saved {taxi_type} PyTorch model bundle and metrics.")

    print("\n--- Finalizing Artifacts ---")
    combined = pd.concat(combined_frames, ignore_index=True)
    
    print("Generating summary tables for visualization...")
    save_summary_tables(combined)
    
    print("Saving global metrics...")
    save_metrics(all_metrics)
    
    print("Writing metadata and copying blog background...")
    copy_blog_background()
    write_dataset_card()
    
    print(f"\nSuccess! All artifacts written to {ARTIFACT_DIR}")
    print("You can now launch the app with: python app.py")

if __name__ == "__main__":
    main()
