from __future__ import annotations
import json, shutil
from pathlib import Path
import pandas as pd
import torch
from prototype_pipeline import ARTIFACT_DIR, ROOT_DIR, sample_taxi_data, train_models

def main():
    print("--- Starting Artifact Generation Pipeline ---")
    
    if ARTIFACT_DIR.exists(): 
        print(f"Cleaning existing artifacts at {ARTIFACT_DIR}...")
        shutil.rmtree(ARTIFACT_DIR)
    
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    combined_frames = []

    for taxi_type in ("yellow", "green"):
        print(f"\n>> Processing {taxi_type.upper()} taxi data...")
        df = sample_taxi_data(taxi_type)
        
        if df.empty:
            print(f"   Warning: No data found for {taxi_type}, skipping.")
            continue
            
        print(f"   Loaded {len(df)} rows. Starting training...")
        combined_frames.append(df)
        model, preprocessor, metrics = train_models(df)
        
        # 保存为 PyTorch Bundle
        print(f"   Saving {taxi_type} model bundle...")
        bundle = {
            "model_state": model.state_dict(), 
            "preprocessor": preprocessor, 
            "config": {
                "num_numeric": 6, 
                "cat_cardinalities": [len(c) for c in preprocessor.transformers_[1][1].categories_]
            }
        }
        torch.save(bundle, ARTIFACT_DIR / f"{taxi_type}_v2_model.pth")
        all_metrics[taxi_type] = metrics
        print(f"   {taxi_type.upper()} Training Complete. AUC: {metrics['roc_auc']:.4f}")

    if not combined_frames:
        print("\nERROR: No data was processed. Check your data folder and file names.")
        return

    print("\n>> Generating summary tables and auxiliary files...")
    combined = pd.concat(combined_frames, ignore_index=True)
    
    # 生成统计表
    combined.groupby(["taxi_type", "pickup_month"]).agg(trips=("tip_given", "size"), tip_rate=("tip_given", "mean"), avg_tip_amount=("tip_amount", "mean")).reset_index().to_csv(ARTIFACT_DIR / "monthly_summary.csv", index=False)
    combined.groupby(["taxi_type", "pickup_hour"]).agg(trips=("tip_given", "size"), tip_rate=("tip_given", "mean"), avg_tip_amount=("tip_amount", "mean")).reset_index().to_csv(ARTIFACT_DIR / "hourly_summary.csv", index=False)
    combined.groupby(["taxi_type", "pickup_borough", "pickup_zone"]).agg(trips=("tip_given", "size"), tip_rate=("tip_given", "mean"), avg_tip_amount=("tip_amount", "mean")).reset_index().to_csv(ARTIFACT_DIR / "zone_summary.csv", index=False)
    combined[["pickup_zone", "pickup_borough"]].drop_duplicates().rename(columns={"pickup_zone": "zone", "pickup_borough": "borough"}).to_csv(ARTIFACT_DIR / "zone_options.csv", index=False)
    combined.head(500).to_csv(ARTIFACT_DIR / "sample_rows.csv", index=False)
    
    print(">> Writing metrics and documentation...")
    with (ARTIFACT_DIR / "metrics.json").open("w") as f: 
        json.dump(all_metrics, f, indent=2)
    
    # 安全拷贝文件
    bg_path = ROOT_DIR / "blog_background.md"
    if bg_path.exists():
        shutil.copyfile(bg_path, ARTIFACT_DIR / "blog_background.md")
    else:
        (ARTIFACT_DIR / "blog_background.md").write_text("# Blog Draft\n(Background file missing during build)")
        
    (ARTIFACT_DIR / "dataset_notes.md").write_text("# Dataset Notes\nTabular Transformer + MDN Prototype.")

    print(f"\n--- SUCCESS! Artifacts generated in {ARTIFACT_DIR} ---")

if __name__ == "__main__": 
    main()