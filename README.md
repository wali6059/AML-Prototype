---
title: Tip or Skip
colorFrom: green
colorTo: yellow
sdk: gradio
python_version: "3.10"
sdk_version: "5.23.3"
app_file: app.py
pinned: false
---

# Tip or Skip

`Tip or Skip` is a prototype Hugging Face Space for an Applied Machine Learning project on NYC taxi tipping behavior.

## What the Space shows

- A compact version of the 2025 NYC TLC yellow and green taxi trip dataset pipeline.
- A first-pass two-stage ML workflow:
  - Stage 1 predicts whether a credit-card trip receives a recorded electronic tip.
  - Stage 2 predicts the tip amount when a tip occurs.
- A started technical write-up with background and prior-work context.

## Local build steps

1. Keep the raw TLC parquet files and `taxi_zone_lookup.csv` in the parent `Prototype/` directory.
2. From this `hf_space/` folder, run:

```bash
python build_artifacts.py
```

3. This writes deployable files into `artifacts/`.
4. Launch locally with:

```bash
python app.py
```

## Deploying to Hugging Face

Upload only the contents of this `hf_space/` directory to the Space repository. The raw parquet files are not needed at runtime after `build_artifacts.py` has been executed.

## Important modeling note

The TLC dictionaries state that `tip_amount` does not include cash tips. For that reason, this prototype trains on credit-card trips only and frames the task as predicting recorded electronic tip behavior.
