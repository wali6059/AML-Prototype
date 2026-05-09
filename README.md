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

Tip or Skip is a Hugging Face Space for an Applied Machine Learning project about NYC taxi tipping. It predicts whether a taxi ride is likely to receive an electronic tip and estimates the expected tip amount.

The project uses NYC TLC Yellow and Green taxi trip data. Cash tips are not recorded in the TLC `tip_amount` field, so the models focus on credit-card trips and recorded electronic tips.

## What The App Does

- Predicts tip probability, conditional tip amount, and expected tip for a user-defined ride.
- Compares the boosted tree hurdle model with the Transformer-MDN on the same ride inputs.
- Shows model metrics, calibration, feature ablations, sequence results, graph results, and driver-copilot checks.
- Provides map and shift-planning views that turn zone-level model outputs into driver-facing recommendations.
- Answers grounded project questions through Ask The Data and Driver Copilot using frozen metrics and model artifacts.
- Provides the printable project blog through `docs/index.html` with figures in `docs/figures/`.

## Main Files

- `app.py`: Gradio app used by the Hugging Face Space.
- `pipeline.py`: data loading, cleaning, feature engineering, training, and prediction utilities.
- `build_artifacts.py`: builds the baseline runtime artifacts from the raw TLC files.
- `src/tip_or_skip/`: final model helpers, deep model inference, driver copilot, fact assistant, and shift planner code.
- `scripts/`: extra model analysis, sequence model, graph model, and driver LLM scripts.
- `artifacts/`: saved runtime files used by the deployed app.
- `data/`: repo-local raw 2024 and 2025 TLC parquet files, `taxi_zone_lookup.csv`, and the processed `final_dataset/` package for reproducible rebuilds.
- `docs/`: printable HTML blog and project figures.

## blog
The blog files are in the docs folder. Also you can see the blog on https://wali6059.github.io/AML-Prototype/

## Artifacts

The deployed app does not train models at runtime. It loads saved files from `artifacts/`, including model bundles, summary tables, final report data, run metrics, and the Transformer-MDN files needed for the same-ride comparison.

The main report is `docs/index.html`. Open it in a browser and use print to save it as a PDF if needed.

## Run Locally

Install the requirements, then run:

```bash
python app.py
```

The app expects the checked-in `artifacts/` files to be present.

## Reproducibility

The repo includes the data needed to rerun the project. Large data files are tracked with Git LFS, so run `git lfs pull` after cloning if the parquet or zip files are missing.

- `data/yellow_tripdata_2024-*.parquet` and `data/green_tripdata_2024-*.parquet` are used by the processed final dataset.
- `data/yellow_tripdata_2025-*.parquet` and `data/green_tripdata_2025-*.parquet` are used by both the baseline artifact pipeline and the processed final dataset.
- `data/taxi_zone_lookup.csv` is used to join TLC location IDs to borough and zone names.
- `data/final_dataset/` contains the processed model-ready dataset used by the final model, plots, sequence experiment, graph experiment, and extra analysis scripts.

There are two dataset paths in the code. `pipeline.py` reads the 2025 raw files from `data/` and is used by `build_artifacts.py` to rebuild the baseline runtime artifacts. The final model and report scripts read `data/final_dataset/tip_or_skip_final_dataset.parquet`, which uses 2024 for training and validation and 2025 for testing.

Training and experiment scripts use fixed sample seeds. The sequence, graph, and driver LLM scripts also expose `--seed`, with `42` as the default.

## Rebuild Artifacts

The raw TLC parquet files, `taxi_zone_lookup.csv`, and processed final dataset package are kept in `data/`, so artifact builds and experiment scripts do not depend on files outside the repo. To rebuild the baseline artifacts, run:

```bash
python build_artifacts.py
```

Extra analysis artifacts can be regenerated with:

```bash
python scripts/model_analysis.py --sample-train 180000 --sample-test 120000
python scripts/train_sequence_model.py --epochs 45 --batch-size 1024 --seed 42
python scripts/train_graph_model.py --epochs 500 --seed 42
python scripts/train_driver_llm.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --batch-size 1 --lr 1e-4 --max-length 384 --lora --seed 42
```

## Deploy

Upload the contents of this folder to the Hugging Face Space. The checked-in artifacts are enough for runtime, and the `data/` folder is included for reproducible rebuilds.

For GitHub Pages, publish the `docs/` folder from the `main` branch.

## Modeling Notes

The main model is a two-stage hurdle model. First it predicts whether a ride receives a tip. Then it predicts the tip amount for rides that receive tips. The expected tip combines both stages.

The boosted tree hurdle model is the strongest point predictor. The Transformer-MDN stays in the project because it gives a range of possible tip amounts, which helps with uncertainty and risk-aware planning.
