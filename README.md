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

## Repository Contents

- `app.py`: Gradio application used in the Hugging Face Space.
- `prototype_pipeline.py`: data loading, feature engineering, model training, artifact creation, and inference utilities.
- `build_artifacts.py`: offline build script that samples the raw TLC files, trains baseline models, and writes runtime artifacts.
- `blog_background.md`: project background text used in the Space blog tab and the GitHub Pages site.
- `artifacts/`: deployable files used at runtime by the Space, including trained model bundles and summary tables.
- `docs/`: GitHub Pages site that renders the project blog as a static webpage.

## What The Prototype Does

This repository contains a compact end-to-end prototype built around the 2025 NYC TLC yellow and green taxi trip datasets. The prototype demonstrates four things:

1. Raw monthly parquet trip files can be ingested and standardized across taxi types.
2. The data can be cleaned and transformed into a supervised learning dataset focused on recorded electronic tipping behavior.
3. A two-stage modeling pipeline can be trained and evaluated on that processed data.
4. The outputs can be deployed in a lightweight public interface through Hugging Face Spaces and GitHub Pages.

## Data Preprocessing

The preprocessing logic is implemented in `prototype_pipeline.py`.

1. The pipeline reads all 12 monthly parquet files for yellow taxis and all 12 monthly parquet files for green taxis.
2. It keeps only the columns needed for the prototype, including trip times, locations, fare values, payment type, and `tip_amount`.
3. It filters the data to credit-card trips only by enforcing `payment_type == 1`, because TLC does not record cash tips in `tip_amount`.
4. It removes rows with nonpositive fare amounts, nonpositive trip distance, or nonpositive trip duration.
5. It derives engineered features such as pickup hour, pickup weekday, pickup month, trip duration in minutes, vendor ID, passenger-count bucket, rate code, and cleaned store-and-forward flags.
6. It joins pickup and dropoff location IDs with `taxi_zone_lookup.csv` so the model sees borough and zone names instead of raw IDs alone.
7. It samples a manageable number of rows per monthly file to keep training and deployment lightweight.
8. It splits the data by month: January to September for training, October for validation-style development use, and November to December for testing.

## How We Implemented The Prototype

The prototype uses a hurdle-style two-stage baseline:

1. Stage 1 is a binary classifier that predicts whether a trip receives any recorded electronic tip.
2. Stage 2 is a regressor trained only on tipped rides to estimate the tip amount conditional on a tip happening.

For both yellow and green taxi subsets, the current baseline uses histogram-based gradient boosting models. Categorical features are one-hot encoded; numeric features are passed through directly. After training, the repository writes:

- model bundles for each taxi type,
- evaluation metrics,
- monthly, hourly, and zone-level summary tables,
- sampled rows for display in the app,
- supporting markdown files for the dataset notes and project blog.

The Hugging Face Space then loads only those saved artifacts at runtime. That means the deployed app does not need the full raw TLC parquet files once artifact generation has already been completed locally.

## What The Space Shows

- An overview tab with model metrics, dataset notes, and sample cleaned rows.
- A prediction tab where a user enters a hypothetical trip and gets the predicted tip probability, conditional tip, and expected tip.
- An exploration tab with precomputed monthly, hourly, and zone-level summaries.
- A blog tab that displays the project background text from `blog_background.md`.

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

## GitHub Pages

This repository also contains a static GitHub Pages site under `docs/`. It renders the contents of `blog_background.md` as a standalone webpage.

To publish it on GitHub Pages:

1. Push this repository to GitHub.
2. In the GitHub repository settings, open `Pages`.
3. Set the source to `Deploy from a branch`.
4. Select branch `main` and folder `/docs`.
5. Save, then wait for GitHub Pages to publish the site.

## Important modeling note

The TLC dictionaries state that `tip_amount` does not include cash tips. For that reason, this prototype trains on credit-card trips only and frames the task as predicting recorded electronic tip behavior.

## Next Steps

- Replace the baseline tree models with a stronger deep tabular architecture.
- Add richer spatial features from the taxi-zone shapefiles and zone adjacency structure.
- Improve evaluation with stronger calibration analysis, threshold tuning, and subgroup breakdowns.
- Extend the second stage from point prediction toward a probabilistic or mixture-based conditional tip model.
- Add visual outputs that better summarize where and when high expected tips occur across NYC.
