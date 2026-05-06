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

The merged app also includes an NYC zone map view built with `folium`, using TLC zone geometry fetched at runtime so the interface can visualize zone-level tip rate and average tip amount patterns across boroughs.

## What The Space Shows

- An overview tab with model metrics, dataset notes, and sample cleaned rows.
- A grounded `Ask The Data` chat tab that answers questions from the frozen dataset summary, final metrics, subgroup table, and zone-risk table.
- A `Driver Copilot` tab where a driver can ask natural ride-choice questions and compare two concrete ride options.
- A prediction tab where a user enters a hypothetical trip and gets the predicted tip probability, conditional tip, and expected tip.
- A what-if sensitivity panel that sweeps pickup hour, fare, distance, or duration to show how predictions move.
- An exploration tab with precomputed monthly, hourly, and zone-level summaries.
- A model lab tab with final model comparisons, frozen-dataset monthly profiles, and borough subgroup metrics.
- An Experiment Lab tab with feature ablations, calibration bins, graph-flow metrics, sequence LSTM metrics, copilot checks, and driver-LLM fine-tune metrics.
- A maps tab that visualizes borough-level NYC tipping patterns.
- A final results tab with the self-contained offline `index.html`, final metrics, and available local report assets.
- A shift planner tab that ranks zones by expected tip, lower-tail risk, or tip probability.
- A blog tab that displays the project background text from `blog_background.md`.

The chat assistant is deterministic by default so the demo works without secrets. If a Hugging Face Inference API model is configured through `HF_INFERENCE_MODEL` plus `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`, the app can rewrite grounded answers through that hosted model while still using the project artifacts as the source of truth.

The driver-facing copilot works the same way: it parses the driver's prompt for TLC zone names or aliases, retrieves expected-tip and downside-risk evidence from the final model artifacts, and returns a recommendation. The structured ride-comparison form uses the deployed trip predictor directly for two user-specified ride options.

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

## Final project report outputs

The final report generator writes both a paper-style PDF and an offline HTML blog:

```bash
python scripts/generate_latex_report.py
```

The model diagnostics and sequence/graph/language experiment artifacts are generated with:

```bash
python scripts/run_extra_analysis.py --sample-train 180000 --sample-test 120000
python scripts/train_sequence_model.py --epochs 45 --batch-size 1024
python scripts/train_graph_model.py --epochs 500
python scripts/train_driver_llm.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --batch-size 1 --lr 1e-4 --max-length 384 --lora
python scripts/generate_latex_report.py
python scripts/package_submission.py
```

Important outputs:

- `../report/Tip_or_Skip_Final_Report.pdf`
- `../report/Tip_or_Skip_Final_Report.tex`
- `../report/index.html`
- `artifacts/final_report/`
- `artifacts/experiments/`
- `../submission/tip_or_skip_courseworks_blog.zip`

The CourseWorks guideline requires a local `index.html`; the generated `../report/index.html` is self-contained and embeds the report figures directly.
The CourseWorks zip contains the blog `index.html`, figures, PDF report, and a README with the GitHub and Hugging Face links.

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

## Final modeling note

The boosted tree hurdle baseline is the strongest point-prediction model on the final held-out 2025 split. The Transformer-MDN remains part of the final system because it contributes a distribution over positive tips, which powers uncertainty intervals and risk-aware zone ranking.

The Hugging Face app's Model Lab also has a same-ride check. It runs the tree hurdle model and the Transformer-MDN on the same trip inputs, so the user can see the point prediction beside the deep model's lower and upper tip range.
