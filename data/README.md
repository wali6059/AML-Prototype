# Data

This folder contains the raw files needed to rebuild the project artifacts:

- `yellow_tripdata_2024-01.parquet` through `yellow_tripdata_2025-12.parquet`
- `green_tripdata_2024-01.parquet` through `green_tripdata_2025-12.parquet`
- `taxi_zone_lookup.csv`
- `final_dataset/`, the processed dataset package used by the final model and experiment scripts

The parquet files and packaged dataset files are tracked with Git LFS because they are large. The deployed app does not need to read the raw monthly parquet files at runtime, but `build_artifacts.py` uses them to rebuild the baseline model bundles and summary artifacts. The scripts under `scripts/` use `data/final_dataset/` for final-model training and analysis.
