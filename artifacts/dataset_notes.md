# Dataset Notes

This Space ships compact baseline runtime artifacts generated from the repo-local NYC TLC 2025 taxi trip data.

- Raw data directory: the repo-local `data/` folder.
- Source tables: 12 monthly yellow taxi parquet files and 12 monthly green taxi parquet files.
- Taxi zones available: 265 location IDs.
- Training scope: credit-card trips only, because TLC `tip_amount` excludes cash tips.
- Cleaning rules: dropped rows with nonpositive fare, nonpositive trip distance, and nonpositive trip duration.
- Baseline split policy: January-September train, October validation, November-December test.

The app reads only saved artifacts and does not require the raw parquet files at runtime.