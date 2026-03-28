# Dataset Notes

This Space ships compact artifacts generated from the NYC TLC 2025 taxi trip data stored locally during development.

- Raw data directory during development: the parent `Prototype/` folder.
- Source tables: 12 monthly yellow taxi parquet files and 12 monthly green taxi parquet files.
- Taxi zones available: 265 location IDs.
- Training scope: credit-card trips only, because TLC `tip_amount` excludes cash tips.
- Cleaning rules: dropped rows with nonpositive fare, nonpositive trip distance, and nonpositive trip duration.
- Split policy: January-September train, October validation, November-December test.

The app reads only saved artifacts and does not require the raw parquet files at runtime.
