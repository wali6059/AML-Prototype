# Tip or Skip Final Dataset

This is the model-ready dataset for the NYC taxi tipping project described in `Proposal.pdf`.

## Scope

- Source: official NYC TLC Yellow and Green Taxi Trip Record parquet files.
- Coverage: monthly files for 2024 and 2025.
- Fleets: yellow and green taxis.
- Rows retained: credit-card trips only (`payment_type == 1`), because TLC `tip_amount` does not record cash tips.
- Cleaning: removed rows with nonpositive fare, nonpositive trip distance, nonpositive trip duration, or pickup timestamps outside the source file's year-month.
- Sampling: fixed reproducible sample per fleet-month: 30,000 yellow rows and 12,000 green rows where available.
- Zone join: pickup/dropoff location IDs joined to `taxi_zone_lookup.csv`.

## Labels

- Stage 1 label: `tip_given = 1[tip_amount > 0]`.
- Stage 2 label: `log_tip_amount = log(1 + tip_amount)`, used on rows where `tip_given == 1`.

## Split Policy

- `train`: January 2024 through September 2024.
- `valid`: October 2024 through December 2024.
- `test`: all 2025 rows.

This supports the proposal's time-shift experiment: train on early 2024, validate on later 2024, and test on 2025.

## Row Counts

- Total rows: 1,008,000
- Train rows: 378,000
- Validation rows: 126,000
- Test rows: 504,000
- Overall recorded electronic tip rate: 92.894%

## Files

- `tip_or_skip_final_dataset.parquet`: full model-ready dataset.
- `tip_or_skip_final_dataset.csv.gz`: full model-ready dataset in compressed CSV format.
- `tip_or_skip_preview_10000.csv`: small preview for quick inspection.
- `data_dictionary.csv`: column roles, feature flags, and descriptions.
- `source_manifest.csv`: raw files, official source URLs, raw row counts, filtered row counts, and sampled row counts.
- `monthly_profile.csv`, `split_profile.csv`, `hourly_profile.csv`, `zone_profile.csv`: summary tables for EDA and report figures.
- `taxi_zone_lookup.csv`: TLC taxi-zone lookup used in the join.
- `generate_tip_or_skip_dataset.py`: reproducible generator.

## Modeling Notes

- Do not use `tip_amount`, `tip_given`, or `log_tip_amount` as input features.
- Do not use `total_amount` as an input feature because it includes the tip and leaks target information.
- `payment_type` is retained only as provenance/filter documentation; it is constant in this dataset.
- Positive tips here mean recorded electronic tips, not all real-world tips.

## Source File Summary

- Raw source files processed: 48
- Raw rows scanned: 91,143,915
- Rows after cleaning before sampling: 61,342,406

## Checksums

- `tip_or_skip_final_dataset.parquet`: `f80cad6d4cd915b5527c33200ce128a144b1df846c400f754ffe2daceaf3c282`
- `tip_or_skip_final_dataset.csv.gz`: `3a57cf29941af093d345fe7308cc0e44ff2d054777f39fa58a4ae3c6273289d7`
- `tip_or_skip_preview_10000.csv`: `97e559ff3b2042a9c54d22004eaa6744151a2e2305a3164c141a6dc1fcd9e987`
- `data_dictionary.csv`: `8dc96fcdf9cbef1ec5ed61a64ac29728cbcee3679d39155e6962c1923d30fd41`
- `source_manifest.csv`: `29d7851647aad96fcb6bb28df6c2dced4a643c637ba3aa72957a3be36da3f2f1`
- `monthly_profile.csv`: `64a4c6c7c4689abd74aef6a62b2334fef8dc28494e6304ec9e60ccda81ec62d9`
- `split_profile.csv`: `36e95c977696c4c4bd3215316fa1370e650425ad9b75fd1dcc54c0aa64c3a666`
- `hourly_profile.csv`: `5e65937428ce11ff3c1b59e062b7c79552ae75bce527fd396b634d24120a50c0`
- `zone_profile.csv`: `192d6822cd4d4b4988eddbccfe2ed31b0734c0764f1fee778fce594ab59080a9`
- `model_feature_columns.json`: `1d7d689c8b4b36698cf9b4efc91210d34319a2936395ea0643d73a5b816f0be8`
- `taxi_zone_lookup.csv`: `1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed`
- `generate_tip_or_skip_dataset.py`: `072af2811b90d778461033e8798214219e5454fb561ec01d0f41495f64aa30b2`
