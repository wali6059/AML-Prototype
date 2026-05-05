from __future__ import annotations

import pandas as pd


def zone_risk_summary(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([df.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    return (
        merged.groupby(["taxi_type", "pickup_borough", "pickup_zone"], dropna=False)
        .agg(
            rows=("tip_given", "size"),
            observed_tip_rate=("tip_given", "mean"),
            predicted_tip_probability=("tip_probability", "mean"),
            observed_avg_tip=("tip_amount", "mean"),
            expected_tip=("expected_tip", "mean"),
            q10_tip=("q10_tip", "mean"),
            q50_tip=("q50_tip", "mean"),
            q90_tip=("q90_tip", "mean"),
        )
        .reset_index()
        .sort_values(["taxi_type", "expected_tip"], ascending=[True, False])
    )

