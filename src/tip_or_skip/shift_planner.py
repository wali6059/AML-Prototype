from __future__ import annotations

import pandas as pd


OBJECTIVE_COLUMNS = {
    "risk_neutral": "expected_tip",
    "risk_averse": "q10_tip",
    "probability": "tip_probability",
}


def rank_destinations(candidates: pd.DataFrame, objective: str, top_k: int = 10) -> pd.DataFrame:
    if objective not in OBJECTIVE_COLUMNS:
        raise ValueError(f"Unknown objective {objective}. Expected one of {sorted(OBJECTIVE_COLUMNS)}.")
    score_column = OBJECTIVE_COLUMNS[objective]
    ranked = candidates.copy()
    if score_column == "tip_probability" and score_column not in ranked.columns:
        score_column = "predicted_tip_probability"
    ranked["score"] = ranked[score_column]
    sort_cols = ["score", "observed_trips"] if "observed_trips" in ranked.columns else ["score"]
    return ranked.sort_values(sort_cols, ascending=False).head(top_k).reset_index(drop=True)


def build_candidate_routes(df: pd.DataFrame, pickup_zone: str, taxi_type: str, min_trips: int = 50) -> pd.DataFrame:
    subset = df[(df["pickup_zone"] == pickup_zone) & (df["taxi_type"] == taxi_type)]
    routes = (
        subset.groupby(["dropoff_zone", "dropoff_borough"], dropna=False)
        .agg(
            observed_trips=("tip_given", "size"),
            avg_distance=("trip_distance", "mean"),
            avg_fare=("fare_amount", "mean"),
            avg_duration=("trip_duration_minutes", "mean"),
            observed_tip_rate=("tip_given", "mean"),
            observed_avg_tip=("tip_amount", "mean"),
        )
        .reset_index()
    )
    return routes[routes["observed_trips"] >= min_trips].sort_values("observed_trips", ascending=False)
