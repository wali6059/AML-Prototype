from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def ablation_columns(columns: Iterable[str], name: str) -> list[str]:
    cols = list(columns)
    groups = {
        "no_zones": ["zone", "borough", "locationid", "service_zone"],
        "no_time": ["pickup_year", "pickup_month", "pickup_day", "pickup_hour", "pickup_weekday", "is_weekend", "daypart", "datetime"],
        "no_fare": ["fare", "extra", "tax", "toll", "surcharge", "airport", "cbd", "congestion", "improvement"],
        "no_distance": ["distance", "duration"],
    }
    words = groups.get(name, [])
    if not words:
        return cols
    out = []
    for col in cols:
        key = col.lower()
        if not any(word in key for word in words):
            out.append(col)
    return out


def calibration_table(y_true: Iterable[float], y_prob: Iterable[float], bins: int = 10) -> pd.DataFrame:
    y = pd.Series(y_true, dtype=float)
    p = pd.Series(y_prob, dtype=float).clip(0, 1)
    edges = np.linspace(0, 1, bins + 1)
    labels = pd.cut(p, edges, include_lowest=True, duplicates="drop")
    rows = []
    for interval, idx in labels.groupby(labels, observed=True).groups.items():
        if len(idx) == 0:
            continue
        pred = round(float(p.iloc[list(idx)].mean()), 12)
        actual = round(float(y.iloc[list(idx)].mean()), 12)
        rows.append(
            {
                "low": max(0.0, float(interval.left)),
                "high": float(interval.right),
                "rows": int(len(idx)),
                "predicted": pred,
                "actual": actual,
                "gap": actual - pred,
            }
        )
    return pd.DataFrame(rows)


def flow_features(df: pd.DataFrame) -> pd.DataFrame:
    trips = df.copy()
    if "tip_given" not in trips.columns:
        trips["tip_given"] = (trips["tip_amount"] > 0).astype(int)

    pu = trips.groupby("PULocationID").agg(
        pickup_count=("PULocationID", "size"),
        pickup_zone=("pickup_zone", "first"),
        avg_tip=("tip_amount", "mean"),
        tip_rate=("tip_given", "mean"),
    )
    do = trips.groupby("DOLocationID").agg(
        dropoff_count=("DOLocationID", "size"),
        dropoff_zone=("dropoff_zone", "first"),
    )
    out_degree = trips.groupby("PULocationID")["DOLocationID"].nunique().rename("out_degree")
    in_degree = trips.groupby("DOLocationID")["PULocationID"].nunique().rename("in_degree")
    zones = sorted(set(pu.index).union(set(do.index)))
    out = pd.DataFrame(index=zones)
    out.index.name = "zone_id"
    out = out.join([pu, do, out_degree, in_degree]).reset_index()
    out["zone"] = out["pickup_zone"].fillna(out["dropoff_zone"])
    out["pickup_count"] = out["pickup_count"].fillna(0).astype(int)
    out["dropoff_count"] = out["dropoff_count"].fillna(0).astype(int)
    out["out_degree"] = out["out_degree"].fillna(0).astype(int)
    out["in_degree"] = out["in_degree"].fillna(0).astype(int)
    out["flow_count"] = out["pickup_count"] + out["dropoff_count"]
    out["avg_tip"] = out["avg_tip"].fillna(0.0)
    out["tip_rate"] = out["tip_rate"].fillna(0.0)
    keep = ["zone_id", "zone", "pickup_count", "dropoff_count", "out_degree", "in_degree", "flow_count", "avg_tip", "tip_rate"]
    return out[keep].sort_values(["pickup_count", "zone_id"], ascending=[False, True]).reset_index(drop=True)


def sequence_rows(df: pd.DataFrame, by_zone: bool = False) -> pd.DataFrame:
    trips = df.copy()
    trips["pickup_datetime"] = pd.to_datetime(trips["pickup_datetime"])
    trips["date"] = trips["pickup_datetime"].dt.date
    trips["hour"] = trips["pickup_datetime"].dt.hour
    if "tip_given" not in trips.columns:
        trips["tip_given"] = (trips["tip_amount"] > 0).astype(int)
    keys = ["taxi_type", "date", "hour"]
    if by_zone:
        keys.append("PULocationID")
    hourly = trips.groupby(keys).agg(
        rows=("tip_amount", "size"),
        tip_rate=("tip_given", "mean"),
        avg_tip=("tip_amount", "mean"),
        median_tip=("tip_amount", "median"),
    ).reset_index()
    sort_keys = [key for key in keys if key != "hour"] + ["hour"]
    hourly = hourly.sort_values(sort_keys).reset_index(drop=True)
    group_keys = [key for key in keys if key != "hour"]
    hourly["tip_rate_next"] = hourly.groupby(group_keys)["tip_rate"].shift(-1)
    hourly["avg_tip_next"] = hourly.groupby(group_keys)["avg_tip"].shift(-1)
    hourly["hour_next"] = hourly.groupby(group_keys)["hour"].shift(-1)
    hourly = hourly[hourly["hour_next"] == hourly["hour"] + 1].copy()
    return hourly.drop(columns=["hour_next"]).reset_index(drop=True)


def score_answers(cases: pd.DataFrame, answers: list[str]) -> pd.DataFrame:
    rows = []
    for i, case in cases.reset_index(drop=True).iterrows():
        answer = answers[i] if i < len(answers) else ""
        text = answer.lower()
        zone = str(case.get("expected_zone", "")).lower()
        tip = float(case.get("expected_tip", 0.0))
        must = str(case.get("must_include", "")).lower()
        number = f"{tip:.2f}"
        has_zone = int(bool(zone) and zone in text)
        has_number = int(number in text or f"${number}" in text)
        has_caveat = int(not must or must in text)
        rows.append(
            {
                "question": case.get("question", ""),
                "answer": answer,
                "has_zone": has_zone,
                "has_number": has_number,
                "has_caveat": has_caveat,
                "score": float(np.mean([has_zone, has_number, has_caveat])),
            }
        )
    return pd.DataFrame(rows)


def llm_examples(zones: pd.DataFrame, limit: int = 200) -> list[dict[str, object]]:
    if zones.empty:
        return []
    data = zones.copy().sort_values("expected_tip", ascending=False).head(limit).reset_index(drop=True)
    examples = []
    for _, row in data.iterrows():
        zone = str(row["pickup_zone"])
        tip = float(row["expected_tip"])
        q10 = float(row.get("q10_tip", tip))
        prob = float(row.get("predicted_tip_probability", row.get("tip_rate", 0.0)))
        rows = int(row.get("rows", 0))
        user = f"I can start in {zone}. Is this likely to be a good tipping area?"
        assistant = (
            f"{zone} is a reasonable option if the trip details are similar to the evaluation data. "
            f"The expected card tip is about ${tip:.2f}, the lower-tail estimate is ${q10:.2f}, "
            f"and the model estimates a {prob:.0%} chance of receiving a tip from {rows:,} matched rides. "
            "This excludes cash tips and should be used with wait time and pickup distance."
        )
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": "You are Tip-or-Skip, a driver-facing assistant grounded only in NYC taxi tip model outputs."},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )
    return examples


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
