from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip_or_skip.config import FINAL_ARTIFACT_DIR, FIGURE_DIR, REPORT_DIR, ensure_directories
from tip_or_skip.data import load_dataset
from tip_or_skip.maps import zone_risk_summary
from tip_or_skip.metrics import subgroup_classification_metrics


def _flatten_metrics() -> pd.DataFrame:
    rows = []
    baseline_path = FINAL_ARTIFACT_DIR / "baselines" / "baseline_metrics.json"
    if baseline_path.exists():
        baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
        for model, metrics in baseline_metrics.items():
            rows.append({"model": model, **metrics})

    deep_path = FINAL_ARTIFACT_DIR / "transformer_mdn" / "deep_metrics.json"
    if deep_path.exists():
        deep = json.loads(deep_path.read_text(encoding="utf-8"))
        rows.append({"model": "transformer_mdn", **deep["test_metrics"]})
    return pd.DataFrame(rows)


def _plot_metrics(metrics: pd.DataFrame) -> None:
    metric_map = {
        "class_roc_auc": "ROC-AUC",
        "class_brier": "Brier Score",
        "class_ece": "ECE",
        "logtip_rmse": "Log Tip RMSE",
        "expected_tip_mae": "Expected Tip MAE",
    }
    available = [column for column in metric_map if column in metrics.columns]
    fig, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for ax, column in zip(axes, available):
        ax.bar(metrics["model"], metrics[column], color=["#176b54", "#e09f3e", "#335c67"][: len(metrics)])
        ax.set_title(metric_map[column])
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def _plot_dataset_profiles() -> None:
    df = load_dataset(
        columns=[
            "taxi_type",
            "pickup_year",
            "pickup_month",
            "daypart",
            "time_split",
            "tip_given",
            "tip_amount",
            "fare_amount",
        ]
    )
    monthly = (
        df.groupby(["taxi_type", "pickup_year", "pickup_month"], observed=True)
        .agg(rows=("tip_given", "size"), tip_rate=("tip_given", "mean"), avg_tip=("tip_amount", "mean"))
        .reset_index()
    )
    monthly["year_month"] = monthly["pickup_year"].astype(str) + "-" + monthly["pickup_month"].astype(str).str.zfill(2)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for taxi_type, group in monthly.groupby("taxi_type"):
        ax.plot(group["year_month"], group["tip_rate"], marker="o", label=taxi_type.title())
    ax.set_title("Recorded Electronic Tip Rate by Month")
    ax.set_xlabel("Pickup month")
    ax.set_ylabel("Tip rate")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "monthly_tip_rate.png", dpi=180)
    plt.close(fig)

    split = df.groupby(["time_split", "taxi_type"], observed=True).size().reset_index(name="rows")
    split.to_csv(REPORT_DIR / "split_counts.csv", index=False)
    monthly.to_csv(REPORT_DIR / "monthly_profile_for_report.csv", index=False)


def _build_deep_subgroup_outputs() -> None:
    pred_path = FINAL_ARTIFACT_DIR / "transformer_mdn" / "deep_predictions_test.parquet"
    frame_path = FINAL_ARTIFACT_DIR / "transformer_mdn" / "deep_test_frame.parquet"
    if not pred_path.exists() or not frame_path.exists():
        return
    pred = pd.read_parquet(pred_path)
    frame = pd.read_parquet(frame_path).reset_index(drop=True)
    combined = pd.concat([frame, pred], axis=1)
    subgroup = subgroup_classification_metrics(
        combined,
        ["taxi_type", "pickup_borough"],
        min_rows=500,
    )
    subgroup.to_csv(REPORT_DIR / "subgroup_metrics.csv", index=False)
    zones = zone_risk_summary(frame, pred)
    zones.to_csv(REPORT_DIR / "zone_risk_summary.csv", index=False)
    top = zones[zones["pickup_borough"] == "Manhattan"].head(20)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(top["pickup_zone"], top["expected_tip"], color="#176b54")
    ax.invert_yaxis()
    ax.set_title("Top Manhattan Pickup Zones by Predicted Expected Tip")
    ax.set_xlabel("Expected tip ($)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "top_zone_expected_tip.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    metrics = _flatten_metrics()
    metrics.to_csv(REPORT_DIR / "final_metrics.csv", index=False)
    if not metrics.empty:
        _plot_metrics(metrics)
    _plot_dataset_profiles()
    _build_deep_subgroup_outputs()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

