from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

os.environ["GRADIO_SSR_MODE"] = "false"
os.environ.setdefault("MPLBACKEND", "Agg")

import folium
import gradio as gr
import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from prototype_pipeline import ARTIFACT_DIR, predict_tip

matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from tip_or_skip.driver_copilot import answer_driver_question, build_driver_context
from tip_or_skip.fact_assistant import answer_question, build_fact_context, fact_cards_markdown
from tip_or_skip.shift_planner import rank_destinations

TLC_ZONES_URL = (
    "https://data.cityofnewyork.us/api/views/8meu-9t5y/rows.geojson?accessType=DOWNLOAD"
)
_ZONE_CENTROIDS_CACHE: dict[str, tuple[float, float]] | None = None


def load_blog_background() -> str:
    root_blog_path = ROOT_DIR / "blog_background.md"
    if root_blog_path.exists():
        return root_blog_path.read_text(encoding="utf-8")
    return (ARTIFACT_DIR / "blog_background.md").read_text(encoding="utf-8")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_zone_centroids() -> dict[str, tuple[float, float]]:
    global _ZONE_CENTROIDS_CACHE
    if _ZONE_CENTROIDS_CACHE is not None:
        return _ZONE_CENTROIDS_CACHE

    with urllib.request.urlopen(TLC_ZONES_URL, timeout=30) as response:
        geojson = json.loads(response.read())

    centroids: dict[str, tuple[float, float]] = {}
    for feature in geojson["features"]:
        zone = feature["properties"]["zone"]
        geometry = feature["geometry"]
        rings: list[list] = []
        if geometry["type"] == "Polygon":
            rings = [geometry["coordinates"][0]]
        elif geometry["type"] == "MultiPolygon":
            rings = [polygon[0] for polygon in geometry["coordinates"]]

        points = [point for ring in rings for point in ring]
        if not points:
            continue

        lon = sum(point[0] for point in points) / len(points)
        lat = sum(point[1] for point in points) / len(points)
        centroids[zone] = (lat, lon)

    _ZONE_CENTROIDS_CACHE = centroids
    return centroids


def load_artifacts() -> dict:
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    packaged_summary_path = ARTIFACT_DIR / "final_report" / "build_summary.json"
    summary_path = (
        packaged_summary_path
        if packaged_summary_path.exists()
        else ROOT_DIR.parent / "final_dataset" / "build_summary.json"
    )
    final_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {
            "rows": 0,
            "columns": 0,
            "tip_rate": 0.0,
            "split_counts": {},
            "taxi_type_counts": {},
        }
    )
    monthly = pd.read_csv(ARTIFACT_DIR / "monthly_summary.csv")
    hourly = pd.read_csv(ARTIFACT_DIR / "hourly_summary.csv")
    zones = pd.read_csv(ARTIFACT_DIR / "zone_summary.csv")
    sample_rows = pd.read_csv(ARTIFACT_DIR / "sample_rows.csv")
    zone_options = pd.read_csv(ARTIFACT_DIR / "zone_options.csv")
    dataset_notes = (ARTIFACT_DIR / "dataset_notes.md").read_text(encoding="utf-8")
    blog_background = load_blog_background()
    models = {
        "yellow": joblib.load(ARTIFACT_DIR / "yellow_model_bundle.joblib"),
        "green": joblib.load(ARTIFACT_DIR / "green_model_bundle.joblib"),
    }
    final_dir = ARTIFACT_DIR / "final"
    packaged_report_dir = ARTIFACT_DIR / "final_report"
    report_dir = packaged_report_dir if packaged_report_dir.exists() else ROOT_DIR.parent / "report"
    final_metrics_path = report_dir / "final_metrics.csv"
    zone_risk_path = report_dir / "zone_risk_summary.csv"
    subgroup_path = report_dir / "subgroup_metrics.csv"
    monthly_profile_path = report_dir / "monthly_profile_for_report.csv"
    a_plus_dir = ARTIFACT_DIR / "a_plus"
    final_metrics = _read_csv(final_metrics_path)
    zone_risk = _read_csv(zone_risk_path)
    subgroup_metrics = _read_csv(subgroup_path)
    monthly_profile = _read_csv(monthly_profile_path)
    return {
        "metrics": metrics,
        "final_summary": final_summary,
        "monthly": monthly,
        "hourly": hourly,
        "zones": zones,
        "sample_rows": sample_rows,
        "zone_options": zone_options,
        "dataset_notes": dataset_notes,
        "blog_background": blog_background,
        "models": models,
        "final_dir": final_dir,
        "report_dir": report_dir,
        "final_metrics": final_metrics,
        "zone_risk": zone_risk,
        "subgroup_metrics": subgroup_metrics,
        "monthly_profile": monthly_profile,
        "a_plus_dir": a_plus_dir,
        "ablation_metrics": _read_csv(a_plus_dir / "ablation_metrics.csv"),
        "calibration_bins": _read_csv(a_plus_dir / "calibration_bins.csv"),
        "zone_flow_features": _read_csv(a_plus_dir / "zone_flow_features.csv"),
        "sequence_metrics": _read_json(a_plus_dir / "sequence_metrics.json"),
        "graph_metrics": _read_csv(a_plus_dir / "graph_metrics.csv"),
        "copilot_eval_summary": _read_csv(a_plus_dir / "copilot_eval_summary.csv"),
        "copilot_eval": _read_csv(a_plus_dir / "copilot_eval.csv"),
        "llm_finetune_metrics": _read_json(a_plus_dir / "llm_finetune_metrics.json"),
    }


ARTIFACTS = load_artifacts()
FACT_CONTEXT = build_fact_context(
    ARTIFACTS["final_summary"],
    ARTIFACTS["final_metrics"],
    ARTIFACTS["zone_risk"],
    ARTIFACTS["subgroup_metrics"],
)
DRIVER_CONTEXT = build_driver_context(
    ARTIFACTS["zone_risk"],
    ARTIFACTS["zone_options"]["zone"].tolist(),
)
ZONE_CHOICES = ARTIFACTS["zone_options"]["zone"].tolist()
DEFAULT_PICKUP = "Midtown Center"
DEFAULT_DROPOFF = "Upper East Side North"
WEEKDAY_CHOICES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _weekday_to_int(value: int | float | str) -> int:
    if isinstance(value, str):
        cleaned = value.strip()
        weekday_lookup = {name.lower(): idx for idx, name in enumerate(WEEKDAY_CHOICES)}
        if cleaned.lower() in weekday_lookup:
            return weekday_lookup[cleaned.lower()]
        return int(cleaned)
    return int(value)


def metrics_markdown() -> str:
    blocks = ["## Baseline Results", ""]
    for taxi_type, values in ARTIFACTS["metrics"].items():
        blocks.append(f"### {taxi_type.title()} taxi")
        blocks.append(f"- Test ROC-AUC: {values['roc_auc']:.3f}")
        blocks.append(f"- Test F1 @ 0.50: {values['f1_at_0_5']:.3f}")
        blocks.append(f"- Tip-rate in test split: {values['tip_rate_test']:.3f}")
        blocks.append(f"- Conditional tip RMSE on log scale: {values['rmse_log_tip']:.3f}")
        blocks.append("")
    blocks.append(
        "These models are trained on credit-card trips only because TLC `tip_amount` does not include cash tips."
    )
    return "\n".join(blocks)


def final_results_markdown() -> str:
    metrics = ARTIFACTS["final_metrics"]
    if metrics.empty:
        return "Final L4-trained artifacts have not been generated in this Space checkout yet."

    best = metrics.sort_values("expected_tip_mae").iloc[0]
    blocks = [
        "## Final 2024-2025 L4 Training Results",
        "",
        f"Best point-prediction model: **{best['model']}** with expected-tip MAE **${best['expected_tip_mae']:.2f}** on the held-out 2025 split.",
        "",
    ]
    for _, row in metrics.iterrows():
        blocks.append(f"### {row['model']}")
        blocks.append(f"- ROC-AUC: {row['class_roc_auc']:.3f}")
        blocks.append(f"- Brier score: {row['class_brier']:.3f}")
        blocks.append(f"- ECE: {row['class_ece']:.3f}")
        blocks.append(f"- Log-tip RMSE: {row['logtip_rmse']:.3f}")
        blocks.append(f"- Expected-tip MAE: ${row['expected_tip_mae']:.2f}")
        if "interval80_coverage" in row and pd.notna(row["interval80_coverage"]):
            blocks.append(f"- MDN 80% interval coverage: {row['interval80_coverage']:.3f}")
        blocks.append("")
    blocks.append("Dataset split: train Jan-Sep 2024, validation Oct-Dec 2024, test all 2025.")
    return "\n".join(blocks)


def _call_optional_hf_llm(question: str, grounded_answer: str, persona: str = "data-science assistant") -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_id = os.getenv("HF_INFERENCE_MODEL", "").strip()
    if not token or not model_id:
        return grounded_answer

    prompt = (
        f"You are a concise {persona} for an Applied Machine Learning project about NYC taxi tipping. "
        "Answer only from the grounded facts below. If the facts do not support a claim, say so.\n\n"
        f"Question: {question}\n\n"
        f"Grounded facts:\n{grounded_answer}\n\n"
        "Answer:"
    )
    request = urllib.request.Request(
        f"https://api-inference.huggingface.co/models/{model_id}",
        data=json.dumps(
            {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 220, "temperature": 0.2, "return_full_text": False},
            }
        ).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.loads(response.read())
    except Exception:
        return grounded_answer
    if isinstance(payload, list) and payload and "generated_text" in payload[0]:
        generated = str(payload[0]["generated_text"]).strip()
        return generated or grounded_answer
    if isinstance(payload, dict) and "generated_text" in payload:
        generated = str(payload["generated_text"]).strip()
        return generated or grounded_answer
    return grounded_answer


def tipping_assistant(message: str, history):
    grounded = answer_question(message, FACT_CONTEXT)
    answer = _call_optional_hf_llm(message, grounded)
    return answer + "\n\n_Source: frozen project dataset, final model metrics, and report artifacts._"


def driver_copilot(message: str, history):
    grounded = answer_driver_question(message, DRIVER_CONTEXT)
    answer = _call_optional_hf_llm(message, grounded, persona="driver-facing ride-planning copilot")
    return answer + "\n\n_Source: final model zone-risk table and TLC credit-card tip artifacts._"


def compare_ride_options(
    taxi_type: str,
    current_zone: str,
    option_a_zone: str,
    option_b_zone: str,
    pickup_hour: int,
    pickup_weekday: int | str,
    pickup_month: int,
    option_a_distance: float,
    option_a_fare: float,
    option_a_duration: float,
    option_b_distance: float,
    option_b_fare: float,
    option_b_duration: float,
):
    option_specs = [
        ("Option A", option_a_zone, option_a_distance, option_a_fare, option_a_duration),
        ("Option B", option_b_zone, option_b_distance, option_b_fare, option_b_duration),
    ]
    rows = []
    for label, dropoff_zone, distance, fare, duration in option_specs:
        feature_row = _build_feature_row(
            taxi_type=taxi_type,
            pickup_zone=current_zone,
            dropoff_zone=dropoff_zone,
            pickup_hour=pickup_hour,
            pickup_weekday=_weekday_to_int(pickup_weekday),
            pickup_month=pickup_month,
            trip_distance=distance,
            fare_amount=fare,
            trip_duration_minutes=duration,
            vendor_id="1",
            passenger_bucket="1",
            ratecode="1",
            store_and_fwd_flag="N",
        )
        prediction = predict_tip(ARTIFACTS["models"][taxi_type], feature_row)
        rows.append(
            {
                "Ride option": label,
                "Pickup area": current_zone,
                "Dropoff area": dropoff_zone,
                "Tip probability": prediction["tip_probability"],
                "Conditional tip": prediction["conditional_tip"],
                "Expected tip": prediction["expected_tip"],
                "Fare": fare,
                "Distance": distance,
                "Duration": duration,
            }
        )
    table = pd.DataFrame(rows).sort_values("Expected tip", ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    other = table.iloc[1]
    gap = float(best["Expected tip"]) - float(other["Expected tip"])
    summary = (
        f"### Recommendation\n"
        f"Choose **{best['Ride option']} to {best['Dropoff area']}** for the higher expected electronic tip.\n\n"
        f"- Expected tip: **${best['Expected tip']:.2f}**\n"
        f"- Tip probability: **{best['Tip probability']:.1%}**\n"
        f"- Gap over the other option: **${gap:.2f}** expected tip\n\n"
        "This compares recorded electronic tips only, using the deployed two-stage trip prediction model."
    )
    return summary, table.round(4)


def model_comparison_table(sort_by: str):
    metrics = ARTIFACTS["final_metrics"]
    if metrics.empty:
        return pd.DataFrame({"message": ["Final metrics are not available."]})
    sort_map = {
        "Expected Tip MAE": ("expected_tip_mae", True),
        "ROC-AUC": ("class_roc_auc", False),
        "Calibration Error": ("class_ece", True),
        "Log Tip RMSE": ("logtip_rmse", True),
    }
    column, ascending = sort_map[sort_by]
    display_cols = [
        "model",
        "class_roc_auc",
        "class_log_loss",
        "class_brier",
        "class_ece",
        "class_f1",
        "logtip_rmse",
        "expected_tip_mae",
        "interval80_coverage",
    ]
    available = [col for col in display_cols if col in metrics.columns]
    out = metrics.sort_values(column, ascending=ascending)[available].copy()
    return out.round(4)


def plot_final_metric(metric_label: str):
    metrics = ARTIFACTS["final_metrics"]
    metric_map = {
        "Expected Tip MAE": ("expected_tip_mae", "Lower is better", "$"),
        "ROC-AUC": ("class_roc_auc", "Higher is better", ""),
        "Calibration Error": ("class_ece", "Lower is better", ""),
        "Log Tip RMSE": ("logtip_rmse", "Lower is better", ""),
        "F1": ("class_f1", "Higher is better", ""),
    }
    column, subtitle, prefix = metric_map[metric_label]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#176b54", "#e09f3e", "#335c67"]
    ax.bar(metrics["model"], metrics[column], color=colors[: len(metrics)])
    ax.set_title(f"{metric_label} ({subtitle})")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    if prefix == "$":
        ax.set_ylabel("Dollars")
    fig.tight_layout()
    return fig


def final_monthly_plot(taxi_type: str, metric_label: str):
    monthly = ARTIFACTS["monthly_profile"]
    if monthly.empty:
        return plot_monthly_trends(taxi_type)
    metric_col = "tip_rate" if metric_label == "Tip Rate" else "avg_tip"
    subset = monthly[monthly["taxi_type"] == taxi_type].sort_values(["pickup_year", "pickup_month"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(subset["year_month"], subset[metric_col], marker="o", linewidth=2.2, color="#335c67")
    ax.set_title(f"{taxi_type.title()} Taxi {metric_label} Across the Frozen Dataset")
    ax.set_xlabel("Pickup month")
    ax.set_ylabel("Tip rate" if metric_col == "tip_rate" else "Average tip ($)")
    ax.tick_params(axis="x", rotation=50)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def subgroup_metrics_table(min_rows: int):
    subgroup = ARTIFACTS["subgroup_metrics"]
    if subgroup.empty:
        return pd.DataFrame({"message": ["Subgroup metrics are not available."]})
    subset = subgroup[subgroup["rows"] >= int(min_rows)].copy()
    subset = subset.sort_values(["rows"], ascending=False)
    return subset.round(4)


def a_plus_markdown() -> str:
    blocks = ["## A+ Stretch Experiment Artifacts", ""]
    ablation = ARTIFACTS["ablation_metrics"]
    if not ablation.empty:
        best = ablation.sort_values("expected_tip_mae").iloc[0]
        blocks.append(f"Best ablation run: **{best['ablation']}** with expected-tip MAE **${best['expected_tip_mae']:.2f}**.")
    sequence = ARTIFACTS["sequence_metrics"]
    if sequence:
        blocks.append(
            f"Sequence LSTM: next-hour average-tip MAE **${sequence.get('avg_tip_mae', 0):.2f}** "
            f"over **{int(sequence.get('test_rows', 0)):,}** test windows."
        )
    graph = ARTIFACTS["graph_metrics"]
    if not graph.empty:
        row = graph.iloc[0]
        blocks.append(
            f"Graph model: test zone-tip MAE **${row.get('test_tip_mae', 0):.2f}**, "
            f"naive temporal baseline **${row.get('naive_test_tip_mae', 0):.2f}**."
        )
    copilot = ARTIFACTS["copilot_eval_summary"]
    if not copilot.empty:
        overall = copilot[copilot["check"] == "overall"]
        if not overall.empty:
            blocks.append(f"Driver Copilot grounding score: **{float(overall.iloc[0]['pass_rate']):.0%}** on the scripted ride-choice checks.")
    llm = ARTIFACTS["llm_finetune_metrics"]
    if llm:
        blocks.append(
            f"Fine-tuned driver LLM artifact: **{llm.get('base_model', 'model')}**, "
            f"eval perplexity **{llm.get('eval_perplexity', 0):.2f}**, zone mention rate **{llm.get('zone_mention_rate', 0):.0%}**."
        )
    if len(blocks) == 2:
        blocks.append("Run `python scripts/run_extra_analysis.py`, `python scripts/train_sequence_model.py`, and `python scripts/train_graph_model.py` to generate these artifacts.")
    return "\n\n".join(blocks)


def metric_json_table(name: str) -> pd.DataFrame:
    data = ARTIFACTS[name]
    if not data:
        return pd.DataFrame({"message": ["Metric file is not available."]})
    return pd.DataFrame([data]).round(4)


def run_sensitivity(
    taxi_type: str,
    pickup_zone: str,
    dropoff_zone: str,
    pickup_month: int,
    pickup_weekday: int | str,
    base_hour: int,
    base_distance: float,
    base_fare: float,
    base_duration: float,
    sweep_by: str,
):
    if sweep_by == "Pickup hour":
        values = list(range(24))
    elif sweep_by == "Fare amount":
        values = [round(v, 2) for v in list(pd.Series([base_fare * x for x in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]]))]
    elif sweep_by == "Trip distance":
        values = [round(v, 2) for v in list(pd.Series([base_distance * x for x in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]]))]
    else:
        values = [round(v, 2) for v in list(pd.Series([base_duration * x for x in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]]))]

    rows = []
    for value in values:
        hour = int(value) if sweep_by == "Pickup hour" else int(base_hour)
        distance = float(value) if sweep_by == "Trip distance" else float(base_distance)
        fare = float(value) if sweep_by == "Fare amount" else float(base_fare)
        duration = float(value) if sweep_by == "Trip duration" else float(base_duration)
        feature_row = _build_feature_row(
            taxi_type=taxi_type,
            pickup_zone=pickup_zone,
            dropoff_zone=dropoff_zone,
            pickup_hour=hour,
            pickup_weekday=pickup_weekday,
            pickup_month=pickup_month,
            trip_distance=distance,
            fare_amount=fare,
            trip_duration_minutes=duration,
            vendor_id="1",
            passenger_bucket="1",
            ratecode="1",
            store_and_fwd_flag="N",
        )
        prediction = predict_tip(ARTIFACTS["models"][taxi_type], feature_row)
        rows.append(
            {
                sweep_by: value,
                "Tip Probability": prediction["tip_probability"],
                "Conditional Tip": prediction["conditional_tip"],
                "Expected Tip": prediction["expected_tip"],
            }
        )
    table = pd.DataFrame(rows)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(table[sweep_by], table["Expected Tip"], marker="o", color="#176b54", label="Expected tip")
    ax1.set_xlabel(sweep_by)
    ax1.set_ylabel("Expected tip ($)")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(table[sweep_by], table["Tip Probability"], marker="s", color="#e09f3e", label="Tip probability")
    ax2.set_ylabel("Tip probability")
    ax1.set_title(f"What-if sensitivity: {sweep_by}")
    fig.tight_layout()
    return fig, table.round(4)


def _build_feature_row(
    taxi_type: str,
    pickup_zone: str,
    dropoff_zone: str,
    pickup_hour: int,
    pickup_weekday: int,
    pickup_month: int,
    trip_distance: float,
    fare_amount: float,
    trip_duration_minutes: float,
    vendor_id: str,
    passenger_bucket: str,
    ratecode: str,
    store_and_fwd_flag: str,
) -> dict:
    lookup = ARTIFACTS["zone_options"].rename(
        columns={"zone": "pickup_zone", "borough": "pickup_borough"}
    )

    p_borough_match = lookup.loc[lookup["pickup_zone"] == pickup_zone, "pickup_borough"]
    pickup_borough = p_borough_match.iloc[0] if not p_borough_match.empty else "Unknown"

    d_borough_match = lookup.loc[lookup["pickup_zone"] == dropoff_zone, "pickup_borough"]
    dropoff_borough = d_borough_match.iloc[0] if not d_borough_match.empty else "Unknown"

    return {
        "pickup_hour": int(pickup_hour),
        "pickup_weekday": _weekday_to_int(pickup_weekday),
        "pickup_month": int(pickup_month),
        "trip_distance": float(trip_distance),
        "fare_amount": float(fare_amount),
        "trip_duration_minutes": float(trip_duration_minutes),
        "vendor_id": str(vendor_id),
        "passenger_bucket": str(passenger_bucket),
        "ratecode": str(ratecode),
        "store_and_fwd_flag": str(store_and_fwd_flag),
        "pickup_borough": pickup_borough,
        "pickup_zone": pickup_zone,
        "dropoff_borough": dropoff_borough,
        "dropoff_zone": dropoff_zone,
    }


def run_prediction(
    taxi_type: str,
    pickup_zone: str,
    dropoff_zone: str,
    pickup_hour: int,
    pickup_weekday: int,
    pickup_month: int,
    trip_distance: float,
    fare_amount: float,
    trip_duration_minutes: float,
    vendor_id: str,
    passenger_bucket: str,
    ratecode: str,
    store_and_fwd_flag: str,
):
    feature_row = _build_feature_row(
        taxi_type=taxi_type,
        pickup_zone=pickup_zone,
        dropoff_zone=dropoff_zone,
        pickup_hour=pickup_hour,
        pickup_weekday=pickup_weekday,
        pickup_month=pickup_month,
        trip_distance=trip_distance,
        fare_amount=fare_amount,
        trip_duration_minutes=trip_duration_minutes,
        vendor_id=vendor_id,
        passenger_bucket=passenger_bucket,
        ratecode=ratecode,
        store_and_fwd_flag=store_and_fwd_flag,
    )
    prediction = predict_tip(ARTIFACTS["models"][taxi_type], feature_row)
    summary = (
        f"Estimated chance of a recorded electronic tip: {prediction['tip_probability']:.1%}\n\n"
        f"Predicted tip amount if a tip happens: ${prediction['conditional_tip']:.2f}\n\n"
        f"Expected tip value for this ride: ${prediction['expected_tip']:.2f}"
    )
    detail = pd.DataFrame(
        [
            {"metric": "Tip probability", "value": round(prediction["tip_probability"], 4)},
            {
                "metric": "Conditional tip amount",
                "value": round(prediction["conditional_tip"], 2),
            },
            {"metric": "Expected tip amount", "value": round(prediction["expected_tip"], 2)},
        ]
    )
    return summary, detail


def plot_monthly_trends(taxi_type: str):
    df = ARTIFACTS["monthly"]
    subset = df[df["taxi_type"] == taxi_type].sort_values("pickup_month")
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(subset["pickup_month"], subset["tip_rate"], marker="o", linewidth=2, color="#0b6e4f")
    ax1.set_title(f"{taxi_type.title()} Taxi: Sampled Monthly Tip Rate")
    ax1.set_xlabel("Month of 2025")
    ax1.set_ylabel("Tip rate")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.2)
    return fig


def plot_hourly_trends(taxi_type: str):
    df = ARTIFACTS["hourly"]
    subset = df[df["taxi_type"] == taxi_type].sort_values("pickup_hour")
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(subset["pickup_hour"], subset["avg_tip_amount"], color="#f2a541")
    ax1.set_title(f"{taxi_type.title()} Taxi: Average Recorded Tip by Pickup Hour")
    ax1.set_xlabel("Pickup hour")
    ax1.set_ylabel("Average tip amount ($)")
    ax1.grid(axis="y", alpha=0.2)
    return fig


def top_zones_table(taxi_type: str):
    subset = ARTIFACTS["zones"]
    subset = subset[subset["taxi_type"] == taxi_type].copy()
    subset = subset[subset["trips"] >= 100]
    subset = subset.head(15)[
        ["pickup_borough", "pickup_zone", "trips", "tip_rate", "avg_tip_amount"]
    ]
    subset["tip_rate"] = subset["tip_rate"].round(3)
    subset["avg_tip_amount"] = subset["avg_tip_amount"].round(2)
    return subset


def _manhattan_zones(taxi_type: str) -> pd.DataFrame:
    zones_df = ARTIFACTS["zones"]
    subset = zones_df[
        (zones_df["taxi_type"] == taxi_type) & (zones_df["pickup_borough"] == "Manhattan")
    ].copy()
    return subset.sort_values("trips", ascending=False)


def build_nyc_map(taxi_type: str, metric: str) -> str:
    metric_col = "tip_rate" if metric == "Tip Rate" else "avg_tip_amount"
    zones = _manhattan_zones(taxi_type)
    centroids = _load_zone_centroids()

    fig = folium.Figure(width="100%", height="480px")
    nyc_map = folium.Map(
        location=[40.754, -73.984],
        zoom_start=12,
        tiles="CartoDB positron",
    )
    fig.add_child(nyc_map)

    valid = zones[zones["pickup_zone"].isin(centroids)]
    if valid.empty:
        return fig._repr_html_()

    max_val = valid[metric_col].max()
    min_val = valid[metric_col].min()
    val_range = max(max_val - min_val, 1e-6)

    for _, row in valid.iterrows():
        zone = row["pickup_zone"]
        lat, lon = centroids[zone]
        value = row[metric_col]
        t = (value - min_val) / val_range

        r = int(30 + 80 * (1 - t))
        g = int(110 + 130 * t)
        b = int(200 - 150 * t)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        display_val = f"{value:.1%}" if metric_col == "tip_rate" else f"${value:.2f}"
        radius = int(8 + 12 * t)

        popup_html = (
            f"<b>{zone}</b><br>"
            f"Trips sampled: {int(row['trips']):,}<br>"
            f"Avg tip rate: {row['tip_rate']:.1%}<br>"
            f"Avg tip amount: ${row['avg_tip_amount']:.2f}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#ffffff",
            weight=1.5,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"<b>{zone}</b>: {display_val}",
        ).add_to(nyc_map)

    return fig._repr_html_()


def build_map_outputs(taxi_type: str, metric: str):
    metric_col = "tip_rate" if metric == "Tip Rate" else "avg_tip_amount"
    zones = _manhattan_zones(taxi_type).sort_values(metric_col, ascending=False)
    table = zones[["pickup_zone", "trips", "tip_rate", "avg_tip_amount"]].rename(
        columns={
            "pickup_zone": "Zone",
            "trips": "Trips",
            "tip_rate": "Tip Rate",
            "avg_tip_amount": "Avg Tip ($)",
        }
    )
    table["Tip Rate"] = table["Tip Rate"].round(3)
    table["Avg Tip ($)"] = table["Avg Tip ($)"].round(2)

    try:
        map_html = build_nyc_map(taxi_type, metric)
    except Exception as exc:
        map_html = (
            "<div style='padding:1rem;border:1px solid #ddd;border-radius:12px;'>"
            "<strong>Map unavailable.</strong><br>"
            f"Could not load NYC zone geometry: {exc}"
            "</div>"
        )

    return map_html, table


def final_zone_rankings(taxi_type: str, borough: str, objective: str, top_k: int):
    zone_risk = ARTIFACTS["zone_risk"]
    if zone_risk.empty:
        return pd.DataFrame({"message": ["Final zone risk summary is not available."]})
    subset = zone_risk[
        (zone_risk["taxi_type"] == taxi_type) & (zone_risk["pickup_borough"] == borough)
    ].copy()
    if subset.empty:
        return pd.DataFrame({"message": ["No zones match the selected filters."]})
    subset = subset.rename(columns={"pickup_zone": "dropoff_zone", "rows": "observed_trips"})
    ranked = rank_destinations(subset, objective=objective, top_k=int(top_k))
    return ranked[
        [
            "dropoff_zone",
            "score",
            "expected_tip",
            "q10_tip",
            "predicted_tip_probability",
            "observed_trips",
        ]
    ].rename(
        columns={
            "dropoff_zone": "Zone",
            "score": "Score",
            "expected_tip": "Expected Tip",
            "q10_tip": "Downside Q10 Tip",
            "predicted_tip_probability": "Predicted Tip Probability",
            "observed_trips": "Observed Trips",
        }
    )


INITIAL_MONTHLY_PLOT = plot_monthly_trends("yellow")
INITIAL_HOURLY_PLOT = plot_hourly_trends("yellow")
INITIAL_ZONE_TABLE = top_zones_table("yellow")
INITIAL_MAP_HTML, INITIAL_MAP_TABLE = build_map_outputs("yellow", "Tip Rate")


with gr.Blocks(title="NYC Taxi Tip Prototype") as demo:
    gr.Markdown(
        """
        # Tip or Skip
        NYC taxi tipping prediction with uncertainty-aware machine learning on TLC Yellow and Green taxi records.
        """
    )

    with gr.Tab("Overview"):
        with gr.Row():
            gr.Markdown(metrics_markdown())
            gr.Markdown(fact_cards_markdown(FACT_CONTEXT))
        with gr.Row():
            gr.Markdown(ARTIFACTS["dataset_notes"])
        gr.Dataframe(
            value=ARTIFACTS["sample_rows"].head(20),
            label="Sampled cleaned rows used for the prototype",
            interactive=False,
        )

    with gr.Tab("Ask The Data"):
        gr.Markdown("## Grounded Tipping Facts Assistant")
        gr.ChatInterface(
            fn=tipping_assistant,
            examples=[
                "What dataset did we use?",
                "Which model performed best and why?",
                "What are the highest expected-tip zones in Queens?",
                "What does the Transformer-MDN add?",
                "What are the main limitations of predicting tips from TLC data?",
            ],
            title=None,
            description=None,
            type="messages",
        )

    with gr.Tab("Driver Copilot"):
        gr.Markdown(
            "## Driver-Facing Ride Planner\n"
            "Ask ride-choice questions in plain language, or compare two specific trip options with structured inputs. "
            "The copilot is grounded in the final model's expected-tip, downside-risk, and tip-probability outputs."
        )
        gr.ChatInterface(
            fn=driver_copilot,
            examples=[
                "I'm at Midtown Center and got two ride options: JFK Airport or LaGuardia Airport. Which should I choose?",
                "Is Battery Park City likely to earn me a good tip tonight?",
                "I am near Queens and can wait for JFK Airport or LaGuardia Airport. Which area has better tip upside?",
                "Which looks better for a yellow taxi: Red Hook or JFK Airport?",
            ],
            title=None,
            description=None,
            type="messages",
        )
        gr.Markdown("## Structured Two-Ride Comparison")
        with gr.Row():
            driver_taxi_type = gr.Dropdown(["yellow", "green"], value="yellow", label="Taxi type")
            driver_current_zone = gr.Dropdown(ZONE_CHOICES, value=DEFAULT_PICKUP, label="Current pickup area")
        with gr.Row():
            driver_hour = gr.Slider(0, 23, value=18, step=1, label="Pickup hour")
            driver_weekday = gr.Dropdown(WEEKDAY_CHOICES, value="Friday", label="Pickup weekday")
            driver_month = gr.Slider(1, 12, value=6, step=1, label="Pickup month")
        with gr.Row():
            option_a_zone = gr.Dropdown(ZONE_CHOICES, value="JFK Airport", label="Option A dropoff area")
            option_b_zone = gr.Dropdown(ZONE_CHOICES, value="LaGuardia Airport", label="Option B dropoff area")
        with gr.Row():
            option_a_distance = gr.Slider(0.1, 35.0, value=16.0, step=0.1, label="Option A distance")
            option_a_fare = gr.Slider(3.0, 150.0, value=58.0, step=0.5, label="Option A fare")
            option_a_duration = gr.Slider(1.0, 150.0, value=45.0, step=1.0, label="Option A duration")
        with gr.Row():
            option_b_distance = gr.Slider(0.1, 35.0, value=9.0, step=0.1, label="Option B distance")
            option_b_fare = gr.Slider(3.0, 150.0, value=38.0, step=0.5, label="Option B fare")
            option_b_duration = gr.Slider(1.0, 150.0, value=28.0, step=1.0, label="Option B duration")
        compare_button = gr.Button("Compare ride options", variant="primary")
        driver_summary = gr.Markdown()
        driver_table = gr.Dataframe(interactive=False, label="Ride comparison")
        compare_button.click(
            fn=compare_ride_options,
            inputs=[
                driver_taxi_type,
                driver_current_zone,
                option_a_zone,
                option_b_zone,
                driver_hour,
                driver_weekday,
                driver_month,
                option_a_distance,
                option_a_fare,
                option_a_duration,
                option_b_distance,
                option_b_fare,
                option_b_duration,
            ],
            outputs=[driver_summary, driver_table],
        )

    with gr.Tab("Predict"):
        gr.Markdown(
            "Use the form below to estimate tip behavior for a hypothetical **credit-card** trip. "
            "The model predicts recorded electronic tips only."
        )
        with gr.Row():
            taxi_type = gr.Dropdown(["yellow", "green"], value="yellow", label="Taxi type")
            pickup_zone = gr.Dropdown(ZONE_CHOICES, value=DEFAULT_PICKUP, label="Pickup zone")
            dropoff_zone = gr.Dropdown(
                ZONE_CHOICES, value=DEFAULT_DROPOFF, label="Dropoff zone"
            )
        with gr.Row():
            pickup_hour = gr.Slider(0, 23, value=18, step=1, label="Pickup hour")
            pickup_weekday = gr.Slider(0, 6, value=4, step=1, label="Pickup weekday (0=Mon)")
            pickup_month = gr.Slider(1, 12, value=6, step=1, label="Pickup month")
        with gr.Row():
            trip_distance = gr.Slider(
                0.1, 30.0, value=3.4, step=0.1, label="Trip distance (miles)"
            )
            fare_amount = gr.Slider(3.0, 120.0, value=18.0, step=0.5, label="Fare amount ($)")
            trip_duration_minutes = gr.Slider(
                1.0, 120.0, value=16.0, step=1.0, label="Trip duration (minutes)"
            )
        with gr.Row():
            vendor_id = gr.Dropdown(["1", "2", "6", "7"], value="1", label="Vendor")
            passenger_bucket = gr.Dropdown(
                ["Unknown", "1", "2", "3", "4", "5", "6+"],
                value="1",
                label="Passenger count bucket",
            )
            ratecode = gr.Dropdown(["1", "2", "3", "4", "5", "6", "99"], value="1", label="Rate code")
            store_and_fwd_flag = gr.Dropdown(
                ["N", "Y", "Unknown"],
                value="N",
                label="Store and forward flag",
            )
        predict_button = gr.Button("Predict tip outcome", variant="primary")
        prediction_text = gr.Markdown()
        prediction_table = gr.Dataframe(interactive=False, label="Prediction details")
        predict_button.click(
            fn=run_prediction,
            inputs=[
                taxi_type,
                pickup_zone,
                dropoff_zone,
                pickup_hour,
                pickup_weekday,
                pickup_month,
                trip_distance,
                fare_amount,
                trip_duration_minutes,
                vendor_id,
                passenger_bucket,
                ratecode,
                store_and_fwd_flag,
            ],
            outputs=[prediction_text, prediction_table],
        )
        gr.Markdown("## What-if Sensitivity")
        with gr.Row():
            sweep_by = gr.Dropdown(
                ["Pickup hour", "Fare amount", "Trip distance", "Trip duration"],
                value="Pickup hour",
                label="Sweep variable",
            )
            sweep_button = gr.Button("Run sensitivity sweep", variant="primary")
        with gr.Row():
            sensitivity_plot = gr.Plot(label="Sensitivity curve")
            sensitivity_table = gr.Dataframe(label="Sensitivity table", interactive=False)
        sweep_button.click(
            fn=run_sensitivity,
            inputs=[
                taxi_type,
                pickup_zone,
                dropoff_zone,
                pickup_month,
                pickup_weekday,
                pickup_hour,
                trip_distance,
                fare_amount,
                trip_duration_minutes,
                sweep_by,
            ],
            outputs=[sensitivity_plot, sensitivity_table],
        )

    with gr.Tab("Explore"):
        gr.Markdown("Explore precomputed summaries from the sampled 2025 prototype dataset.")
        with gr.Row():
            taxi_type_chart = gr.Dropdown(["yellow", "green"], value="yellow", label="Taxi type")
            refresh_button = gr.Button("Refresh charts")
        with gr.Row():
            monthly_plot = gr.Plot(value=INITIAL_MONTHLY_PLOT, label="Monthly trend")
            hourly_plot = gr.Plot(value=INITIAL_HOURLY_PLOT, label="Hourly trend")
        zone_table = gr.Dataframe(
            value=INITIAL_ZONE_TABLE,
            label="Top pickup zones by tip rate",
            interactive=False,
        )
        refresh_button.click(plot_monthly_trends, inputs=taxi_type_chart, outputs=monthly_plot)
        refresh_button.click(plot_hourly_trends, inputs=taxi_type_chart, outputs=hourly_plot)
        refresh_button.click(top_zones_table, inputs=taxi_type_chart, outputs=zone_table)
        taxi_type_chart.change(plot_monthly_trends, inputs=taxi_type_chart, outputs=monthly_plot)
        taxi_type_chart.change(plot_hourly_trends, inputs=taxi_type_chart, outputs=hourly_plot)
        taxi_type_chart.change(top_zones_table, inputs=taxi_type_chart, outputs=zone_table)

    with gr.Tab("Model Lab"):
        gr.Markdown("## Final Model Comparison")
        with gr.Row():
            model_sort = gr.Dropdown(
                ["Expected Tip MAE", "ROC-AUC", "Calibration Error", "Log Tip RMSE"],
                value="Expected Tip MAE",
                label="Sort by",
            )
            model_metric = gr.Dropdown(
                ["Expected Tip MAE", "ROC-AUC", "Calibration Error", "Log Tip RMSE", "F1"],
                value="Expected Tip MAE",
                label="Metric chart",
            )
        with gr.Row():
            model_table = gr.Dataframe(
                value=model_comparison_table("Expected Tip MAE"),
                label="Held-out 2025 model metrics",
                interactive=False,
            )
            model_plot = gr.Plot(value=plot_final_metric("Expected Tip MAE"), label="Metric chart")
        model_sort.change(model_comparison_table, inputs=model_sort, outputs=model_table)
        model_metric.change(plot_final_metric, inputs=model_metric, outputs=model_plot)

        gr.Markdown("## Frozen Dataset Time Profile")
        with gr.Row():
            final_month_taxi = gr.Dropdown(["yellow", "green"], value="yellow", label="Taxi type")
            final_month_metric = gr.Dropdown(["Tip Rate", "Average Tip"], value="Tip Rate", label="Metric")
        final_month_plot = gr.Plot(value=final_monthly_plot("yellow", "Tip Rate"), label="Monthly profile")
        final_month_taxi.change(
            final_monthly_plot,
            inputs=[final_month_taxi, final_month_metric],
            outputs=final_month_plot,
        )
        final_month_metric.change(
            final_monthly_plot,
            inputs=[final_month_taxi, final_month_metric],
            outputs=final_month_plot,
        )

        gr.Markdown("## Borough Subgroup Evaluation")
        subgroup_min_rows = gr.Slider(500, 200000, value=500, step=500, label="Minimum test rows")
        subgroup_table = gr.Dataframe(
            value=subgroup_metrics_table(500),
            label="Transformer-MDN subgroup metrics",
            interactive=False,
        )
        subgroup_min_rows.change(subgroup_metrics_table, inputs=subgroup_min_rows, outputs=subgroup_table)

    with gr.Tab("A+ Lab"):
        gr.Markdown(a_plus_markdown())
        with gr.Row():
            gr.Dataframe(
                value=ARTIFACTS["ablation_metrics"].round(4) if not ARTIFACTS["ablation_metrics"].empty else pd.DataFrame({"message": ["Ablation metrics not generated."]}),
                label="Feature ablations",
                interactive=False,
            )
            gr.Dataframe(
                value=ARTIFACTS["calibration_bins"].round(4) if not ARTIFACTS["calibration_bins"].empty else pd.DataFrame({"message": ["Calibration bins not generated."]}),
                label="Calibration bins",
                interactive=False,
            )
        with gr.Row():
            gr.Dataframe(
                value=metric_json_table("sequence_metrics"),
                label="Sequence LSTM metrics",
                interactive=False,
            )
            gr.Dataframe(
                value=ARTIFACTS["graph_metrics"].round(4) if not ARTIFACTS["graph_metrics"].empty else pd.DataFrame({"message": ["Graph metrics not generated."]}),
                label="Graph flow model metrics",
                interactive=False,
            )
        with gr.Row():
            gr.Dataframe(
                value=ARTIFACTS["copilot_eval_summary"].round(4) if not ARTIFACTS["copilot_eval_summary"].empty else pd.DataFrame({"message": ["Copilot evaluation not generated."]}),
                label="Driver Copilot grounding checks",
                interactive=False,
            )
            gr.Dataframe(
                value=metric_json_table("llm_finetune_metrics"),
                label="Driver LLM fine-tune metrics",
                interactive=False,
            )
        extra_files = [
            ARTIFACTS["a_plus_dir"] / "llm_train.jsonl",
            ARTIFACTS["a_plus_dir"] / "llm_eval.jsonl",
            ARTIFACTS["a_plus_dir"] / "copilot_eval.csv",
        ]
        for extra_file in extra_files:
            if extra_file.exists():
                gr.File(value=str(extra_file), label=extra_file.name)
        for figure_name in [
            "ablation_expected_tip_mae.png",
            "calibration_curve.png",
            "zone_flow_graph_summary.png",
            "sequence_shift_signal.png",
            "sequence_lstm_training.png",
            "graph_gcn_zone_prediction.png",
            "copilot_eval_summary.png",
        ]:
            figure_path = ARTIFACTS["report_dir"] / "figures" / figure_name
            if figure_path.exists():
                gr.Image(value=str(figure_path), label=figure_name)

    with gr.Tab("Maps"):
        gr.Markdown(
            "## Manhattan Zone Tip Map\n"
            "Per-zone tipping patterns across Manhattan neighborhoods. Circle size and color reflect the selected metric."
        )
        with gr.Row():
            map_taxi_type = gr.Dropdown(["yellow", "green"], value="yellow", label="Taxi type")
            map_metric = gr.Dropdown(
                ["Tip Rate", "Avg Tip Amount"],
                value="Tip Rate",
                label="Metric",
            )
        render_map_button = gr.Button("Render Map", variant="primary")
        map_display = gr.HTML(value=INITIAL_MAP_HTML, label="NYC Tip Map")
        map_table = gr.Dataframe(value=INITIAL_MAP_TABLE, label="Manhattan zones", interactive=False)
        render_map_button.click(
            fn=build_map_outputs,
            inputs=[map_taxi_type, map_metric],
            outputs=[map_display, map_table],
        )
        map_taxi_type.change(
            fn=build_map_outputs,
            inputs=[map_taxi_type, map_metric],
            outputs=[map_display, map_table],
        )
        map_metric.change(
            fn=build_map_outputs,
            inputs=[map_taxi_type, map_metric],
            outputs=[map_display, map_table],
        )

    with gr.Tab("Final Results"):
        gr.Markdown(final_results_markdown())
        report_html = ARTIFACTS["report_dir"] / "index.html"
        if report_html.exists():
            gr.File(value=str(report_html), label="Offline technical blog index.html")
            gr.HTML(value=report_html.read_text(encoding="utf-8"))
        report_pdf = ARTIFACTS["report_dir"] / "Tip_or_Skip_Final_Report.pdf"
        if report_pdf.exists():
            gr.File(value=str(report_pdf), label="Final report PDF")
        figure_paths = [
            ARTIFACTS["report_dir"] / "figures" / "model_comparison.png",
            ARTIFACTS["report_dir"] / "figures" / "monthly_tip_rate.png",
            ARTIFACTS["report_dir"] / "figures" / "top_zone_expected_tip.png",
        ]
        for figure_path in figure_paths:
            if figure_path.exists():
                gr.Image(value=str(figure_path), show_label=False)

    with gr.Tab("Shift Planner"):
        gr.Markdown(
            "Rank pickup zones using the final model's risk-neutral expected tip or downside-risk objective."
        )
        with gr.Row():
            planner_taxi_type = gr.Dropdown(["yellow", "green"], value="yellow", label="Taxi type")
            planner_borough = gr.Dropdown(
                ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR", "Unknown"],
                value="Manhattan",
                label="Pickup borough",
            )
            planner_objective = gr.Dropdown(
                ["risk_neutral", "risk_averse", "probability"],
                value="risk_neutral",
                label="Objective",
            )
            planner_top_k = gr.Slider(5, 20, value=10, step=1, label="Top K")
        planner_button = gr.Button("Rank zones", variant="primary")
        planner_table = gr.Dataframe(interactive=False, label="Recommended zones")
        planner_button.click(
            fn=final_zone_rankings,
            inputs=[planner_taxi_type, planner_borough, planner_objective, planner_top_k],
            outputs=planner_table,
        )

    with gr.Tab("Blog Draft"):
        gr.Markdown(ARTIFACTS["blog_background"])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
