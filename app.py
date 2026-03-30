from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["GRADIO_SSR_MODE"] = "false"
os.environ.setdefault("MPLBACKEND", "Agg")

import gradio as gr
import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from prototype_pipeline import ARTIFACT_DIR, predict_tip

matplotlib.use("Agg")
ROOT_DIR = Path(__file__).resolve().parent


def load_blog_background() -> str:
    root_blog_path = ROOT_DIR / "blog_background.md"
    if root_blog_path.exists():
        return root_blog_path.read_text(encoding="utf-8")
    return (ARTIFACT_DIR / "blog_background.md").read_text(encoding="utf-8")


def load_artifacts() -> dict:
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
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
    return {
        "metrics": metrics,
        "monthly": monthly,
        "hourly": hourly,
        "zones": zones,
        "sample_rows": sample_rows,
        "zone_options": zone_options,
        "dataset_notes": dataset_notes,
        "blog_background": blog_background,
        "models": models,
    }


ARTIFACTS = load_artifacts()
ZONE_CHOICES = ARTIFACTS["zone_options"]["zone"].tolist()
DEFAULT_PICKUP = "Midtown Center"
DEFAULT_DROPOFF = "Upper East Side North"


def metrics_markdown() -> str:
    blocks = ["## Baseline Results", ""]
    for taxi_type, values in ARTIFACTS["metrics"].items():
        blocks.append(f"### {taxi_type.title()} taxi")
        blocks.append(f"- Test ROC-AUC: {values['roc_auc']:.3f}")
        blocks.append(f"- Test F1 @ 0.50: {values['f1_at_0_5']:.3f}")
        blocks.append(f"- Tip-rate in test split: {values['tip_rate_test']:.3f}")
        blocks.append(
            f"- Conditional tip RMSE on log scale: {values['rmse_log_tip']:.3f}"
        )
        blocks.append("")
    blocks.append(
        "These models are trained on credit-card trips only because TLC `tip_amount` does not include cash tips."
    )
    return "\n".join(blocks)


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

    d_borough_match = lookup.loc[
        lookup["pickup_zone"] == dropoff_zone, "pickup_borough"
    ]
    dropoff_borough = (
        d_borough_match.iloc[0] if not d_borough_match.empty else "Unknown"
    )

    return {
        "pickup_hour": int(pickup_hour),
        "pickup_weekday": int(pickup_weekday),
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
            {
                "metric": "Tip probability",
                "value": round(prediction["tip_probability"], 4),
            },
            {
                "metric": "Conditional tip amount",
                "value": round(prediction["conditional_tip"], 2),
            },
            {
                "metric": "Expected tip amount",
                "value": round(prediction["expected_tip"], 2),
            },
        ]
    )
    return summary, detail


def plot_monthly_trends(taxi_type: str):
    df = ARTIFACTS["monthly"]
    subset = df[df["taxi_type"] == taxi_type].sort_values("pickup_month")
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(
        subset["pickup_month"],
        subset["tip_rate"],
        marker="o",
        linewidth=2,
        color="#0b6e4f",
    )
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


INITIAL_MONTHLY_PLOT = plot_monthly_trends("yellow")
INITIAL_HOURLY_PLOT = plot_hourly_trends("yellow")
INITIAL_ZONE_TABLE = top_zones_table("yellow")


with gr.Blocks(title="NYC Taxi Tip Prototype") as demo:
    gr.Markdown(
        """
        # Tip or Skip
        A lightweight prototype for NYC taxi tipping prediction built from 2025 TLC yellow and green taxi trip records.

        This Space demonstrates meaningful progress for an Applied Machine Learning project:
        1. the dataset is identified and processed,
        2. an initial two-stage ML pipeline is trained,
        3. the background section of the technical write-up is started.
        """
    )

    with gr.Tab("Overview"):
        with gr.Row():
            gr.Markdown(metrics_markdown())
            gr.Markdown(ARTIFACTS["dataset_notes"])
        gr.Dataframe(
            value=ARTIFACTS["sample_rows"].head(20),
            label="Sampled cleaned rows used for the prototype",
            interactive=False,
        )

    with gr.Tab("Predict"):
        gr.Markdown(
            "Use the form below to estimate tip behavior for a hypothetical **credit-card** trip. "
            "The model predicts recorded electronic tips only."
        )
        with gr.Row():
            taxi_type = gr.Dropdown(
                ["yellow", "green"], value="yellow", label="Taxi type"
            )
            pickup_zone = gr.Dropdown(
                ZONE_CHOICES, value=DEFAULT_PICKUP, label="Pickup zone"
            )
            dropoff_zone = gr.Dropdown(
                ZONE_CHOICES, value=DEFAULT_DROPOFF, label="Dropoff zone"
            )
        with gr.Row():
            pickup_hour = gr.Slider(0, 23, value=18, step=1, label="Pickup hour")
            pickup_weekday = gr.Slider(
                0, 6, value=4, step=1, label="Pickup weekday (0=Mon)"
            )
            pickup_month = gr.Slider(1, 12, value=6, step=1, label="Pickup month")
        with gr.Row():
            trip_distance = gr.Slider(
                0.1, 30.0, value=3.4, step=0.1, label="Trip distance (miles)"
            )
            fare_amount = gr.Slider(
                3.0, 120.0, value=18.0, step=0.5, label="Fare amount ($)"
            )
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
            ratecode = gr.Dropdown(
                ["1", "2", "3", "4", "5", "6", "99"], value="1", label="Rate code"
            )
            store_and_fwd_flag = gr.Dropdown(
                ["N", "Y", "Unknown"], value="N", label="Store and forward flag"
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

    with gr.Tab("Explore"):
        gr.Markdown(
            "Explore precomputed summaries from the sampled 2025 prototype dataset."
        )
        with gr.Row():
            taxi_type_chart = gr.Dropdown(
                ["yellow", "green"], value="yellow", label="Taxi type"
            )
            refresh_button = gr.Button("Refresh charts")
        with gr.Row():
            monthly_plot = gr.Plot(
                value=INITIAL_MONTHLY_PLOT,
                label="Monthly trend",
            )
            hourly_plot = gr.Plot(
                value=INITIAL_HOURLY_PLOT,
                label="Hourly trend",
            )
        zone_table = gr.Dataframe(
            value=INITIAL_ZONE_TABLE,
            label="Top pickup zones by tip rate",
            interactive=False,
        )
        refresh_button.click(
            plot_monthly_trends, inputs=taxi_type_chart, outputs=monthly_plot
        )
        refresh_button.click(
            plot_hourly_trends, inputs=taxi_type_chart, outputs=hourly_plot
        )
        refresh_button.click(
            top_zones_table, inputs=taxi_type_chart, outputs=zone_table
        )
        taxi_type_chart.change(
            plot_monthly_trends, inputs=taxi_type_chart, outputs=monthly_plot
        )
        taxi_type_chart.change(
            plot_hourly_trends, inputs=taxi_type_chart, outputs=hourly_plot
        )
        taxi_type_chart.change(
            top_zones_table, inputs=taxi_type_chart, outputs=zone_table
        )

    with gr.Tab("Blog Draft"):
        gr.Markdown(ARTIFACTS["blog_background"])


if __name__ == "__main__":
    demo.launch()
