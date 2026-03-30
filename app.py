from __future__ import annotations

import json
from pathlib import Path

import folium
import gradio as gr
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from prototype_pipeline import ARTIFACT_DIR, MODEL_FEATURES, predict_tip


BOROUGH_COORDS = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
}


APP_DIR = Path(__file__).resolve().parent


def load_artifacts() -> dict:
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    monthly = pd.read_csv(ARTIFACT_DIR / "monthly_summary.csv")
    hourly = pd.read_csv(ARTIFACT_DIR / "hourly_summary.csv")
    zones = pd.read_csv(ARTIFACT_DIR / "zone_summary.csv")
    sample_rows = pd.read_csv(ARTIFACT_DIR / "sample_rows.csv")
    zone_options = pd.read_csv(ARTIFACT_DIR / "zone_options.csv")
    dataset_notes = (ARTIFACT_DIR / "dataset_notes.md").read_text(encoding="utf-8")
    blog_background = (ARTIFACT_DIR / "blog_background.md").read_text(encoding="utf-8")
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
        blocks.append(f"- Conditional tip RMSE on log scale: {values['rmse_log_tip']:.3f}")
        blocks.append("")
    blocks.append("These models are trained on credit-card trips only because TLC `tip_amount` does not include cash tips.")
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
    lookup = ARTIFACTS["zone_options"].rename(columns={"zone": "pickup_zone", "borough": "pickup_borough"})
    
    p_borough_match = lookup.loc[lookup["pickup_zone"] == pickup_zone, "pickup_borough"]
    pickup_borough = p_borough_match.iloc[0] if not p_borough_match.empty else "Unknown"

    d_borough_match = lookup.loc[lookup["pickup_zone"] == dropoff_zone, "pickup_borough"]
    dropoff_borough = d_borough_match.iloc[0] if not d_borough_match.empty else "Unknown"
    
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
            {"metric": "Tip probability", "value": round(prediction["tip_probability"], 4)},
            {"metric": "Conditional tip amount", "value": round(prediction["conditional_tip"], 2)},
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
    subset = subset.head(15)[["pickup_borough", "pickup_zone", "trips", "tip_rate", "avg_tip_amount"]]
    subset["tip_rate"] = subset["tip_rate"].round(3)
    subset["avg_tip_amount"] = subset["avg_tip_amount"].round(2)
    return subset


def _borough_agg(taxi_type: str) -> pd.DataFrame:
    zones_df = ARTIFACTS["zones"]
    subset = zones_df[zones_df["taxi_type"] == taxi_type].copy()
    stats = subset.groupby("pickup_borough").agg(
        trips=("trips", "sum"),
        tip_rate=("tip_rate", "mean"),
        avg_tip_amount=("avg_tip_amount", "mean"),
        zone_count=("pickup_zone", "count"),
    ).reset_index()
    return stats.sort_values("trips", ascending=False)


def build_nyc_map(taxi_type: str, metric: str) -> str:
    metric_col = "tip_rate" if metric == "Tip Rate" else "avg_tip_amount"
    stats = _borough_agg(taxi_type)

    fig = folium.Figure(width="100%", height="480px")
    m = folium.Map(location=[40.72, -73.96], zoom_start=11, tiles="CartoDB positron")
    fig.add_child(m)

    max_val = stats[metric_col].max()
    min_val = stats[metric_col].min()
    val_range = max(max_val - min_val, 1e-6)

    for _, row in stats.iterrows():
        borough = row["pickup_borough"]
        if borough not in BOROUGH_COORDS:
            continue
        lat, lon = BOROUGH_COORDS[borough]
        val = row[metric_col]
        t = (val - min_val) / val_range  # 0 = low, 1 = high

        # Blue (low) → Green (high)
        r = int(30 + 80 * (1 - t))
        g = int(110 + 130 * t)
        b = int(200 - 150 * t)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        display_val = f"{val:.1%}" if metric_col == "tip_rate" else f"${val:.2f}"
        radius = int(28 + 18 * t)

        popup_html = (
            f"<b>{borough}</b><br>"
            f"Trips sampled: {int(row['trips']):,}<br>"
            f"Avg tip rate: {row['tip_rate']:.1%}<br>"
            f"Avg tip amount: ${row['avg_tip_amount']:.2f}<br>"
            f"Zones covered: {int(row['zone_count'])}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#ffffff",
            weight=2,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=230),
            tooltip=f"<b>{borough}</b>: {display_val}",
        ).add_to(m)

    return fig._repr_html_()


def build_map_outputs(taxi_type: str, metric: str):
    map_html = build_nyc_map(taxi_type, metric)
    stats = _borough_agg(taxi_type)
    table = stats.rename(columns={
        "pickup_borough": "Borough",
        "trips": "Trips",
        "tip_rate": "Tip Rate",
        "avg_tip_amount": "Avg Tip ($)",
        "zone_count": "Zones",
    })
    table["Tip Rate"] = table["Tip Rate"].round(3)
    table["Avg Tip ($)"] = table["Avg Tip ($)"].round(2)
    return map_html, table


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
        gr.Markdown(metrics_markdown())
        gr.Markdown(ARTIFACTS["dataset_notes"])
        gr.Dataframe(
            ARTIFACTS["sample_rows"].head(20),
            label="Sampled cleaned rows used for the prototype",
            interactive=False,
        )

    with gr.Tab("Predict"):
        gr.Markdown(
            "Use the form below to estimate tip behavior for a hypothetical **credit-card** trip. "
            "The model predicts recorded electronic tips only."
        )
        with gr.Row():
            taxi_type = gr.Dropdown(["yellow", "green"], value="yellow", label="Taxi type")
            pickup_zone = gr.Dropdown(ZONE_CHOICES, value=DEFAULT_PICKUP, label="Pickup zone")
            dropoff_zone = gr.Dropdown(ZONE_CHOICES, value=DEFAULT_DROPOFF, label="Dropoff zone")
        with gr.Row():
            pickup_hour = gr.Slider(0, 23, value=18, step=1, label="Pickup hour")
            pickup_weekday = gr.Slider(0, 6, value=4, step=1, label="Pickup weekday (0=Mon)")
            pickup_month = gr.Slider(1, 12, value=6, step=1, label="Pickup month")
        with gr.Row():
            trip_distance = gr.Slider(0.1, 30.0, value=3.4, step=0.1, label="Trip distance (miles)")
            fare_amount = gr.Slider(3.0, 120.0, value=18.0, step=0.5, label="Fare amount ($)")
            trip_duration_minutes = gr.Slider(1.0, 120.0, value=16.0, step=1.0, label="Trip duration (minutes)")
        with gr.Row():
            vendor_id = gr.Dropdown(["1", "2", "6", "7"], value="1", label="Vendor")
            passenger_bucket = gr.Dropdown(["Unknown", "1", "2", "3", "4", "5", "6+"], value="1", label="Passenger count bucket")
            ratecode = gr.Dropdown(["1", "2", "3", "4", "5", "6", "99"], value="1", label="Rate code")
            store_and_fwd_flag = gr.Dropdown(["N", "Y", "Unknown"], value="N", label="Store and forward flag")
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
        taxi_type_chart = gr.Dropdown(["yellow", "green"], value="yellow", label="Taxi type")
        monthly_plot = gr.Plot(label="Monthly trend")
        hourly_plot = gr.Plot(label="Hourly trend")
        zone_table = gr.Dataframe(label="Top pickup zones by tip rate", interactive=False)
        taxi_type_chart.change(plot_monthly_trends, inputs=taxi_type_chart, outputs=monthly_plot)
        taxi_type_chart.change(plot_hourly_trends, inputs=taxi_type_chart, outputs=hourly_plot)
        taxi_type_chart.change(top_zones_table, inputs=taxi_type_chart, outputs=zone_table)
        demo.load(plot_monthly_trends, inputs=taxi_type_chart, outputs=monthly_plot)
        demo.load(plot_hourly_trends, inputs=taxi_type_chart, outputs=hourly_plot)
        demo.load(top_zones_table, inputs=taxi_type_chart, outputs=zone_table)

    with gr.Tab("Maps"):
        gr.Markdown(
            "## Risk-Aware Tip Maps\n"
            "Borough-level tipping patterns across NYC. Circle **size and color** reflect the selected metric. "
            "Click any circle for a breakdown, or hover for a quick read."
        )
        with gr.Row():
            map_taxi_type = gr.Dropdown(["yellow", "green"], value="yellow", label="Taxi type")
            map_metric = gr.Dropdown(["Tip Rate", "Avg Tip Amount"], value="Tip Rate", label="Metric")
        map_display = gr.HTML(label="NYC Tip Map")
        gr.Markdown("### Borough summary")
        map_table = gr.Dataframe(label="Borough statistics", interactive=False)

        map_taxi_type.change(fn=build_map_outputs, inputs=[map_taxi_type, map_metric], outputs=[map_display, map_table])
        map_metric.change(fn=build_map_outputs, inputs=[map_taxi_type, map_metric], outputs=[map_display, map_table])
        demo.load(fn=build_map_outputs, inputs=[map_taxi_type, map_metric], outputs=[map_display, map_table])

    with gr.Tab("Blog Draft"):
        gr.Markdown(ARTIFACTS["blog_background"])


if __name__ == "__main__":
    demo.launch()

