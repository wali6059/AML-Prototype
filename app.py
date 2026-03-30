from __future__ import annotations

import json
import os
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
TLC_ZONES_URL = (
    "https://data.cityofnewyork.us/api/views/8meu-9t5y/rows.geojson?accessType=DOWNLOAD"
)
_ZONE_CENTROIDS_CACHE: dict[str, tuple[float, float]] | None = None


def load_blog_background() -> str:
    root_blog_path = ROOT_DIR / "blog_background.md"
    if root_blog_path.exists():
        return root_blog_path.read_text(encoding="utf-8")
    return (ARTIFACT_DIR / "blog_background.md").read_text(encoding="utf-8")


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
        blocks.append(f"- Conditional tip RMSE on log scale: {values['rmse_log_tip']:.3f}")
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


INITIAL_MONTHLY_PLOT = plot_monthly_trends("yellow")
INITIAL_HOURLY_PLOT = plot_hourly_trends("yellow")
INITIAL_ZONE_TABLE = top_zones_table("yellow")
INITIAL_MAP_HTML, INITIAL_MAP_TABLE = build_map_outputs("yellow", "Tip Rate")


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

    with gr.Tab("Blog Draft"):
        gr.Markdown(ARTIFACTS["blog_background"])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
