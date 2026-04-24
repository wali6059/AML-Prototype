from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from prototype_pipeline import ARTIFACT_DIR, MODEL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TabularTransformerMDN

APP_DIR = Path(__file__).resolve().parent

def load_artifacts() -> dict:
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    monthly = pd.read_csv(ARTIFACT_DIR / "monthly_summary.csv")
    zone_options = pd.read_csv(ARTIFACT_DIR / "zone_options.csv")
    dataset_notes = (ARTIFACT_DIR / "dataset_notes.md").read_text(encoding="utf-8")
    blog_background = (ARTIFACT_DIR / "blog_background.md").read_text(encoding="utf-8")
    
    models = {}
    preprocessors = {}
    
    for taxi_type in ["yellow", "green"]:
        checkpoint_path = ARTIFACT_DIR / f"{taxi_type}_v2_model.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model = TabularTransformerMDN(
                num_numeric=checkpoint["config"]["num_numeric"],
                cat_cardinalities=checkpoint["config"]["cat_cardinalities"]
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            
            models[taxi_type] = model
            preprocessors[taxi_type] = checkpoint["preprocessor"]
            
    return {
        "metrics": metrics,
        "monthly": monthly,
        "zone_options": zone_options,
        "dataset_notes": dataset_notes,
        "blog_background": blog_background,
        "models": models,
        "preprocessors": preprocessors,
    }

ARTIFACTS = load_artifacts()
ZONE_CHOICES = ARTIFACTS["zone_options"]["zone"].tolist()
DEFAULT_PICKUP = "Midtown Center"
DEFAULT_DROPOFF = "Upper East Side North"

def metrics_markdown() -> str:
    blocks = ["## Deep Learning Baseline (Transformer + MDN)", ""]
    for taxi_type, values in ARTIFACTS["metrics"].items():
        blocks.append(f"### {taxi_type.title()} taxi")
        blocks.append(f"- **Test ROC-AUC**: {values['roc_auc']:.3f}")
        blocks.append(f"- **Test Log-Tip RMSE**: {values['rmse_log_tip']:.3f}")
        blocks.append(f"- Tip-rate in test split: {values['tip_rate_test']:.3f}")
        blocks.append("")
    blocks.append("> Architecture: Tabular Transformer with Self-Attention and Mixture Density Network Head.")
    return "\n".join(blocks)

def _build_feature_row(
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
    
    p_match = lookup.loc[lookup["pickup_zone"] == pickup_zone, "pickup_borough"]
    pickup_borough = p_match.iloc[0] if not p_match.empty else "Unknown"
    
    d_match = lookup.loc[lookup["pickup_zone"] == dropoff_zone, "borough"] 
    dropoff_borough = d_match.iloc[0] if not d_match.empty else "Unknown"
    
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

def run_prediction(*args):
    taxi_type = args[0]
    inputs = args[1:]
    
    feature_dict = _build_feature_row(*inputs)
    df = pd.DataFrame([feature_dict])
    
    model = ARTIFACTS["models"][taxi_type]
    preprocessor = ARTIFACTS["preprocessors"][taxi_type]
    
    X_raw = preprocessor.transform(df[MODEL_FEATURES])
    num = torch.FloatTensor(X_raw[:, :len(NUMERIC_FEATURES)])
    cat = torch.LongTensor(X_raw[:, len(NUMERIC_FEATURES):].astype(float) + 1)
    
    with torch.no_grad():
        prob, pi, mu, sigma = model(num, cat)
        cond_log_amt = torch.sum(pi * mu, dim=1).item()
        cond_amt = np.expm1(cond_log_amt)
        expected_tip = prob.item() * cond_amt
        
    summary = (
        f"### Model Output (Transformer + MDN)\n"
        f"- **Estimated chance of a tip**: {prob.item():.1%}\n"
        f"- **Predicted conditional amount**: ${max(0, cond_amt):.2f}\n"
        f"- **Expected tip value**: ${max(0, expected_tip):.2f}\n\n"
        f"*Note: The conditional amount is the mean of the learned Mixture Density distribution.*"
    )
    
    detail = pd.DataFrame([
        {"Metric": "Tip Probability", "Value": round(prob.item(), 4)},
        {"Metric": "MDN Mean (log)", "Value": round(cond_log_amt, 4)},
        {"Metric": "Expected Amount ($)", "Value": round(expected_tip, 2)}
    ])
    
    return summary, detail

def plot_monthly_trends(taxi_type: str):
    df = ARTIFACTS["monthly"]
    subset = df[df["taxi_type"] == taxi_type].sort_values("pickup_month")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(subset["pickup_month"], subset["tip_rate"], marker="s", color="#2e7d32")
    ax.set_title(f"{taxi_type.title()} Taxi: Monthly Recorded Tip Rate")
    ax.set_ylim(0, 1)
    return fig

with gr.Blocks(title="NYC Taxi Tip - Deep Learning Prototype") as demo:
    gr.Markdown("# Tip or Skip: GenAI Edition")
    gr.Markdown("Prototype utilizing a **Tabular Transformer** backbone and **MDN** heads.")

    with gr.Tab("Overview"):
        gr.Markdown(metrics_markdown())
        gr.Markdown(ARTIFACTS["dataset_notes"])

    with gr.Tab("Predict"):
        with gr.Row():
            taxi_type_in = gr.Dropdown(["yellow", "green"], value="yellow", label="Taxi Type")
            p_zone_in = gr.Dropdown(ZONE_CHOICES, value=DEFAULT_PICKUP, label="Pickup Zone")
            d_zone_in = gr.Dropdown(ZONE_CHOICES, value=DEFAULT_DROPOFF, label="Dropoff Zone")
        with gr.Row():
            hour_in = gr.Slider(0, 23, value=18, label="Hour")
            wday_in = gr.Slider(0, 6, value=4, label="Weekday (0=Mon)")
            month_in = gr.Slider(1, 12, value=6, label="Month")
        with gr.Row():
            dist_in = gr.Slider(0.1, 30.0, value=3.0, label="Distance (mi)")
            fare_in = gr.Slider(3.0, 100.0, value=15.0, label="Fare ($)")
            dur_in = gr.Slider(1, 120, value=15, label="Duration (min)")
        with gr.Row():
            vend_in = gr.Dropdown(["1", "2"], value="1", label="Vendor")
            pass_in = gr.Dropdown(["1", "2", "3", "4", "5", "6+"], value="1", label="Passengers")
            rate_in = gr.Dropdown(["1", "2", "3", "4", "5"], value="1", label="Rate Code")
            flag_in = gr.Dropdown(["N", "Y"], value="N", label="Store & Fwd")
            
        btn = gr.Button("Estimate Tip Distribution", variant="primary")
        out_text = gr.Markdown()
        out_table = gr.Dataframe(interactive=False)
        
        btn.click(
            fn=run_prediction,
            inputs=[taxi_type_in, p_zone_in, d_zone_in, hour_in, wday_in, month_in, dist_in, fare_in, dur_in, vend_in, pass_in, rate_in, flag_in],
            outputs=[out_text, out_table]
        )

    with gr.Tab("Explore"):
        taxi_sel = gr.Dropdown(["yellow", "green"], value="yellow", label="Select Taxi")
        plot_out = gr.Plot()
        taxi_sel.change(plot_monthly_trends, inputs=taxi_sel, outputs=plot_out)
        demo.load(plot_monthly_trends, inputs=taxi_sel, outputs=plot_out)

    with gr.Tab("Technical Blog"):
        gr.Markdown(ARTIFACTS["blog_background"])

if __name__ == "__main__":
    demo.launch()
