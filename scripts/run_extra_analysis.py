from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
matplotlib.use("Agg")

from tip_or_skip.config import ARTIFACT_DIR, FIGURE_DIR, NUMERIC_FEATURES, REPORT_DIR, ensure_directories, model_features
from tip_or_skip.data import load_dataset, load_splits, sample_for_development
from tip_or_skip.driver_copilot import answer_driver_question, build_driver_context
from tip_or_skip.extra import ablation_columns, calibration_table, flow_features, llm_examples, score_answers, sequence_rows, write_jsonl
from tip_or_skip.metrics import classification_summary, regression_summary

A_PLUS_DIR = ARTIFACT_DIR / "a_plus"


def _xy(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    num_cols = [col for col in cols if col in NUMERIC_FEATURES]
    cat_cols = [col for col in cols if col not in num_cols]
    pieces_train = []
    pieces_test = []
    if num_cols:
        tr = train[num_cols].apply(pd.to_numeric, errors="coerce")
        te = test[num_cols].apply(pd.to_numeric, errors="coerce")
        mean = tr.mean().fillna(0.0)
        std = tr.std().replace(0, 1.0).fillna(1.0)
        pieces_train.append(((tr.fillna(mean) - mean) / std).to_numpy(dtype=np.float32))
        pieces_test.append(((te.fillna(mean) - mean) / std).to_numpy(dtype=np.float32))
    for col in cat_cols:
        values = train[col].fillna("Unknown").astype(str)
        codes = {value: i + 1 for i, value in enumerate(values.value_counts().head(250).index)}
        pieces_train.append(values.map(codes).fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1))
        pieces_test.append(test[col].fillna("Unknown").astype(str).map(codes).fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1))
    return np.concatenate(pieces_train, axis=1), np.concatenate(pieces_test, axis=1)


def _fit_ablation(train: pd.DataFrame, test: pd.DataFrame, name: str, cols: list[str]) -> dict[str, float]:
    x_train, x_test = _xy(train, test, cols)
    y = train["tip_given"].to_numpy(dtype=int)
    clf = HistGradientBoostingClassifier(max_iter=90, learning_rate=0.07, l2_regularization=0.03, random_state=42, early_stopping=True)
    clf.fit(x_train, y)
    tipped = train["tip_given"] == 1
    reg = HistGradientBoostingRegressor(max_iter=90, learning_rate=0.07, l2_regularization=0.03, random_state=42, early_stopping=True)
    reg.fit(x_train[tipped.to_numpy()], train.loc[tipped, "log_tip_amount"].to_numpy(dtype=float))
    prob = clf.predict_proba(x_test)[:, 1]
    log_tip = reg.predict(x_test)
    expected = prob * np.expm1(log_tip).clip(min=0)
    class_metrics = classification_summary(test["tip_given"], prob)
    tipped_test = test["tip_given"] == 1
    log_metrics = regression_summary(test.loc[tipped_test, "log_tip_amount"], log_tip[tipped_test.to_numpy()])
    tip_metrics = regression_summary(test["tip_amount"], expected)
    return {
        "ablation": name,
        "features": len(cols),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "class_roc_auc": class_metrics["roc_auc"],
        "class_brier": class_metrics["brier"],
        "class_ece": class_metrics["ece"],
        "class_f1": class_metrics["f1"],
        "logtip_rmse": log_metrics["rmse"],
        "expected_tip_mae": tip_metrics["mae"],
    }


def _run_ablations(args) -> pd.DataFrame:
    cols = model_features() + ["tip_given", "tip_amount", "log_tip_amount", "time_split"]
    train, valid, test = load_splits(columns=cols)
    train = pd.concat([train, valid], ignore_index=True)
    train = sample_for_development(train, args.sample_train, random_state=17)
    test = sample_for_development(test, args.sample_test, random_state=29)
    features = model_features()
    names = ["full", "no_zones", "no_time", "no_fare", "no_distance"]
    rows = []
    for name in names:
        use_cols = ablation_columns(features, name)
        rows.append(_fit_ablation(train, test, name, use_cols))
    out = pd.DataFrame(rows)
    out.to_csv(A_PLUS_DIR / "ablation_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    order = out.sort_values("expected_tip_mae")
    ax.bar(order["ablation"], order["expected_tip_mae"], color="#335c67")
    ax.set_title("Feature Ablation: Expected Tip MAE")
    ax.set_ylabel("MAE ($)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ablation_expected_tip_mae.png", dpi=180)
    plt.close(fig)
    return out


def _predictions(model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(ARTIFACT_DIR / "final" / "transformer_mdn" / "deep_test_frame.parquet")
    if model_name == "transformer_mdn":
        pred = pd.read_parquet(ARTIFACT_DIR / "final" / "transformer_mdn" / "deep_predictions_test.parquet")
    else:
        all_pred = pd.read_parquet(ARTIFACT_DIR / "final" / "baselines" / "baseline_predictions_test.parquet")
        pred = all_pred[all_pred["model"] == model_name].drop(columns=["model", "tip_given", "tip_amount"]).reset_index(drop=True)
    return frame.reset_index(drop=True), pred.reset_index(drop=True)


def _run_calibration(model_name: str) -> pd.DataFrame:
    frame, pred = _predictions(model_name)
    table = calibration_table(frame["tip_given"], pred["tip_probability"], bins=10)
    table.to_csv(A_PLUS_DIR / "calibration_bins.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--")
    ax.plot(table["predicted"], table["actual"], marker="o", linewidth=2.0, color="#176b54")
    ax.set_title(f"Calibration Curve: {model_name}")
    ax.set_xlabel("Predicted tip probability")
    ax.set_ylabel("Observed tip rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration_curve.png", dpi=180)
    plt.close(fig)
    return table


def _run_flow_and_sequence() -> tuple[pd.DataFrame, pd.DataFrame]:
    trips = load_dataset(
        columns=[
            "taxi_type",
            "time_split",
            "pickup_datetime",
            "PULocationID",
            "DOLocationID",
            "pickup_zone",
            "dropoff_zone",
            "tip_given",
            "tip_amount",
        ]
    )
    test = trips[trips["time_split"] == "test"].copy()
    flow = flow_features(test)
    flow.to_csv(A_PLUS_DIR / "zone_flow_features.csv", index=False)
    top = flow.head(15).sort_values("flow_count")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(top["zone"], top["flow_count"], color="#e09f3e")
    ax.set_title("Busiest Test-Period Flow Zones")
    ax.set_xlabel("Pickup + dropoff count")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "zone_flow_graph_summary.png", dpi=180)
    plt.close(fig)

    seq = sequence_rows(trips)
    seq.to_csv(A_PLUS_DIR / "sequence_rows.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(seq["tip_rate"], seq["tip_rate_next"], alpha=0.35, s=16, color="#335c67")
    ax.set_title("Hourly Sequence Signal")
    ax.set_xlabel("Current hour tip rate")
    ax.set_ylabel("Next hour tip rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sequence_shift_signal.png", dpi=180)
    plt.close(fig)
    return flow, seq


def _run_copilot_and_llm() -> pd.DataFrame:
    zones = pd.read_csv(REPORT_DIR / "zone_risk_summary.csv")
    options = pd.read_csv(ARTIFACT_DIR / "zone_options.csv")
    context = build_driver_context(zones, options["zone"].tolist())
    pairs = [
        ("JFK Airport", "LaGuardia Airport"),
        ("Midtown Center", "Battery Park City"),
        ("Times Sq/Theatre District", "Upper East Side North"),
        ("Financial District North", "Red Hook"),
    ]
    cases = []
    answers = []
    for a, b in pairs:
        ranked = zones[zones["pickup_zone"].isin([a, b])].sort_values(["expected_tip", "rows"], ascending=False)
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        question = f"I am choosing between {a} and {b}. Which area is better for tips?"
        cases.append(
            {
                "question": question,
                "expected_zone": best["pickup_zone"],
                "expected_tip": best["expected_tip"],
                "must_include": "cash tips",
            }
        )
        answers.append(answer_driver_question(question, context))
    case_df = pd.DataFrame(cases)
    scored = score_answers(case_df, answers)
    scored.to_csv(A_PLUS_DIR / "copilot_eval.csv", index=False)
    summary = pd.DataFrame(
        [
            {"check": "zone", "pass_rate": scored["has_zone"].mean()},
            {"check": "number", "pass_rate": scored["has_number"].mean()},
            {"check": "caveat", "pass_rate": scored["has_caveat"].mean()},
            {"check": "overall", "pass_rate": scored["score"].mean()},
        ]
    )
    summary.to_csv(A_PLUS_DIR / "copilot_eval_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(summary["check"], summary["pass_rate"], color="#176b54")
    ax.set_ylim(0, 1.05)
    ax.set_title("Driver Copilot Grounding Checks")
    ax.set_ylabel("Pass rate")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "copilot_eval_summary.png", dpi=180)
    plt.close(fig)

    examples = llm_examples(zones[zones["rows"] >= 50], limit=220)
    split = max(1, int(len(examples) * 0.85))
    write_jsonl(A_PLUS_DIR / "llm_train.jsonl", examples[:split])
    write_jsonl(A_PLUS_DIR / "llm_eval.jsonl", examples[split:])
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-train", type=int, default=180000)
    parser.add_argument("--sample-test", type=int, default=120000)
    parser.add_argument("--calibration-model", default="tree_hurdle")
    args = parser.parse_args()
    ensure_directories()
    A_PLUS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ablation": len(_run_ablations(args)),
        "calibration": len(_run_calibration(args.calibration_model)),
        "flow_sequence": [len(x) for x in _run_flow_and_sequence()],
        "copilot": len(_run_copilot_and_llm()),
    }
    (A_PLUS_DIR / "extra_summary.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
