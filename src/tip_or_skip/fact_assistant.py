from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FactContext:
    summary: dict[str, Any]
    metrics: pd.DataFrame
    zone_risk: pd.DataFrame
    subgroup_metrics: pd.DataFrame


def build_fact_context(
    summary: dict[str, Any],
    metrics: pd.DataFrame | None,
    zone_risk: pd.DataFrame | None,
    subgroup_metrics: pd.DataFrame | None,
) -> FactContext:
    return FactContext(
        summary=summary,
        metrics=metrics.copy() if metrics is not None else pd.DataFrame(),
        zone_risk=zone_risk.copy() if zone_risk is not None else pd.DataFrame(),
        subgroup_metrics=subgroup_metrics.copy() if subgroup_metrics is not None else pd.DataFrame(),
    )


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "unknown"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "unknown"


def _fmt_money(value: Any) -> str:
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _best_model(metrics: pd.DataFrame) -> pd.Series | None:
    if metrics.empty or "expected_tip_mae" not in metrics.columns:
        return None
    ranked = metrics.dropna(subset=["expected_tip_mae"]).sort_values("expected_tip_mae")
    if ranked.empty:
        return None
    return ranked.iloc[0]


def _model_name(name: Any) -> str:
    text = str(name)
    if text == "transformer_mdn":
        return "Transformer-MDN"
    if text == "tree_hurdle":
        return "tree_hurdle"
    if text == "logistic_ridge":
        return "logistic_ridge"
    return text


def fact_cards_markdown(context: FactContext) -> str:
    summary = context.summary
    split_counts = summary.get("split_counts", {})
    taxi_counts = summary.get("taxi_type_counts", {})
    best = _best_model(context.metrics)
    best_line = "Final model metrics are not available."
    if best is not None:
        best_line = (
            f"Best point-prediction model: **{_model_name(best['model'])}** "
            f"with expected-tip MAE **{_fmt_money(best['expected_tip_mae'])}**."
        )
    return "\n".join(
        [
            "### Grounded Project Facts",
            f"- Frozen dataset: **{_fmt_int(summary.get('rows'))} credit-card taxi trips** and **{_fmt_int(summary.get('columns'))} columns**.",
            f"- Split: train **{_fmt_int(split_counts.get('train'))}**, validation **{_fmt_int(split_counts.get('valid'))}**, test **{_fmt_int(split_counts.get('test'))}**.",
            f"- Taxi mix: Yellow **{_fmt_int(taxi_counts.get('yellow'))}**, Green **{_fmt_int(taxi_counts.get('green'))}**.",
            f"- Recorded electronic tip rate: **{_fmt_pct(summary.get('tip_rate'))}**.",
            f"- {best_line}",
            "- Important target note: TLC `tip_amount` captures recorded electronic tips, not cash tips.",
        ]
    )


def _dataset_answer(context: FactContext) -> str:
    summary = context.summary
    split_counts = summary.get("split_counts", {})
    taxi_counts = summary.get("taxi_type_counts", {})
    return (
        "The project uses a frozen NYC TLC Yellow and Green taxi dataset built from 2024 and 2025 monthly trip "
        "records. It keeps credit-card trips because TLC `tip_amount` records electronic tips but not cash tips. "
        f"After cleaning, the final dataset has **{_fmt_int(summary.get('rows'))} rows** and "
        f"**{_fmt_int(summary.get('columns'))} columns**. The chronological split is "
        f"**{_fmt_int(split_counts.get('train'))} train rows** from Jan-Sep 2024, "
        f"**{_fmt_int(split_counts.get('valid'))} validation rows** from Oct-Dec 2024, and "
        f"**{_fmt_int(split_counts.get('test'))} test rows** from all of 2025. "
        f"The taxi mix is {_fmt_int(taxi_counts.get('yellow'))} Yellow rows and "
        f"{_fmt_int(taxi_counts.get('green'))} Green rows."
    )


def _model_answer(context: FactContext) -> str:
    metrics = context.metrics
    if metrics.empty:
        return "The final metrics table is not available in this Space checkout."
    best = _best_model(metrics)
    lines = ["The final evaluation compares logistic_ridge, tree_hurdle, and Transformer-MDN on the held-out 2025 split."]
    if best is not None:
        lines.append(
            f"The strongest point-prediction model is **{_model_name(best['model'])}**, "
            f"with expected-tip MAE **{_fmt_money(best['expected_tip_mae'])}**."
        )
    transformer = metrics[metrics["model"] == "transformer_mdn"]
    if not transformer.empty:
        row = transformer.iloc[0]
        interval = ""
        if "interval80_coverage" in row and pd.notna(row["interval80_coverage"]):
            interval = f" Its 80% interval coverage is **{float(row['interval80_coverage']):.3f}**."
        lines.append(
            "The **Transformer-MDN** is still useful because it models a conditional distribution over tip amounts, "
            "which gives uncertainty estimates and lower-tail risk measures for shift planning."
            + interval
        )
    return "\n\n".join(lines)


def _zone_answer(question: str, context: FactContext) -> str:
    zones = context.zone_risk
    if zones.empty:
        return "Zone-level risk summaries are not available in this Space checkout."
    lower = question.lower()
    subset = zones.copy()
    for borough in ["manhattan", "queens", "brooklyn", "bronx", "staten island"]:
        if borough in lower and "pickup_borough" in subset.columns:
            subset = subset[subset["pickup_borough"].str.lower() == borough]
            break
    if "taxi_type" in subset.columns:
        if "yellow" in lower:
            subset = subset[subset["taxi_type"] == "yellow"]
        elif "green" in lower:
            subset = subset[subset["taxi_type"] == "green"]
    if "rows" in subset.columns:
        subset = subset[subset["rows"] >= 50]
    if subset.empty:
        return "I could not find enough zone rows for that filter. Try asking about Manhattan, Queens, Brooklyn, Yellow, or Green taxi zones."
    top = subset.sort_values("expected_tip", ascending=False).head(5)
    rows = []
    for _, row in top.iterrows():
        rows.append(
            f"- **{row['pickup_zone']}** ({row['taxi_type']}, {row['pickup_borough']}): "
            f"expected tip {_fmt_money(row['expected_tip'])}, downside Q10 {_fmt_money(row.get('q10_tip'))}, "
            f"predicted tip probability {_fmt_pct(row.get('predicted_tip_probability'))}, "
            f"observed trips {_fmt_int(row.get('rows'))}"
        )
    return "Top zones by final-model expected tip among zones with at least 50 observed test rows:\n\n" + "\n".join(rows)


def _method_answer() -> str:
    return (
        "The project is a two-stage hurdle model. Stage 1 predicts whether a trip receives any recorded electronic tip. "
        "Stage 2 models the positive tip amount after applying `log1p(tip_amount)`. The baseline systems use logistic/ridge "
        "and gradient-boosted tree models. The deep model embeds categorical trip features, combines them with normalized "
        "numeric features in a Tabular Transformer, and uses a Mixture Density Network head to represent a distribution over "
        "positive tips instead of only one point estimate."
    )


def _limitation_answer() -> str:
    return (
        "The most important limitation is the target definition: TLC `tip_amount` does not include cash tips, so the model "
        "predicts recorded electronic tipping behavior rather than all real tipping. The dataset is also a reproducible "
        "project-scale sample of TLC trips rather than every raw trip record. Results should be used for analysis and "
        "decision support, not as a guarantee of driver income."
    )


def _subgroup_answer(context: FactContext) -> str:
    subgroup = context.subgroup_metrics
    if subgroup.empty:
        return "Subgroup metrics are not available in this Space checkout."
    display = subgroup.sort_values("rows", ascending=False).head(6)
    rows = []
    for _, row in display.iterrows():
        rows.append(
            f"- **{row['taxi_type']} / {row['pickup_borough']}**: "
            f"{_fmt_int(row['rows'])} rows, ROC-AUC {float(row['roc_auc']):.3f}, F1 {float(row['f1']):.3f}"
        )
    return "Largest subgroup evaluation slices on the 2025 test split:\n\n" + "\n".join(rows)


def answer_question(question: str, context: FactContext) -> str:
    clean = (question or "").strip()
    if not clean:
        return "Ask me about the dataset, best model, uncertainty, borough performance, top zones, or project limitations."
    lower = clean.lower()
    if any(term in lower for term in ["dataset", "data", "rows", "split", "credit", "cash", "tlc"]):
        return _dataset_answer(context)
    if any(term in lower for term in ["best", "model", "auc", "mae", "transformer", "mdn", "uncertainty", "performance"]):
        return _model_answer(context)
    if any(term in lower for term in ["zone", "where", "airport", "manhattan", "queens", "brooklyn", "bronx"]):
        return _zone_answer(clean, context)
    if any(term in lower for term in ["method", "hurdle", "stage", "mixture", "density", "architecture"]):
        return _method_answer()
    if any(term in lower for term in ["subgroup", "borough", "fair", "slice"]):
        return _subgroup_answer(context)
    if any(term in lower for term in ["limit", "bias", "ethic", "caution"]):
        return _limitation_answer()
    return (
        "I can answer this from the project artifacts at a high level: the system predicts recorded electronic NYC taxi "
        "tips from TLC trip records using a two-stage hurdle setup, compares classical baselines to a Transformer-MDN, "
        "and exposes zone-level risk rankings in the demo. Try asking for the dataset, best model, top zones in Queens, "
        "or why cash tips are a limitation."
    )
