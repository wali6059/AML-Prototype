from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip_or_skip.config import ARTIFACT_DIR, FINAL_DATASET_DIR, FIGURE_DIR, REPORT_DIR, ensure_directories


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _fmt(value: object, kind: str = "float") -> str:
    if pd.isna(value):
        return "--"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "pct":
        return f"{float(value):.1%}"
    if kind == "money":
        return rf"\${float(value):.2f}"
    return f"{float(value):.3f}"


def _table_rows(metrics: pd.DataFrame) -> str:
    preferred = [
        "class_roc_auc",
        "class_log_loss",
        "class_brier",
        "class_ece",
        "class_f1",
        "logtip_rmse",
        "expected_tip_mae",
        "interval80_coverage",
    ]
    cols = [column for column in preferred if column in metrics.columns]
    labels = {
        "class_roc_auc": "ROC-AUC",
        "class_log_loss": "Log loss",
        "class_brier": "Brier",
        "class_ece": "ECE",
        "class_f1": "F1",
        "logtip_rmse": "Log-tip RMSE",
        "expected_tip_mae": "Expected-tip MAE",
        "interval80_coverage": "80\\% cov.",
    }
    rows = []
    for _, row in metrics.iterrows():
        values = [_latex_escape(row["model"])]
        for col in cols:
            values.append(_fmt(row[col], "money" if col == "expected_tip_mae" else "float"))
        rows.append(" & ".join(values) + r" \\")
    header = "Model & " + " & ".join(labels[col] for col in cols) + r" \\"
    return header + "\n" + r"\midrule" + "\n" + "\n".join(rows)


def _split_table(summary: dict) -> str:
    rows = []
    split_counts = summary["split_counts"]
    for split, label in [("train", "Train: Jan--Sep 2024"), ("valid", "Validation: Oct--Dec 2024"), ("test", "Test: Jan--Dec 2025")]:
        rows.append(f"{label} & {split_counts[split]:,} \\\\")
    rows.append(f"Total & {summary['rows']:,} \\\\")
    return "\n".join(rows)


def _subgroup_rows(subgroup: pd.DataFrame) -> str:
    if subgroup.empty:
        return r"No subgroup metrics were generated. \\"
    display = subgroup.sort_values("rows", ascending=False).head(6)
    rows = []
    for _, row in display.iterrows():
        rows.append(
            " & ".join(
                [
                    _latex_escape(row["taxi_type"]),
                    _latex_escape(row["pickup_borough"]),
                    _fmt(row["rows"], "int"),
                    _fmt(row["roc_auc"]),
                    _fmt(row["f1"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def _zone_rows(zones: pd.DataFrame) -> str:
    if zones.empty:
        return r"No zone summaries were generated. \\"
    display = zones[zones["rows"] >= 50].sort_values("expected_tip", ascending=False).head(6)
    rows = []
    for _, row in display.iterrows():
        rows.append(
            " & ".join(
                [
                    _latex_escape(row["taxi_type"]),
                    _latex_escape(row["pickup_borough"]),
                    _latex_escape(row["pickup_zone"]),
                    _fmt(row["rows"], "int"),
                    _fmt(row["expected_tip"], "money"),
                    _fmt(row["q10_tip"], "money"),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def _img_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="data-table", escape=True)


def _write_index_html(summary: dict, metrics: pd.DataFrame, subgroup: pd.DataFrame, zones: pd.DataFrame) -> None:
    best = metrics.sort_values("expected_tip_mae").iloc[0]
    split = pd.DataFrame(
        [
            {"Split": "Train: Jan-Sep 2024", "Rows": f"{summary['split_counts']['train']:,}"},
            {"Split": "Validation: Oct-Dec 2024", "Rows": f"{summary['split_counts']['valid']:,}"},
            {"Split": "Test: Jan-Dec 2025", "Rows": f"{summary['split_counts']['test']:,}"},
        ]
    )
    metric_cols = [
        "model",
        "class_roc_auc",
        "class_brier",
        "class_ece",
        "class_f1",
        "logtip_rmse",
        "expected_tip_mae",
        "interval80_coverage",
    ]
    metrics_display = metrics[[c for c in metric_cols if c in metrics.columns]].round(4)
    subgroup_display = subgroup.sort_values("rows", ascending=False).head(6).round(4)
    zone_display = zones[zones["rows"] >= 50].sort_values("expected_tip", ascending=False).head(6).round(4)
    model_img = _img_base64(FIGURE_DIR / "model_comparison.png")
    monthly_img = _img_base64(FIGURE_DIR / "monthly_tip_rate.png")
    zone_img = _img_base64(FIGURE_DIR / "top_zone_expected_tip.png")
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tip or Skip: Uncertainty-Aware NYC Taxi Tipping Prediction</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; color: #1d2528; background: #f7f7f2; line-height: 1.58; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 44px 24px 72px; }}
    header {{ border-bottom: 3px solid #1f6f5b; padding-bottom: 24px; margin-bottom: 32px; }}
    h1 {{ font-size: 42px; line-height: 1.08; margin: 0 0 12px; color: #173b35; }}
    h2 {{ margin-top: 36px; color: #173b35; border-bottom: 1px solid #c9d5ce; padding-bottom: 6px; }}
    h3 {{ color: #335c67; }}
    .dek {{ font-size: 19px; max-width: 820px; }}
    .meta {{ color: #566; font-size: 14px; }}
    .callout {{ background: #fff; border-left: 5px solid #e09f3e; padding: 16px 18px; margin: 20px 0; }}
    figure {{ margin: 24px 0; background: #fff; padding: 16px; border: 1px solid #d8ddd9; }}
    figcaption {{ color: #4e5b5c; font-size: 14px; margin-top: 8px; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    .data-table {{ width: 100%; border-collapse: collapse; margin: 16px 0 24px; background: #fff; font-size: 14px; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #d8ddd9; padding: 8px 10px; text-align: left; }}
    .data-table th {{ background: #e8efe9; color: #173b35; }}
    code {{ background: #edf1ed; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
<main>
<header>
  <p class="meta">Applied Machine Learning Final Project - May 2026</p>
  <h1>Tip or Skip: Uncertainty-Aware NYC Taxi Tipping Prediction</h1>
  <p class="dek">A technical project report on modeling recorded electronic tips in NYC taxi trips with baselines, a Tabular Transformer Mixture Density Network, uncertainty-aware route ranking, and a driver-facing LLM copilot in an interactive Hugging Face demo.</p>
  <p class="meta">Wali Ahmed, Geoffrey Kim, Jiachen Tu</p>
</header>

<section>
  <h2>Research Question</h2>
  <p>This project asks whether taxi tipping can be framed as more than a point-prediction task. Our hypothesis is that a calibrated two-stage model, especially one that returns a distribution over possible positive tips, gives more useful decision-support signals than a single regression model. The application is concrete: a driver or analyst wants to compare zones, trip conditions, and risk-sensitive objectives. The machine learning question is broader: how do classical tabular baselines compare with a deep distributional model when the target is sparse, skewed, and only partially observed?</p>
  <p>This distinction matters because a standard regression framing can look successful while still hiding the behavior that users care about. A model can be close on average but badly calibrated, insensitive to neighborhood structure, or unable to describe downside risk. The final demo therefore treats the model output as an object to inspect rather than a number to accept. The research question becomes: what kind of model output is most useful for understanding and acting on tipping data?</p>
  <div class="callout">Best point-prediction model: <strong>{html.escape(str(best['model']))}</strong> with expected-tip MAE <strong>${best['expected_tip_mae']:.2f}</strong>. The Transformer-MDN did not win on point error, but it gives uncertainty intervals and lower-tail risk estimates used by the shift-planning demo.</div>
</section>

<section>
  <h2>Dataset</h2>
  <p>The dataset is built from official NYC TLC Yellow and Green taxi monthly trip records for 2024 and 2025. We keep credit-card trips because the TLC <code>tip_amount</code> field records electronic tips and does not capture cash tips. Rows with nonpositive fare, distance, duration, or invalid month alignment are removed. Pickup and dropoff location identifiers are joined to TLC taxi-zone names and boroughs so the model can learn both spatial identifiers and interpretable geographic groupings.</p>
  <p>The frozen model-ready package contains <strong>{summary['rows']:,}</strong> rows, <strong>{summary['columns']}</strong> columns, and an overall recorded electronic tip rate of <strong>{summary['tip_rate']:.1%}</strong>. The split is chronological, so the held-out test set evaluates future-year generalization rather than random-row memorization.</p>
  <p>Several preprocessing choices are designed to avoid leakage. We do not include <code>total_amount</code> because it can mechanically contain the tip, and we do not include payment-type variation because the final target is defined only over credit-card trips. Instead, the model receives trip context available before the tip is known: pickup time, trip distance and duration, fare and surcharge fields, taxi type, vendor, rate code, passenger bucket, and zone information. This makes the prediction task closer to the decision-support scenario shown in the demo.</p>
  <p>The chronological split also makes the experiment more honest. If the rows were randomly shuffled, the model could benefit from near-identical seasonal, policy, or travel-pattern conditions appearing in both train and test. Testing on 2025 forces the system to confront a future period. That matters for this project because the final interface is meant to support exploration under changing city conditions, not simply memorize one static snapshot of taxi behavior. It is a stricter and more realistic evaluation setup.</p>
</section>

<section>
  <h2>ML Formulation</h2>
  <p>For each trip, the input vector includes temporal features, fare and surcharge fields, trip distance and duration, taxi type, passenger bucket, rate code, and pickup/dropoff zone attributes. Stage 1 predicts whether a trip receives any recorded electronic tip. Stage 2 predicts the positive tip amount after applying <code>log1p(tip_amount)</code>. Evaluation uses ROC-AUC, log loss, Brier score, expected calibration error, F1, log-tip RMSE, and expected-tip MAE.</p>
  <p>The formulation is intentionally a hurdle model rather than a single regressor. Tipping has two qualitatively different events: deciding whether a recorded tip exists, and deciding how large that tip is. Separating those events makes the prediction target easier to interpret and supports expected-value calculations of the form probability of tip times conditional tip amount.</p>
</section>

<section>
  <h2>Methods</h2>
  <p>We compare three model families. The logistic-ridge baseline uses logistic regression for the binary stage and ridge regression for the conditional amount stage. The tree hurdle model uses histogram gradient-boosted trees for both stages and is a strong classical tabular baseline. The deep model embeds categorical features, combines them with normalized numeric features in a Tabular Transformer backbone, and uses two heads: a binary classifier and a Mixture Density Network over positive log tips.</p>
  <p>The MDN is the generative component of the project. Instead of returning one positive-tip value, it estimates a mixture of Gaussian components in log-tip space. That distribution supports conditional means, medians, lower-tail quantiles, and interval coverage. This makes it possible to ask risk-aware questions, such as which zone has the better downside profile, not only which zone has the highest mean.</p>
  <p>The model comparison is also an ablation. The linear baseline tests whether engineered features alone are enough. The tree model tests whether nonlinear tabular interactions are enough. The Transformer-MDN tests whether representation learning and distributional prediction add useful behavior. This is why a negative result for the Transformer on point error is still informative: it separates point accuracy from uncertainty usefulness.</p>
</section>

<section>
  <h2>Results</h2>
  {_html_table(metrics_display)}
  <figure><img alt="Model comparison" src="data:image/png;base64,{model_img}"><figcaption>Figure 1. Model comparison across classification, calibration, conditional amount, and expected-tip metrics.</figcaption></figure>
  <figure><img alt="Monthly tip rate" src="data:image/png;base64,{monthly_img}"><figcaption>Figure 2. Recorded electronic tip rates over time for Yellow and Green taxi trips.</figcaption></figure>
  <p>The strongest point-prediction result comes from the tree hurdle baseline, not the Transformer-MDN. That negative result is important: on this tabular dataset, boosted trees remain a hard baseline to beat. The deep model is still useful because it changes the output object from a point prediction to a distribution, which the demo uses for risk-aware rankings.</p>
  <p>The calibration results are especially important for an applied system. Expected-tip MAE says how close the final dollar prediction is on average, while Brier score and expected calibration error measure whether probabilities behave like probabilities. The tree model performs strongly on these metrics, suggesting that the main deployment value of the Transformer-MDN is not higher point accuracy but the ability to expose a range of plausible positive-tip outcomes.</p>
</section>

<section>
  <h2>Subgroups and Zones</h2>
  <p>Subgroup metrics show that performance is not uniform across borough and taxi type. Large Manhattan slices dominate the dataset, while smaller borough slices can be noisier and less calibrated. The zone table shows the model's top expected-tip zones among zones with at least 50 held-out test rows.</p>
  {_html_table(subgroup_display)}
  {_html_table(zone_display)}
  <figure><img alt="Top zones" src="data:image/png;base64,{zone_img}"><figcaption>Figure 3. Top Manhattan pickup zones by final-model expected tip.</figcaption></figure>
</section>

<section>
  <h2>Interactive Demo</h2>
  <p>The Hugging Face demo turns the trained artifacts into an interactive ML system. Users can ask a grounded tipping-facts assistant about the dataset, model behavior, top zones, uncertainty, and limitations. They can edit hypothetical trip inputs, run what-if sensitivity sweeps over hour, fare, distance, or duration, inspect final model metrics, compare subgroup performance, view maps, and rank zones using risk-neutral, risk-averse, or probability objectives.</p>
  <p>The assistant is deliberately grounded in project artifacts rather than open-ended text generation. If an optional Hugging Face Inference API model and token are configured, the app can rewrite grounded answers through a hosted language model; otherwise it uses deterministic retrieval from the final dataset summary, metrics, subgroup table, and zone-risk table. This keeps the demo reliable for grading while still showing how language interfaces can sit on top of ML results.</p>
  <p>The most useful way to inspect the demo is to move across tabs as if evaluating a model audit. Start with the assistant to ask what the dataset is and which model wins. Then use the prediction form to create a trip, run a sensitivity sweep, and watch how predicted expected tip changes. Finally, use the model lab and shift planner to compare aggregate metrics against zone-level recommendations. This workflow makes the project interactive without hiding the underlying evidence.</p>
</section>

<section>
  <h2>Driver-Facing LLM Copilot</h2>
  <p>The final demo adds a driver-facing LLM layer called Driver Copilot. The goal is not to make a generic chatbot that talks about taxis; it is to make a natural-language decision layer over the trained tipping system. A driver can ask questions such as: “I am at Midtown Center and got ride options to JFK Airport or LaGuardia Airport. Which should I choose?” The copilot extracts the relevant TLC zones, retrieves the final model’s expected tip, downside Q10 tip, predicted tip probability, and observed trip count, then returns a recommendation with evidence.</p>
  <p>We built this as a retrieval-grounded LLM interface rather than fine-tuning a large language model from scratch. The dataset is structured and numeric, so the reliable part of the system should be the trained tipping model and report artifacts. The language layer parses the driver’s prompt, maps area names and common aliases to TLC zones, compares candidate areas, and optionally passes the grounded answer through a Hugging Face text-generation model if an inference token is configured. Without a token, the deterministic grounded response still works in the public Space.</p>
  <p>The result is a practical shift-planning assistant. For single-area prompts, it labels an area as strong, solid, or lower relative to comparable zones. For two-option prompts, it recommends the option with higher expected electronic tip and reports the expected-tip gap plus downside-risk evidence. The structured comparison form goes further by using the deployed two-stage trip model to compare two concrete rides with user-specified pickup area, dropoff areas, hour, weekday, month, distance, fare, and duration. This makes the LLM layer an interface to the machine learning system, not a replacement for it.</p>
</section>

<section>
  <h2>Limitations</h2>
  <p>The largest limitation is the target itself. TLC <code>tip_amount</code> excludes cash tips, so the model predicts recorded electronic tipping behavior rather than all tipping. The dataset is a reproducible project-scale sample, not the full raw TLC archive. Zone-level rankings can also be unstable for small sample zones, which is why the demo exposes observed trip counts and lower-tail quantiles instead of only showing a sorted leaderboard.</p>
  <p>The main technical limitation is that the deep model did not outperform the strongest tree baseline on point metrics. This is not hidden in the report because the project goal is model-behavior insight, not a Kaggle-style accuracy claim. The result suggests that, for structured taxi-trip data, distributional outputs and uncertainty may be a better reason to use the deep model than marginal point-prediction accuracy.</p>
  <p>A second limitation is causality. The model can identify associations between trip context and recorded tips, but it cannot prove that changing a route or pickup zone will cause a higher tip for a specific driver. Demand, passenger mix, traffic, airport rules, and unobserved rider behavior all matter. For that reason, the shift planner should be read as an exploratory ranking tool rather than a prescriptive routing system.</p>
</section>

<section>
  <h2>Conclusion</h2>
  <p>Tip or Skip reframes taxi tipping prediction as a calibrated, uncertainty-aware decision-support problem. The final system includes a frozen dataset, strong baselines, a deep generative component, ablations through model comparisons, subgroup evaluation, a risk-aware shift planner, and an offline blog-style report. The central finding is pragmatic: boosted trees are the strongest point predictor, while the Transformer-MDN contributes distributional information that makes the interactive demo more useful for analyzing risk and uncertainty.</p>
</section>
</main>
</body>
</html>
"""
    (REPORT_DIR / "index.html").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_directories()
    summary = json.loads((FINAL_DATASET_DIR / "build_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(REPORT_DIR / "final_metrics.csv")
    subgroup = pd.read_csv(REPORT_DIR / "subgroup_metrics.csv") if (REPORT_DIR / "subgroup_metrics.csv").exists() else pd.DataFrame()
    zones = pd.read_csv(REPORT_DIR / "zone_risk_summary.csv") if (REPORT_DIR / "zone_risk_summary.csv").exists() else pd.DataFrame()
    metric_rows = _table_rows(metrics)
    split_rows = _split_table(summary)
    subgroup_rows = _subgroup_rows(subgroup)
    zone_rows = _zone_rows(zones)
    best = metrics.sort_values("expected_tip_mae").iloc[0]

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{float}}
\usepackage{{array}}
\title{{Tip or Skip: Uncertainty-Aware NYC Taxi Tipping Prediction}}
\author{{Wali Ahmed \and Geoffrey Kim \and Jiachen Tu}}
\date{{May 2026}}
\begin{{document}}
\maketitle

\begin{{abstract}}
This project investigates recorded electronic tipping behavior in New York City taxi trips using official TLC Yellow and Green Taxi records. The central hypothesis is that tipping should be modeled as a calibrated two-stage distributional problem rather than a single point-regression problem: first predict whether a trip receives any recorded electronic tip, then model the conditional distribution of positive tips. We build a frozen 2024--2025 dataset with {summary["rows"]:,} cleaned credit-card trips, compare logistic/ridge and boosted-tree hurdle baselines against a Tabular Transformer Mixture Density Network (MDN), and deploy the results in an interactive Hugging Face demo with a grounded tipping-facts assistant, what-if sensitivity sweeps, model comparison panels, maps, and a risk-aware shift planner. The strongest point-prediction model is the tree hurdle baseline, with expected-tip MAE of {_fmt(best["expected_tip_mae"], "money")}, while the Transformer-MDN provides uncertainty estimates and lower-tail quantiles that support decision-focused analysis.
\end{{abstract}}

\section{{Introduction}}
Taxi tipping is a small individual decision repeated millions of times across a dense urban transportation system. It is also a useful applied machine learning problem because the observed target is skewed, partially censored, and shaped by trip context. A naive formulation would ask for one number: the predicted tip amount. That formulation hides two important facts. First, a trip may or may not receive a recorded electronic tip at all. Second, when a tip occurs, the amount can vary widely across airport trips, short neighborhood trips, boroughs, time of day, and fare levels.

Our project asks whether a hurdle-style, uncertainty-aware formulation produces a more useful model than a single point predictor. The practical application is a driver-facing or analyst-facing decision-support tool: compare zones, inspect risk, ask natural-language questions about the data, and run what-if experiments. The machine learning question is broader: when does a deep distributional model add value on structured tabular data where strong classical baselines are already available?

This framing follows the project guidelines by focusing on a testable hypothesis and model behavior rather than only leaderboard performance. We include strong baselines, a deep generative component, ablation through model-family comparisons, subgroup evaluation, and a discussion of negative results. The final result is not simply ``train a taxi model.'' It is a small research-style study of calibration, uncertainty, and decision-support outputs for a real tabular dataset.

The project is also designed around inspectability. A useful applied model should let a user ask why one model is preferred, where the model is reliable, and how predictions move when inputs change. For that reason, the final artifact includes both a technical report and an interactive demo. The demo is not separate from the experiment; it is a way to expose model behavior, subgroup performance, and uncertainty to a reader who wants to interrogate the results directly.

\section{{Related Work and Motivation}}
The modeling design draws on three ideas. The first is hurdle modeling: when outcomes have a mass at zero and a continuous positive tail, separating occurrence from amount can be more interpretable and often easier to optimize than fitting one regressor. Tipping naturally has this structure because the event of leaving any recorded electronic tip is distinct from the conditional amount.

The second idea is strong tabular baselines. Gradient-boosted tree models remain difficult to beat on many structured datasets because they capture nonlinear threshold effects and feature interactions without requiring large-scale representation learning. For that reason, the project treats boosted trees as a serious baseline rather than as a strawman.

The third idea is distributional prediction. Mixture Density Networks model a conditional density instead of only a mean. In this project the MDN head estimates a mixture distribution over positive log tips. That makes the deep model useful even when its point-prediction error is not the best result: it gives quantiles, interval coverage, and lower-tail risk estimates that can be used directly in the shift planner.

A fourth motivation is calibration. In a decision-support setting, a probability should be meaningful, not just rank examples correctly. A driver comparing two options needs to know whether a predicted probability and expected value are stable enough to trust. We therefore report both discrimination metrics such as ROC-AUC and calibration-sensitive metrics such as Brier score and expected calibration error.

\section{{Dataset and Preprocessing}}
The dataset is built from official NYC TLC Yellow and Green taxi monthly parquet files for 2024 and 2025. We retain credit-card trips only because TLC \texttt{{tip\_amount}} records electronic tips and does not include cash tips. This choice avoids mixing observed zeros with unobserved cash tips, but it also means the target is recorded electronic tipping behavior rather than all real-world tipping.

Cleaning removes rows with nonpositive fare, distance, or duration, invalid pickup-month alignment, and inconsistent basic trip fields. We derive pickup year, month, hour, weekday, weekend indicator, daypart, trip duration in minutes, passenger-count buckets, rate-code categories, and normalized store-and-forward flags. Pickup and dropoff location identifiers are joined to TLC taxi-zone names, boroughs, and service zones. The final model feature set contains temporal, fare, surcharge, location, taxi-type, vendor, and trip-structure variables.

The frozen dataset contains {summary["rows"]:,} rows and {summary["columns"]} columns. Its overall recorded electronic tip rate is {summary["tip_rate"]:.1%}. The split is chronological: January--September 2024 for training ({summary["split_counts"]["train"]:,} rows), October--December 2024 for validation ({summary["split_counts"]["valid"]:,} rows), and all of 2025 for testing ({summary["split_counts"]["test"]:,} rows). This setup makes the held-out test split a future-period evaluation instead of a random split that could overstate performance through temporal leakage.

We also remove leakage-prone variables from the model inputs. In particular, \texttt{{total\_amount}} is excluded because it can encode the tip after the fact, and the task is restricted to credit-card trips rather than letting payment type directly reveal target observability. The retained variables describe trip context: time, distance, duration, fare components, taxi type, vendor, passenger bucket, rate code, and pickup/dropoff geography. This matches the interactive prediction setting, where the user asks what the model expects from a hypothetical trip before the tip is known.

\section{{Machine Learning Formulation}}
For a trip $i$ with features $x_i$, Stage 1 defines
\[
y_i = \mathbb{{1}}[\texttt{{tip\_amount}}_i > 0]
\]
and estimates $P(y_i=1 \mid x_i)$. Stage 2 models the transformed positive amount
\[
z_i = \log(1+\texttt{{tip\_amount}}_i)
\]
for trips where $y_i=1$. The expected electronic tip is then approximated by combining the probability of a tip with the conditional positive-tip prediction.

For the MDN model, the conditional positive-tip density is
\[
p_\theta(z \mid y=1,x)=\sum_{{k=1}}^K \pi_k(x)\mathcal{{N}}(z;\mu_k(x),\sigma_k^2(x)).
\]
This formulation supports multiple summaries of the same trip: expected tip, median tip, lower-tail quantiles, and interval coverage. We evaluate classification with ROC-AUC, average precision, log loss, Brier score, expected calibration error, precision, recall, and F1. We evaluate conditional amount prediction with log-tip MAE/RMSE and evaluate full decision output with expected-tip MAE.

\section{{Models and Ablations}}
We compare three model families. The logistic-ridge baseline combines logistic regression for Stage 1 with ridge regression for Stage 2. It tests how much signal is recoverable from a simple linear model with regularization. The tree hurdle baseline uses histogram gradient-boosted trees for both stages. It is the strongest classical tabular baseline and captures nonlinear interactions among fare, distance, time, and zone variables.

The deep model embeds categorical features and combines them with normalized numeric features in a Tabular Transformer. It has a binary classification head and an MDN head for the positive-tip distribution. This is the project requirement's deep/generative component. The comparison across linear, tree, and Transformer-MDN models functions as the main ablation: it asks whether the extra representational and distributional machinery improves point prediction, uncertainty, or decision-support outputs.

The ablation is intentionally framed around model behavior. The linear baseline tests whether the engineered features are already sufficient under a simple decision boundary. The tree model tests whether nonlinear tabular interactions explain most of the signal. The Transformer-MDN tests whether a deep representation and a generative output distribution add value beyond those tabular baselines. This lets us interpret the final outcome even when the deep model does not win every metric.

\section{{Experimental Results}}
\begin{{table}}[H]
\centering
\caption{{Final model comparison on the held-out 2025 test split.}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
{metric_rows}
\bottomrule
\end{{tabular}}}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/model_comparison.png}}
\caption{{Comparison across classification, calibration, conditional amount, and expected-tip metrics.}}
\end{{figure}}

The tree hurdle model is the best point predictor, achieving the lowest expected-tip MAE. This is a meaningful negative result for the deep model: the Tabular Transformer-MDN does not dominate a strong boosted-tree baseline on structured taxi data. However, the deep model changes the form of the output. It provides an estimated conditional distribution, which allows the project to rank zones using lower-tail risk and to report interval coverage. This is more useful for the interactive shift-planning setting than a single regression output.

The results therefore separate two notions of success. For point prediction, the boosted-tree hurdle model is the most practical choice. For uncertainty-aware analysis, the Transformer-MDN contributes information the tree model does not directly provide. This distinction is the main insight of the project: the best model for a leaderboard metric is not necessarily the only model worth deploying in an exploratory ML interface.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/monthly_tip_rate.png}}
\caption{{Recorded electronic tip rate by month for Yellow and Green taxi trips in the frozen dataset.}}
\end{{figure}}

\section{{Subgroup Behavior and Spatial Risk}}
Because the dataset is geographically uneven, aggregate metrics can hide important differences. We therefore compute subgroup metrics by taxi type and pickup borough. Large Manhattan slices dominate the data, while smaller borough slices tend to be noisier and less calibrated. This matters for deployment because a driver-facing recommendation should show uncertainty and observed trip counts, not only a sorted list of high-value zones.

\begin{{table}}[H]
\centering
\caption{{Largest Transformer-MDN subgroup evaluation slices.}}
\begin{{tabular}}{{llrrr}}
\toprule
Taxi & Pickup borough & Rows & ROC-AUC & F1 \\
\midrule
{subgroup_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Top final-model pickup zones by expected tip among zones with at least 50 observed test rows.}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lllrrr}}
\toprule
Taxi & Borough & Zone & Rows & Expected tip & Q10 tip \\
\midrule
{zone_rows}
\bottomrule
\end{{tabular}}}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.9\linewidth]{{figures/top_zone_expected_tip.png}}
\caption{{Top Manhattan pickup zones by predicted expected tip according to the final uncertainty-aware model.}}
\end{{figure}}

\section{{Interactive Hugging Face Demo}}
The Hugging Face Space is part of the final ML artifact, not only a visualization wrapper. The demo includes a grounded tipping-facts assistant that answers questions using the frozen dataset summary, final metrics, subgroup table, and zone-risk table. If a Hugging Face Inference API token and model are configured, the app can route these grounded facts through a hosted language model; otherwise it uses deterministic artifact retrieval. This keeps the assistant reliable for grading and prevents unsupported claims.

The demo also includes a prediction form for hypothetical credit-card trips, a what-if sensitivity sweep over pickup hour, fare, distance, or duration, final model-comparison controls, monthly dataset profiles, subgroup metrics, a Manhattan map, and a risk-aware shift planner. The shift planner can optimize for expected tip, downside $Q_{{0.10}}$ tip, or predicted tip probability. These controls expose the machine learning behavior directly: users can see how changing inputs changes predictions and how different objectives change zone rankings.

The intended inspection workflow is sequential. A reader can first ask the assistant what dataset and target were used, then compare model metrics in the model lab, then construct a hypothetical trip and run a sensitivity sweep, and finally inspect whether zone recommendations change under risk-neutral versus risk-averse objectives. This makes the demo a compact model-audit tool rather than a static dashboard.

\section{{Driver-Facing LLM Copilot}}
The final interface includes a driver-facing LLM layer called Driver Copilot. Its purpose is to let a driver ask natural questions about ride choice and shift planning, such as: ``I am at Midtown Center and got two ride options, JFK Airport or LaGuardia Airport. Which one should I choose?'' The copilot parses the prompt, identifies known TLC zones and aliases, retrieves final model outputs for those zones, and answers with a recommendation grounded in expected tip, downside $Q_{{0.10}}$ tip, predicted tip probability, and observed held-out trip count.

We did not fine-tune a general language model from scratch because the core evidence is structured. Instead, we built a retrieval-grounded LLM interface over the trained tipping artifacts. The deterministic layer maps driver language to zone-level and trip-level model outputs. If a Hugging Face Inference API token and model are configured, the app can pass the grounded answer through a hosted text-generation model for conversational rewriting. If no token is configured, the public demo still works because the grounded response itself is generated from the project artifacts.

This component turns the machine learning results into a usable driver workflow. For a single area, the copilot classifies the area as strong, solid, or lower relative to comparable zones. For two candidate areas, it recommends the option with higher expected electronic tip and reports the gap. The structured comparison form additionally uses the deployed two-stage trip predictor to compare two rides with specified pickup area, dropoff areas, hour, weekday, month, distance, fare, and duration. The result is an LLM-style planning layer whose outputs are auditable rather than free-form.

\section{{Limitations and Ethics}}
The most important limitation is target observability. Cash tips are not recorded in TLC \texttt{{tip\_amount}}, so the model should be described as predicting recorded electronic tips. A zero in the data does not necessarily mean a rider left no tip; it means no electronic tip was recorded. This affects interpretation, especially across neighborhoods or trip types where cash behavior may differ.

There are also deployment risks. Zone rankings could amplify existing demand patterns if treated as instructions rather than analysis. Small-sample zones can produce unstable estimates, so the demo exposes observed trip counts and lower-tail quantiles. The model should be used as decision support, not as a guarantee of income or as a replacement for local driver knowledge.

The model is not causal. It can show that certain trip contexts are associated with higher recorded electronic tips, but it cannot prove that choosing a particular route will cause a higher tip for a particular driver. Passenger mix, traffic, airport rules, event timing, and unobserved rider behavior all affect outcomes. This is why the final interface emphasizes comparison, uncertainty, and evidence rather than prescriptive instructions.

\section{{Conclusion}}
Tip or Skip reframes NYC taxi tipping prediction as an uncertainty-aware applied ML problem. The project produces a reproducible dataset, clear ML formulation, strong baselines, a deep generative component, ablations through model comparisons, subgroup and spatial analyses, and an interactive demo. The main empirical finding is pragmatic: boosted trees are the best point predictor, while the Transformer-MDN contributes distributional information that makes the system more useful for risk-aware exploration. This gives a more honest and more interesting result than simply reporting an accuracy number.

\section{{Reproducibility}}
The frozen dataset package is stored under \texttt{{Prototype/final\_dataset}}. The main training and reporting commands are:
\begin{{verbatim}}
python scripts/train_baselines.py
python scripts/train_transformer_mdn.py --epochs 16 --batch-size 8192
python scripts/evaluate_models.py
python scripts/generate_latex_report.py
\end{{verbatim}}
The final project materials include this PDF, the offline \texttt{{index.html}} technical blog, the Hugging Face Space source, plots, metrics, model artifacts, and the frozen dataset package.

\begin{{thebibliography}}{{9}}
\bibitem{{tlc}} New York City Taxi and Limousine Commission. TLC Trip Record Data and Data Dictionaries.
\bibitem{{mdn}} Bishop, C. M. Mixture Density Networks. Aston University technical report, 1994.
\bibitem{{tabtransformer}} Huang, X. et al. TabTransformer: Tabular Data Modeling Using Contextual Embeddings. 2020.
\bibitem{{calibration}} Guo, C. et al. On Calibration of Modern Neural Networks. ICML, 2017.
\end{{thebibliography}}

\end{{document}}
"""
    report_path = REPORT_DIR / "Tip_or_Skip_Final_Report.tex"
    report_path.write_text(tex.strip() + "\n", encoding="utf-8")
    _write_index_html(summary, metrics, subgroup, zones)

    if shutil.which("pdflatex"):
        for suffix in [".aux", ".log", ".out"]:
            aux_path = report_path.with_suffix(suffix)
            if aux_path.exists():
                aux_path.unlink()
        subprocess.run(["pdflatex", "-interaction=nonstopmode", report_path.name], cwd=REPORT_DIR, check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", report_path.name], cwd=REPORT_DIR, check=True)

    packaged = ARTIFACT_DIR / "final_report"
    if packaged.exists():
        shutil.rmtree(packaged)
    shutil.copytree(
        REPORT_DIR,
        packaged,
        ignore=shutil.ignore_patterns(
            "*.aux",
            "*.log",
            "*.out",
            "rendered_pages",
            "last_report_build_output.txt",
        ),
    )
    shutil.copy2(FINAL_DATASET_DIR / "build_summary.json", packaged / "build_summary.json")
    print(report_path)
    print(REPORT_DIR / "index.html")


if __name__ == "__main__":
    main()
