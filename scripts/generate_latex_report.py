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


def _ablation_rows(ablation: pd.DataFrame) -> str:
    if ablation.empty:
        return r"No feature ablation metrics were generated. \\"
    rows = []
    for _, row in ablation.sort_values("expected_tip_mae").iterrows():
        rows.append(
            " & ".join(
                [
                    _latex_escape(row["ablation"]),
                    _fmt(row["features"], "int"),
                    _fmt(row["class_roc_auc"]),
                    _fmt(row["class_ece"]),
                    _fmt(row["expected_tip_mae"], "money"),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def _calibration_rows(calibration: pd.DataFrame) -> str:
    if calibration.empty:
        return r"No calibration bins were generated. \\"
    rows = []
    for _, row in calibration.iterrows():
        rows.append(
            " & ".join(
                [
                    f"{row['low']:.1f}--{row['high']:.1f}",
                    _fmt(row["rows"], "int"),
                    _fmt(row["predicted"]),
                    _fmt(row["actual"]),
                    _fmt(row["gap"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def _figure_tex(name: str, width: str, caption: str) -> str:
    if not (FIGURE_DIR / name).exists():
        return ""
    return rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width={width}]{{figures/{name}}}
\caption{{{_latex_escape(caption)}}}
\end{{figure}}
"""


def _extra_tex(extra: dict[str, object]) -> str:
    ablation = extra["ablation"]
    calibration = extra["calibration"]
    graph = extra["graph"]
    sequence = extra["sequence"]
    copilot = extra["copilot"]
    llm = extra["llm"]
    if ablation.empty and calibration.empty and graph.empty and copilot.empty and not sequence and not llm:
        return ""
    seq_text = "The sequence LSTM metrics were not generated."
    if sequence:
        seq_text = (
            f"The LSTM next-hour experiment used {int(sequence.get('test_rows', 0)):,} test windows. "
            f"Its average-tip MAE was {_fmt(sequence.get('avg_tip_mae', 0), 'money')}, compared with "
            f"{_fmt(sequence.get('naive_avg_tip_mae', 0), 'money')} for a last-hour naive baseline. "
            f"The tip-rate MAE was {float(sequence.get('tip_rate_mae', 0)):.3f}."
        )
    graph_text = "The graph metrics were not generated."
    if not graph.empty:
        row = graph.iloc[0]
        graph_text = (
            f"The graph model used {int(row.get('nodes', 0))} TLC zone nodes. "
            f"Its test zone-tip MAE was {_fmt(row.get('test_tip_mae', 0), 'money')}; "
            f"the temporal naive baseline was {_fmt(row.get('naive_test_tip_mae', 0), 'money')}. "
            "This was a useful negative result because it showed that simple year-to-year zone stability was hard to beat."
        )
    copilot_text = "The copilot checks were not generated."
    if not copilot.empty:
        overall = copilot[copilot["check"] == "overall"]
        if not overall.empty:
            copilot_text = f"The fixed prompt check reached an overall grounding score of {float(overall.iloc[0]['pass_rate']):.1%}."
    llm_text = "The driver LLM fine-tuning run was not generated."
    if llm:
        llm_text = (
            f"The compact driver LLM run fine-tuned {llm.get('base_model', 'a causal language model')} on "
            f"{int(llm.get('train_examples', 0))} generated ride-planning examples, with eval perplexity "
            f"{float(llm.get('eval_perplexity', 0)):.2f} and zone mention rate {float(llm.get('zone_mention_rate', 0)):.1%}."
        )
    fig_ablation = _figure_tex("ablation_expected_tip_mae.png", "0.82\\linewidth", "Feature ablation results using expected-tip MAE.")
    fig_calibration = _figure_tex("calibration_curve.png", "0.62\\linewidth", "Calibration curve comparing predicted tip probability with observed tip rate.")
    fig_flow = _figure_tex("zone_flow_graph_summary.png", "0.82\\linewidth", "Busiest taxi-zone nodes in the held-out pickup-dropoff flow graph.")
    fig_graph = _figure_tex("graph_gcn_zone_prediction.png", "0.68\\linewidth", "Graph model zone-tip predictions against 2025 observed zone averages.")
    fig_lstm = _figure_tex("sequence_lstm_training.png", "0.75\\linewidth", "Sequence LSTM training and validation loss.")
    fig_seq = _figure_tex("sequence_shift_signal.png", "0.62\\linewidth", "Current-hour versus next-hour tip-rate signal.")
    fig_copilot = _figure_tex("copilot_eval_summary.png", "0.62\\linewidth", "Driver Copilot grounding check pass rates.")
    return rf"""
\section{{Model Checks Over Features, Time, Zones, and Language}}
After the main model comparison, the next step was to check model behavior. The checks cover feature groups, probability calibration, zone flow, hourly history, and the driver language layer. They are part of the main experiment. They show whether the model is reliable, not just whether the average error is low.

\begin{{table}}[H]
\centering
\caption{{Feature ablation metrics. Lower expected-tip MAE and ECE are better; higher ROC-AUC is better.}}
\begin{{tabular}}{{lrrrr}}
\toprule
Ablation & Features & ROC-AUC & ECE & Expected-tip MAE \\
\midrule
{_ablation_rows(ablation)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Calibration bins for the selected hurdle model.}}
\begin{{tabular}}{{lrrrr}}
\toprule
Probability bin & Rows & Predicted & Actual & Gap \\
\midrule
{_calibration_rows(calibration)}
\bottomrule
\end{{tabular}}
\end{{table}}

{fig_ablation}
{fig_calibration}

The ablation shows which inputs matter most. Zone features, time features, fare fields, and distance fields are removed one group at a time. The change in error shows which groups the model depends on. The calibration curve checks if predicted tip probabilities match real tip rates. This matters because the demo ranks rides using these probabilities.

The ablation plot shows that the error changes only a little when one feature group is removed. This means the signal is spread across several parts of the trip record. Fare and zone fields still matter, but no single feature group explains the whole problem. The calibration plot compares predicted probability to the real tip rate in each bin. Points close to the diagonal mean the probability is usable as a probability, not just as a ranking score.

The graph experiment treats TLC zones as nodes. Pickup and dropoff traffic form the edges. Node features include pickup volume, dropoff volume, in degree, out degree, average tip, and tip rate from the training period. A small graph convolution model then predicts future zone level tip behavior. {graph_text}

{fig_flow}
{fig_graph}

The zone flow plot shows which TLC zones carry the most pickup and dropoff traffic. These busy zones are important because they shape the graph edges and dominate many driver decisions. The graph prediction plot compares predicted zone tips with the observed 2025 zone averages. The graph model captures some spatial structure, but the simple year to year baseline is still very hard to beat. This is a useful result because it shows that stable zone history is already a strong signal.

The sequence experiment groups rides by hour, taxi type, and pickup borough. A small LSTM reads the previous six hours and predicts the next hour's tip rate and average tip. {seq_text} It also makes the shift planning part use time, not only zones.

{fig_lstm}
{fig_seq}

The LSTM training plot shows whether the sequence model learned steadily instead of only fitting noise. The current hour versus next hour plot shows how much short term signal exists in the hourly data. If the points follow a clear pattern, recent hours carry information about the next hour. If the points are scattered, the next hour is harder to predict from recent history alone.

The driver copilot was checked with a fixed set of ride choice questions. This means the project used a small test set of prompts with known expected answers. Each prompt had known zones and known values from the final zone risk table. The check looked for the right zone, the expected tip value, and the cash tip limitation. It was a grounding check, not a human user study. {copilot_text} The same zone risk table was also turned into instruction examples for a small driver LLM. {llm_text} The public demo keeps deterministic retrieval as the default because it is more stable.

{fig_copilot}

The copilot plot reports the pass rate for the grounding checks. A high score means the copilot answer used the intended zone, included the expected number, and kept the cash tip caveat. This matters because the driver assistant should explain model results, not invent unsupported advice.
"""


def _img_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="data-table", escape=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _maybe_figure(name: str, caption: str) -> str:
    path = FIGURE_DIR / name
    if not path.exists():
        return ""
    return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{_img_base64(path)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def _extra_html(extra: dict[str, object]) -> str:
    ablation = extra["ablation"]
    calibration = extra["calibration"]
    graph = extra["graph"]
    copilot = extra["copilot"]
    sequence = extra["sequence"]
    llm = extra["llm"]
    if ablation.empty and calibration.empty and graph.empty and copilot.empty and not sequence and not llm:
        return ""
    seq_table = _html_table(pd.DataFrame([sequence]).round(4)) if sequence else ""
    llm_table = _html_table(pd.DataFrame([llm]).round(4)) if llm else ""
    graph_table = _html_table(graph.round(4)) if not graph.empty else ""
    copilot_table = _html_table(copilot.round(4)) if not copilot.empty else ""
    ablation_table = _html_table(ablation.round(4)) if not ablation.empty else ""
    calibration_table_html = _html_table(calibration.round(4)) if not calibration.empty else ""
    return f"""
<section>
  <h2>Model Checks Over Features, Time, Zones, and Language</h2>
  <p>After the main model comparison, the next step was to check model behavior. The checks cover feature groups, calibration, zone flow, hourly history, and the driver language layer. They are part of the main experiment. They show reliability, not only average error.</p>
  <p>The feature ablation removes groups of inputs and retrains the same tree hurdle model. The calibration table compares predicted tip probabilities with real tip rates. The graph experiment builds a pickup and dropoff network over TLC zones. The LSTM experiment uses recent hourly history to predict the next hour. The language experiment checks if the driver assistant stays grounded in model numbers.</p>
  <h3>Feature Ablation and Calibration</h3>
  {ablation_table}
  {calibration_table_html}
  {_maybe_figure("ablation_expected_tip_mae.png", "Figure 4. Feature ablation results using expected-tip MAE.")}
  {_maybe_figure("calibration_curve.png", "Figure 5. Calibration curve for the selected hurdle model.")}
  <p>The ablation plot shows that the error changes only a little when one feature group is removed. This means the signal is spread across several parts of the trip record. Fare and zone fields still matter, but no single feature group explains the whole problem. The calibration plot compares predicted probability to the real tip rate in each bin. Points close to the diagonal mean the probability is usable as a probability, not just as a ranking score.</p>
  <h3>Graph, Sequence, and Driver Copilot Checks</h3>
  {graph_table}
  {seq_table}
  {copilot_table}
  {llm_table}
  {_maybe_figure("zone_flow_graph_summary.png", "Figure 6. Busiest taxi-zone nodes in the held-out flow graph.")}
  {_maybe_figure("graph_gcn_zone_prediction.png", "Figure 7. Graph model zone-tip predictions against 2025 observations.")}
  <p>The zone flow plot shows which TLC zones carry the most pickup and dropoff traffic. These busy zones are important because they shape the graph edges and dominate many driver decisions. The graph prediction plot compares predicted zone tips with the observed 2025 zone averages. The graph model captures some spatial structure, but the simple year to year baseline is still very hard to beat.</p>
  {_maybe_figure("sequence_lstm_training.png", "Figure 8. Sequence LSTM training and validation loss.")}
  {_maybe_figure("sequence_shift_signal.png", "Figure 9. Current hour versus next hour signal.")}
  <p>The LSTM training plot shows whether the sequence model learned steadily instead of only fitting noise. The current hour versus next hour plot shows how much short term signal exists in the hourly data. A clear pattern means recent hours carry information about the next hour. A scattered pattern means the next hour is harder to predict from recent history alone.</p>
  {_maybe_figure("copilot_eval_summary.png", "Figure 10. Driver Copilot grounding check pass rates.")}
  <p>The copilot plot reports the pass rate for fixed prompt checks. These were small test prompts with known zones and known values from the final zone risk table. The answer was checked for the right zone, the expected number, and the cash tip caveat. This was a grounding check, not a human user study.</p>
</section>
"""


def _write_index_html(summary: dict, metrics: pd.DataFrame, subgroup: pd.DataFrame, zones: pd.DataFrame, extra: dict[str, object]) -> None:
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
  <title>Tip or Skip NYC Taxi Tipping Prediction</title>
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
  <h1>Tip or Skip NYC Taxi Tipping Prediction</h1>
  <p class="dek">A project report on recorded electronic tips in NYC taxi trips. The project compares simple baselines, boosted trees, and a Tabular Transformer MDN. It also includes maps, risk scores, and a driver copilot demo.</p>
  <p class="meta">Wali Ahmed, Geoffrey Kim, Jiachen Tu</p>
</header>

<section>
  <h2>Research Question</h2>
  <p>This project studies recorded electronic tips in NYC taxi trips. The main idea is that tipping should not be treated as one plain regression problem. A rider first either leaves a recorded electronic tip or does not. If there is a tip, the amount then has to be modeled.</p>
  <p>A two stage setup fits this reason. Stage 1 predicts the chance of a recorded tip. Stage 2 predicts the positive tip amount. The project then compares simple models, boosted trees, and a Tabular Transformer MDN. The demo uses the model results to compare trips and zones.</p>
  <div class="callout">Best point model: <strong>{html.escape(str(best['model']))}</strong> with expected tip MAE <strong>${best['expected_tip_mae']:.2f}</strong>. The Transformer MDN does not win on point error. It is still useful because it gives uncertainty and lower risk estimates.</div>
</section>

<section>
  <h2>Related Work and Positioning</h2>
  <p>The method uses ideas from class. Hurdle models separate the chance of an event from the size of the positive outcome. This matches electronic tips because many trips have zero recorded tip and positive tips are skewed.</p>
  <p>The deep model uses a Mixture Density Network. Instead of predicting one positive tip value, it predicts a small distribution. This gives means, medians, lower quantiles, and interval coverage. Boosted trees are included because they are strong tabular baselines.</p>
  <p>The later experiments use the same tipping problem for other course topics. These include calibration, a graph model over taxi zones, an LSTM over hourly borough history, and a small driver language model.</p>
</section>

<section>
  <h2>Dataset</h2>
  <p>The dataset comes from official NYC TLC Yellow and Green taxi records for 2024 and 2025. Credit card trips are kept because <code>tip_amount</code> records electronic tips. It does not include cash tips. Invalid fare, distance, duration, and date rows are removed.</p>
  <p>Pickup and dropoff locations are joined to TLC zone names and boroughs. This lets the model use both numeric location IDs and readable geography. The frozen package has <strong>{summary['rows']:,}</strong> rows, <strong>{summary['columns']}</strong> columns, and a recorded electronic tip rate of <strong>{summary['tip_rate']:.1%}</strong>.</p>
  <p>The split is based on time. Early 2024 is training data. Late 2024 is validation data. All of 2025 is the test set. This is more honest than a random split because the model has to work on a future year.</p>
  <p>Leakage is also removed. The feature set does not include <code>total_amount</code> because it can contain the tip. The model only gets trip context such as time, distance, duration, fare fields, taxi type, vendor, passenger bucket, rate code, and zones.</p>
</section>

<section>
  <h2>ML Formulation</h2>
  <p>Each trip becomes one feature row. Stage 1 predicts if the trip gets a recorded electronic tip. Stage 2 predicts <code>log1p(tip_amount)</code> for trips with a positive tip.</p>
  <p>The final expected tip is the tip probability times the predicted positive tip. Classification is evaluated with ROC AUC, log loss, Brier score, calibration error, and F1. Amount prediction is evaluated with log tip RMSE and expected tip MAE.</p>
</section>

<section>
  <h2>Methods</h2>
  <p>Three model families are compared. The first is a logistic and ridge hurdle baseline. The second is a boosted tree hurdle model. The third is a Tabular Transformer with an MDN head.</p>
  <p>The MDN is the generative part of the project. It predicts a mixture distribution over positive log tips. This gives expected tips, medians, lower quantiles, and intervals.</p>
  <p>The model comparison is also an ablation. The linear model tests simple feature signal. The tree model tests nonlinear tabular patterns. The Transformer MDN tests whether deep embeddings and a distributional output add useful information.</p>
</section>

<section>
  <h2>Results</h2>
  {_html_table(metrics_display)}
  <figure><img alt="Model comparison" src="data:image/png;base64,{model_img}"><figcaption>Figure 1. Model comparison across classification, calibration, conditional amount, and expected-tip metrics.</figcaption></figure>
  <figure><img alt="Monthly tip rate" src="data:image/png;base64,{monthly_img}"><figcaption>Figure 2. Recorded electronic tip rates over time for Yellow and Green taxi trips.</figcaption></figure>
  <p>Figure 1 shows the main tradeoff in the project. The boosted tree hurdle model has the best expected tip MAE and very low calibration error. The Transformer MDN is worse on point error, but it is the only model in the table that gives interval coverage. That is why the tree model is the strongest predictor while the MDN is still useful for uncertainty.</p>
  <p>Figure 2 shows that the recorded electronic tip rate stays high across months for both taxi types. The line is not perfectly flat, so month and taxi type still matter. The pattern also shows why the chronological split is useful. The model has to handle small shifts over time instead of only memorizing a random sample.</p>
  <p>The tree hurdle model gives the best point prediction. This is important because boosted trees are still very strong for tabular data. The Transformer MDN does not beat it on expected tip MAE.</p>
  <p>The deep model is still useful. It gives a distribution instead of one number. The demo uses that distribution for risk aware rankings and lower tail estimates.</p>
</section>
{_extra_html(extra)}

<section>
  <h2>Subgroups and Zones</h2>
  <p>Performance is not the same for every borough and taxi type. Manhattan has many more rows than some other areas. Smaller borough groups can be noisier. The zone table shows top expected tip zones with at least 50 test rows.</p>
  {_html_table(subgroup_display)}
  {_html_table(zone_display)}
  <figure><img alt="Top zones" src="data:image/png;base64,{zone_img}"><figcaption>Figure 3. Top pickup zones by final model expected tip.</figcaption></figure>
  <p>The zone table shows the highest expected tip pickup zones after filtering out very small zones. Airport zones such as JFK and LaGuardia appear near the top because those trips usually have larger fares and longer distances. The bar plot gives a closer look at high expected tip Manhattan pickup zones. The table also shows trip counts and lower tail values, which keeps the ranking from being read as a simple guarantee.</p>
</section>

<section>
  <h2>Interactive Demo</h2>
  <p>The Hugging Face demo turns the trained files into an interactive ML system. A user can ask about the dataset, model behavior, top zones, uncertainty, and limits. A user can also enter a trip, run what if sweeps, inspect metrics, compare subgroups, view maps, and rank zones.</p>
  <p>The assistant is grounded in project artifacts. It uses the dataset summary, metrics, subgroup table, and zone risk table. If a Hugging Face model token is available, the app can rewrite the answer with a hosted language model. The facts still come from the project files.</p>
  <p>The Model Lab now has a same ride check. It runs the boosted tree hurdle model and the Transformer MDN on the same trip inputs. This shows the point prediction beside the deep model lower and upper tip range.</p>
  <p>The demo is meant to be inspected like a model audit. Start with the assistant. Then use the prediction form and the sensitivity panel. Then compare the model lab, experiment lab, maps, and shift planner.</p>
</section>

<section>
  <h2>Driver-Facing LLM Copilot</h2>
  <p>The final demo adds Driver Copilot. This is a driver facing language layer over the tipping model. A driver can ask about one area or compare two ride options.</p>
  <p>The copilot maps area names to TLC zones. It then looks up expected tip, downside Q10 tip, predicted tip probability, and observed trip count. The answer gives a recommendation with the numbers behind it.</p>
  <p>Instruction examples were also made from the zone risk table, then used to train a compact driver LLM. The public demo keeps retrieval grounded answers as the default because they are more stable. The LLM layer is an interface to the ML results, not a replacement for the model.</p>
</section>

<section>
  <h2>Limitations</h2>
  <p>The biggest limitation is the target. TLC <code>tip_amount</code> excludes cash tips. The model predicts recorded electronic tips, not all tips. A zero in the data means no electronic tip was recorded.</p>
  <p>The deep model also does not beat the boosted tree on point error. This is still a useful result. It shows that the deep model is mainly useful for uncertainty, not for a lower average error.</p>
  <p>The model is not causal. It shows patterns in the data. It cannot prove that changing a pickup zone will cause a higher tip for one driver. The shift planner should be used for exploration, not as a guarantee.</p>
</section>

<section>
  <h2>References</h2>
  <ul>
    <li>New York City Taxi and Limousine Commission. TLC Trip Record Data and Data Dictionaries.</li>
    <li>Bishop, C. M. Mixture Density Networks. Aston University technical report, 1994.</li>
    <li>Huang, X. et al. TabTransformer: Tabular Data Modeling Using Contextual Embeddings. 2020.</li>
    <li>Guo, C. et al. On Calibration of Modern Neural Networks. ICML, 2017.</li>
    <li>Kipf, T. N. and Welling, M. Semi-Supervised Classification with Graph Convolutional Networks. ICLR, 2017.</li>
    <li>Hochreiter, S. and Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.</li>
    <li>Hu, E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021.</li>
  </ul>
</section>

<section>
  <h2>Conclusion</h2>
  <p>Tip or Skip treats taxi tipping as a two stage and uncertainty aware ML problem. The final project includes a frozen dataset, baselines, a deep generative model, model checks, subgroup analysis, maps, a shift planner, and a driver copilot. The main result is simple. Boosted trees are the best point predictor. The Transformer MDN adds useful uncertainty for the interactive demo.</p>
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
    experiment_dir = ARTIFACT_DIR / "experiments"
    extra = {
        "ablation": _read_csv(experiment_dir / "ablation_metrics.csv"),
        "calibration": _read_csv(experiment_dir / "calibration_bins.csv"),
        "graph": _read_csv(experiment_dir / "graph_metrics.csv"),
        "copilot": _read_csv(experiment_dir / "copilot_eval_summary.csv"),
        "sequence": _read_json(experiment_dir / "sequence_metrics.json"),
        "llm": _read_json(experiment_dir / "llm_finetune_metrics.json"),
    }
    metric_rows = _table_rows(metrics)
    split_rows = _split_table(summary)
    subgroup_rows = _subgroup_rows(subgroup)
    zone_rows = _zone_rows(zones)
    best = metrics.sort_values("expected_tip_mae").iloc[0]
    extra_tex = _extra_tex(extra)

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{float}}
\usepackage{{array}}
\title{{Tip or Skip NYC Taxi Tipping Prediction}}
\author{{Wali Ahmed \and Geoffrey Kim \and Jiachen Tu}}
\date{{May 2026}}
\begin{{document}}
\maketitle

\begin{{abstract}}
This project studies recorded electronic tips in NYC taxi trips using official TLC Yellow and Green taxi records. The main idea is to model tipping in two stages. First, the model predicts if a trip gets a recorded electronic tip. Then it predicts the positive tip amount. The frozen 2024 and 2025 dataset has {summary["rows"]:,} cleaned credit card trips. The project compares a logistic and ridge hurdle model, a boosted tree hurdle model, and a Tabular Transformer Mixture Density Network. The tree hurdle model is the best point predictor, with expected tip MAE of {_fmt(best["expected_tip_mae"], "money")}. The Transformer MDN adds uncertainty and lower tail estimates that are useful in the demo.
\end{{abstract}}

\section{{Introduction}}
Taxi tipping is a useful applied machine learning problem. The target is noisy. Many trips have no recorded electronic tip. Positive tips are also skewed. The result depends on time, fare, distance, taxi type, and pickup and dropoff area.

A plain regression model is not a good fit for this setup. It mixes two different tasks. The first task is whether a recorded tip exists. The second task is how large the tip is when it exists. A hurdle model separates these tasks.

The practical goal is a tool that a driver or analyst can inspect. The user can compare zones, run what if changes, view maps, and ask questions about the data. The research goal is to compare classical tabular models with a deep distributional model.

The project also includes a negative result. The deep model does not beat the boosted tree model on point error. This is still useful. It shows that boosted trees are very strong for this tabular dataset. It also shows that the main value of the Transformer MDN is uncertainty, not a lower average error.

\section{{Related Work and Motivation}}
The design uses ideas from the course. The first idea is the hurdle model. When the data has many zeros and a positive continuous tail, it can help to separate the zero part from the positive amount part. Tipping has this shape.

The second idea is strong tabular baselines. Boosted trees often do very well on structured data. They can learn nonlinear patterns without needing a very large neural network. For that reason, the boosted tree hurdle model is treated as a real baseline.

The third idea is distributional prediction. A Mixture Density Network predicts a density instead of one mean value. In this project, the MDN predicts a mixture over positive log tips. This gives quantiles and intervals. These are useful for risk aware ride ranking.

Calibration is also included. A predicted probability should mean something. A model with good ranking can still have bad probabilities. So the report includes ROC AUC along with Brier score and expected calibration error.

\section{{Dataset and Preprocessing}}
The dataset comes from official NYC TLC Yellow and Green taxi monthly parquet files for 2024 and 2025. Credit card trips are kept only. This is because TLC \texttt{{tip\_amount}} records electronic tips and does not include cash tips. So the target is recorded electronic tipping, not total tipping.

Cleaning removes trips with nonpositive fare, distance, or duration. It also removes invalid pickup dates and inconsistent trip fields. The feature set includes year, month, hour, weekday, weekend, daypart, trip duration, passenger bucket, rate code, and store and forward flag. Pickup and dropoff location IDs are joined to TLC zone names and boroughs.

The frozen dataset contains {summary["rows"]:,} rows and {summary["columns"]} columns. The overall recorded electronic tip rate is {summary["tip_rate"]:.1%}. The split is chronological. January to September 2024 is training data with {summary["split_counts"]["train"]:,} rows. October to December 2024 is validation data with {summary["split_counts"]["valid"]:,} rows. All of 2025 is test data with {summary["split_counts"]["test"]:,} rows.

This split is important. A random split could make the task too easy because similar months could appear in train and test. Testing on 2025 is closer to using the model on future trips. Leakage variables are also removed. The feature set does not use \texttt{{total\_amount}} because it can contain the tip after the trip.

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
This formulation supports multiple summaries of the same trip: expected tip, median tip, lower-tail quantiles, and interval coverage. Classification is evaluated with ROC-AUC, average precision, log loss, Brier score, expected calibration error, precision, recall, and F1. Conditional amount prediction is evaluated with log-tip MAE/RMSE. The full decision output is evaluated with expected-tip MAE.
This setup matches the data better than one regression model. The expected tip is made by multiplying the predicted chance of a tip by the predicted positive tip. The MDN version also gives a distribution. That lets us study uncertainty and downside risk.

\section{{Models and Ablations}}
Three model families are compared. The logistic and ridge baseline uses logistic regression for Stage 1 and ridge regression for Stage 2. It is the simplest model.

The tree hurdle model uses histogram gradient boosted trees for both stages. It is the strongest classical tabular baseline. It can learn nonlinear patterns between fare, distance, time, and zones.

The deep model uses categorical embeddings, normalized numeric features, a Tabular Transformer, and two heads. One head predicts tip probability. The other head is an MDN for positive tips. This is the deep and generative part of the project.

The model comparison is also an ablation. The linear model tests simple feature signal. The tree model tests nonlinear tabular signal. The Transformer MDN tests whether deep embeddings and a distributional output add useful information.

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

Figure 1 shows the main tradeoff in the project. The boosted tree hurdle model has the best expected tip MAE and very low calibration error. The Transformer MDN is worse on point error, but it is the only model in the table that gives interval coverage. That is why the tree model is the strongest predictor while the MDN is still useful for uncertainty.

The tree hurdle model is the best point predictor. It has the lowest expected tip MAE. This is a useful negative result for the deep model. The Tabular Transformer MDN does not beat a strong boosted tree baseline on this structured data.

The deep model still changes the output. It gives a conditional distribution instead of one number. This lets the demo rank zones by lower tail risk and show interval coverage. For point prediction, the boosted tree is best. For uncertainty, the Transformer MDN adds useful information.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/monthly_tip_rate.png}}
\caption{{Recorded electronic tip rate by month for Yellow and Green taxi trips in the frozen dataset.}}
\end{{figure}}

Figure 2 shows that the recorded electronic tip rate stays high across months for both taxi types. The line is not perfectly flat, so month and taxi type still matter. The pattern also shows why the chronological split is useful. The model has to handle small shifts over time instead of only memorizing a random sample.

{extra_tex}

\section{{Subgroup Behavior and Spatial Risk}}
The dataset is not evenly spread across the city. Manhattan has many rows. Some other borough groups have fewer rows and noisier estimates. Because of this, aggregate metrics can hide differences. Subgroup metrics are computed by taxi type and pickup borough.

This matters for the demo. A driver facing tool should not only show a ranked list. It should also show trip counts and uncertainty. Small sample zones can look better or worse than they really are.

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
\caption{{Top pickup zones by predicted expected tip from the final model.}}
\end{{figure}}

The zone table shows the highest expected tip pickup zones after filtering out very small zones. Airport zones such as JFK and LaGuardia appear near the top because those trips usually have larger fares and longer distances. The bar plot gives a closer look at high expected tip Manhattan pickup zones. The table also shows trip counts and lower tail values, which keeps the ranking from being read as a simple guarantee.

\section{{Interactive Hugging Face Demo}}
The Hugging Face Space is part of the final project. It is not only a visualization. It lets a reader inspect the model and the data.

The demo has a grounded tipping facts assistant. It answers using the frozen dataset summary, final metrics, subgroup table, and zone risk table. If a Hugging Face model token is set, the app can rewrite answers with a hosted language model. The facts still come from the project files.

The demo also has a prediction form, a what if sensitivity panel, model comparison controls, monthly profiles, subgroup metrics, maps, and a shift planner. The Model Lab includes a same ride check. It runs the boosted tree hurdle model and the Transformer MDN on the same trip inputs. This puts the point prediction beside the deep model lower and upper tip range. The shift planner can rank zones by expected tip, downside $Q_{{0.10}}$ tip, or tip probability. This makes the model behavior visible.

A good way to inspect the demo is simple. First ask the assistant what data was used. Then compare the model metrics. Then run the same ride model check. Then create a trip and run a sensitivity sweep. Finally compare zone rankings under different objectives.

\section{{Driver-Facing LLM Copilot}}
The final interface includes Driver Copilot. This is a driver facing language layer over the tipping model. A driver can ask about one area or compare two ride options. For example, the driver can ask about Midtown, JFK, or LaGuardia.

The copilot parses the prompt and maps area names to TLC zones. It then retrieves expected tip, downside $Q_{{0.10}}$ tip, tip probability, and observed test trip count. The answer gives a recommendation and shows the numbers behind it.

The live demo uses retrieval grounded answers as the default. This is more stable because the evidence is structured and numeric. Instruction tuning examples were also generated from the zone risk table and used to train a compact driver LLM. The tuned model is evaluated for grounding, but the public Space keeps the deterministic layer available even without an inference token.

This turns the ML results into a simple driver workflow. For one area, the copilot labels the area as strong, solid, or lower. For two areas, it recommends the one with higher expected electronic tip and reports the gap. The structured comparison form uses the deployed two stage model for two full ride options.

\section{{Limitations and Ethics}}
The biggest limitation is the target. Cash tips are not recorded in TLC \texttt{{tip\_amount}}. The model predicts recorded electronic tips. A zero in the data does not always mean that the rider left no tip. It means no electronic tip was recorded.

There are also deployment risks. Zone rankings could be misread as instructions. Small sample zones can have unstable estimates. For that reason, the demo shows trip counts and lower tail estimates.

The model is not causal. It can show patterns in the data. It cannot prove that choosing one route or zone will cause a higher tip for one driver. Traffic, passenger mix, airport rules, events, and unobserved rider behavior all matter.

\section{{Conclusion}}
Tip or Skip treats NYC taxi tipping as a two stage and uncertainty aware ML problem. The project includes a reproducible dataset, a clear formulation, strong baselines, a deep generative model, model checks, subgroup analysis, maps, and an interactive demo.

The main result is practical. Boosted trees are the best point predictor. The Transformer MDN adds distributional information that helps with risk aware exploration. This is more useful than only reporting one accuracy number.

\section{{Reproducibility}}
The frozen dataset package is stored under \texttt{{Prototype/final\_dataset}}. The main training and reporting commands are:
\begin{{verbatim}}
python scripts/train_baselines.py
python scripts/train_transformer_mdn.py --epochs 16 --batch-size 8192
python scripts/evaluate_models.py
python scripts/generate_latex_report.py
\end{{verbatim}}
The final project materials include this PDF, the offline \texttt{{index.html}} blog, the Hugging Face Space source, plots, metrics, model artifacts, and the frozen dataset package.

\begin{{thebibliography}}{{9}}
\bibitem{{tlc}} New York City Taxi and Limousine Commission. TLC Trip Record Data and Data Dictionaries.
\bibitem{{mdn}} Bishop, C. M. Mixture Density Networks. Aston University technical report, 1994.
\bibitem{{tabtransformer}} Huang, X. et al. TabTransformer: Tabular Data Modeling Using Contextual Embeddings. 2020.
\bibitem{{calibration}} Guo, C. et al. On Calibration of Modern Neural Networks. ICML, 2017.
\bibitem{{gcn}} Kipf, T. N. and Welling, M. Semi-Supervised Classification with Graph Convolutional Networks. ICLR, 2017.
\bibitem{{lstm}} Hochreiter, S. and Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.
\bibitem{{lora}} Hu, E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021.
\end{{thebibliography}}

\end{{document}}
"""
    report_path = REPORT_DIR / "Tip_or_Skip_Final_Report.tex"
    report_path.write_text(tex.strip() + "\n", encoding="utf-8")
    _write_index_html(summary, metrics, subgroup, zones, extra)

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
            "rendered_pages_simple_check",
            "last_report_build_output.txt",
        ),
    )
    shutil.copy2(FINAL_DATASET_DIR / "build_summary.json", packaged / "build_summary.json")
    print(report_path)
    print(REPORT_DIR / "index.html")


if __name__ == "__main__":
    main()
