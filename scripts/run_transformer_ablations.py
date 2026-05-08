from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
matplotlib.use("Agg")

import tip_or_skip.features as feature_module
from tip_or_skip.config import ARTIFACT_DIR, FIGURE_DIR, ensure_directories, model_features
from tip_or_skip.data import load_splits, sample_for_development
from tip_or_skip.extra import ablation_columns
from tip_or_skip.training import TrainConfig, evaluate_deep_model, train_transformer_mdn


ABLATIONS = ["full", "no_zones", "no_time", "no_fare", "no_distance"]


def _write_partial_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    if not rows:
        return
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "transformer_ablation_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    order = table.sort_values("expected_tip_mae")
    ax.bar(order["ablation"], order["expected_tip_mae"], color="#335c67")
    ax.set_title("Transformer-MDN Feature Ablation: Expected Tip MAE")
    ax.set_ylabel("MAE ($)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "transformer_ablation_expected_tip_mae.png", dpi=180)
    fig.savefig(FIGURE_DIR / "transformer_ablation_expected_tip_mae.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-mix", type=int, default=5)
    parser.add_argument("--train-sample", type=int, default=None)
    parser.add_argument("--valid-sample", type=int, default=None)
    parser.add_argument("--test-sample", type=int, default=None)
    parser.add_argument("--only", nargs="*", choices=ABLATIONS, default=ABLATIONS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_directories()
    output_dir = ARTIFACT_DIR / "runs" / "transformer_ablation_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_features = model_features()
    columns = sorted(set(base_features + ["tip_given", "tip_amount", "log_tip_amount", "time_split"]))
    train, valid, test = load_splits(columns=columns)
    train = sample_for_development(train, args.train_sample, random_state=101)
    valid = sample_for_development(valid, args.valid_sample, random_state=102)
    test = sample_for_development(test, args.test_sample, random_state=103)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_mix=args.n_mix,
        train_sample=None,
    )

    rows: list[dict[str, object]] = []
    for name in args.only:
        run_dir = output_dir / name
        metric_path = run_dir / "deep_metrics.json"
        if metric_path.exists() and not args.force:
            row = json.loads(metric_path.read_text(encoding="utf-8"))["ablation_metrics"]
            rows.append(row)
            print(json.dumps({"ablation": name, "status": "skipped_existing", **row}), flush=True)
            continue

        use_cols = ablation_columns(base_features, name)
        keep_cols = use_cols + ["tip_given", "tip_amount", "log_tip_amount", "time_split"]
        feature_module.model_features = lambda cols=use_cols: list(cols)

        model, encoder, train_info = train_transformer_mdn(
            train[keep_cols].copy(),
            valid[keep_cols].copy(),
            run_dir,
            config,
        )
        metrics, predictions = evaluate_deep_model(model, encoder, test[keep_cols].copy(), device=train_info["device"])
        predictions.to_parquet(run_dir / "deep_predictions_test.parquet", index=False)

        row = {
            "ablation": name,
            "features": len(use_cols),
            "rows_train": int(len(train)),
            "rows_valid": int(len(valid)),
            "rows_test": int(len(test)),
            "device": train_info["device"],
            "best_valid_loss": train_info["best_valid_loss"],
            "class_roc_auc": metrics["class_roc_auc"],
            "class_average_precision": metrics["class_average_precision"],
            "class_log_loss": metrics["class_log_loss"],
            "class_brier": metrics["class_brier"],
            "class_ece": metrics["class_ece"],
            "logtip_mae": metrics["logtip_mae"],
            "logtip_rmse": metrics["logtip_rmse"],
            "interval80_coverage": metrics["interval80_coverage"],
            "interval80_mean_width": metrics["interval80_mean_width"],
            "expected_tip_mae": metrics["expected_tip_mae"],
        }
        payload = {
            "model": "transformer_mdn",
            "ablation": name,
            "removed_feature_group": name if name != "full" else None,
            "features": use_cols,
            "train_config": asdict(config),
            "train_info": train_info,
            "test_metrics": metrics,
            "ablation_metrics": row,
        }
        metric_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append(row)
        _write_partial_summary(rows, output_dir)
        print(json.dumps({"ablation": name, "status": "completed", **row}), flush=True)

    _write_partial_summary(rows, output_dir)
    print(pd.DataFrame(rows).to_csv(index=False), flush=True)


if __name__ == "__main__":
    main()
