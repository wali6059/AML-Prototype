from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip_or_skip.config import FINAL_ARTIFACT_DIR, ensure_directories, model_features
from tip_or_skip.data import load_splits, sample_for_development
from tip_or_skip.training import TrainConfig, evaluate_deep_model, train_transformer_mdn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-mix", type=int, default=5)
    parser.add_argument("--train-sample", type=int, default=None)
    parser.add_argument("--test-sample", type=int, default=None)
    args = parser.parse_args()

    ensure_directories()
    columns = sorted(set(model_features() + ["tip_given", "tip_amount", "log_tip_amount", "time_split"]))
    train, valid, test = load_splits(columns=columns)
    test_eval = sample_for_development(test, args.test_sample, random_state=5)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_mix=args.n_mix,
        train_sample=args.train_sample,
    )
    output_dir = FINAL_ARTIFACT_DIR / "transformer_mdn"
    model, encoder, train_info = train_transformer_mdn(train, valid, output_dir, config)
    metrics, predictions = evaluate_deep_model(model, encoder, test_eval, device=train_info["device"])
    output = {
        "model": "transformer_mdn",
        "train_config": asdict(config),
        "train_info": train_info,
        "test_metrics": metrics,
        "test_rows_evaluated": int(len(test_eval)),
    }
    (output_dir / "deep_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    predictions.to_parquet(output_dir / "deep_predictions_test.parquet", index=False)
    test_eval.reset_index(drop=True).to_parquet(output_dir / "deep_test_frame.parquet", index=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

