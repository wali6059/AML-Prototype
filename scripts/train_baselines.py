from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip_or_skip.baselines import evaluate_baseline, fit_baselines, save_baselines
from tip_or_skip.config import FINAL_ARTIFACT_DIR, ensure_directories, model_features
from tip_or_skip.data import load_splits, sample_for_development


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-train", type=int, default=None)
    parser.add_argument("--sample-test", type=int, default=None)
    args = parser.parse_args()

    ensure_directories()
    columns = sorted(set(model_features() + ["tip_given", "tip_amount", "log_tip_amount", "time_split"]))
    train, valid, test = load_splits(columns=columns)
    train = sample_for_development(train, args.sample_train, random_state=1)
    valid = sample_for_development(valid, args.sample_train // 3 if args.sample_train else None, random_state=2)
    test_eval = sample_for_development(test, args.sample_test, random_state=3)

    output_dir = FINAL_ARTIFACT_DIR / "baselines"
    models = fit_baselines(train, valid)
    save_baselines(models, output_dir)

    metrics = {}
    prediction_frames = []
    for name, model in models.items():
        metrics[name] = evaluate_baseline(model, test_eval)
        pred = model.predict_frame(test_eval)
        pred.insert(0, "model", name)
        pred["tip_given"] = test_eval["tip_given"].to_numpy()
        pred["tip_amount"] = test_eval["tip_amount"].to_numpy()
        prediction_frames.append(pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output_dir / "baseline_predictions_test.parquet", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

