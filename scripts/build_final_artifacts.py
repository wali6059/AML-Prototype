from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip_or_skip.config import FINAL_ARTIFACT_DIR, REPORT_DIR, SUBMISSION_DIR, ensure_directories


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--train-sample", type=int, default=None)
    parser.add_argument("--test-sample", type=int, default=None)
    parser.add_argument("--skip-extra", action="store_true")
    parser.add_argument("--extra-train-sample", type=int, default=180000)
    parser.add_argument("--extra-test-sample", type=int, default=120000)
    parser.add_argument("--sequence-epochs", type=int, default=45)
    parser.add_argument("--sequence-batch-size", type=int, default=1024)
    parser.add_argument("--sequence-seed", type=int, default=42)
    parser.add_argument("--graph-epochs", type=int, default=500)
    parser.add_argument("--graph-seed", type=int, default=42)
    parser.add_argument("--driver-seed", type=int, default=42)
    parser.add_argument("--train-driver-llm", action="store_true")
    args = parser.parse_args()

    ensure_directories()
    run([sys.executable, "scripts/train_baselines.py", *(["--sample-train", str(args.train_sample)] if args.train_sample else []), *(["--sample-test", str(args.test_sample)] if args.test_sample else [])])
    deep_cmd = [
        sys.executable,
        "scripts/train_transformer_mdn.py",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.train_sample:
        deep_cmd += ["--train-sample", str(args.train_sample)]
    if args.test_sample:
        deep_cmd += ["--test-sample", str(args.test_sample)]
    run(deep_cmd)
    run([sys.executable, "scripts/evaluate_models.py"])
    if not args.skip_extra:
        run(
            [
                sys.executable,
                "scripts/model_analysis.py",
                "--sample-train",
                str(args.extra_train_sample),
                "--sample-test",
                str(args.extra_test_sample),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/train_sequence_model.py",
                "--epochs",
                str(args.sequence_epochs),
                "--batch-size",
                str(args.sequence_batch_size),
                "--seed",
                str(args.sequence_seed),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/train_graph_model.py",
                "--epochs",
                str(args.graph_epochs),
                "--seed",
                str(args.graph_seed),
            ]
        )
        if args.train_driver_llm:
            run(
                [
                    sys.executable,
                    "scripts/train_driver_llm.py",
                    "--model",
                    "Qwen/Qwen2.5-0.5B-Instruct",
                    "--epochs",
                    "3",
                    "--batch-size",
                    "1",
                    "--lr",
                    "1e-4",
                    "--max-length",
                    "384",
                    "--lora",
                    "--seed",
                    str(args.driver_seed),
                ]
            )

    package = SUBMISSION_DIR / "tip_or_skip_completion_outputs.zip"
    if package.exists():
        package.unlink()
    package_roots = [FINAL_ARTIFACT_DIR, REPORT_DIR, ROOT / "docs"]
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root in package_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(root.parent)))
    readme = SUBMISSION_DIR / "README.md"
    readme.write_text(
        "Tip or Skip final completion package.\n\n"
        f"Artifacts: {FINAL_ARTIFACT_DIR}\n"
        f"Report data: {REPORT_DIR}\n"
        f"Blog: {ROOT / 'docs' / 'index.html'}\n"
        f"Zip: {package}\n",
        encoding="utf-8",
    )
    print(package)


if __name__ == "__main__":
    main()
