from __future__ import annotations

import argparse
import shutil
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
    run([sys.executable, "scripts/generate_latex_report.py"])

    package = SUBMISSION_DIR / "tip_or_skip_completion_outputs.zip"
    if package.exists():
        package.unlink()
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root in [FINAL_ARTIFACT_DIR, REPORT_DIR]:
            for path in root.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(root.parent)))
    readme = SUBMISSION_DIR / "README.md"
    readme.write_text(
        "Tip or Skip final completion package.\n\n"
        f"Artifacts: {FINAL_ARTIFACT_DIR}\n"
        f"Report: {REPORT_DIR}\n"
        f"Zip: {package}\n",
        encoding="utf-8",
    )
    print(package)


if __name__ == "__main__":
    main()

