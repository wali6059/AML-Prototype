from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tip_or_skip.config import REPORT_DIR, SUBMISSION_DIR, ensure_directories


def main() -> None:
    ensure_directories()
    out_dir = SUBMISSION_DIR / "courseworks_blog"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copy2(REPORT_DIR / "index.html", out_dir / "index.html")
    pdf = REPORT_DIR / "Tip_or_Skip_Final_Report.pdf"
    if pdf.exists():
        shutil.copy2(pdf, out_dir / pdf.name)
    fig_src = REPORT_DIR / "figures"
    if fig_src.exists():
        shutil.copytree(fig_src, out_dir / "figures")
    readme = out_dir / "README_SUBMISSION.md"
    readme.write_text(
        "# Tip or Skip submission\n\n"
        "Open `index.html` for the technical blog.\n\n"
        "Code and reproducibility repo: https://github.com/wali6059/AML-Prototype\n\n"
        "Interactive demo: https://huggingface.co/spaces/wali6059/Tip-or-Skip-Final\n",
        encoding="utf-8",
    )
    zip_path = SUBMISSION_DIR / "tip_or_skip_courseworks_blog.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in out_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir))
    print(zip_path)


if __name__ == "__main__":
    main()
