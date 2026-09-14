"""Zip exactly what the Kaggle notebook reads: the training set, the template references,
and kaggle_sft.py.

    python -m student.pack_kaggle          writes student/sft_bundle.zip (git-ignored)

Upload the zip as one Kaggle dataset; Kaggle unpacks it. See student/KAGGLE.md.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "sft_bundle.zip"
MEMBERS = ("kaggle_sft.py", "trainset/train.jsonl", "trainset/valid.jsonl", "trainset/manifest.json",
           "templates/rendered_reference.json", "templates/Qwen2.5-1.5B-Instruct.jinja",
           "templates/Llama-3.2-3B-Instruct.jinja", "templates/Llama-3.2-3B-Instruct.pinned.jinja")


def pack(dest: Path = BUNDLE) -> Path:
    missing = [m for m in MEMBERS if not (HERE / m).exists()]
    if missing:
        raise SystemExit(f"missing from student/: {missing}")
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for m in MEMBERS:
            z.write(HERE / m, arcname=m)
    return dest


if __name__ == "__main__":
    p = pack()
    print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KiB, {len(MEMBERS)} files)")
