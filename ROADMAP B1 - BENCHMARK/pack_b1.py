"""B1: bundle the benchmark driver for Kaggle.

    python "ROADMAP B1 - BENCHMARK/pack_b1.py"

Writes b1_bundle.zip: this folder's run_b1_benchmark.py, the frozen split (results/b1_splits.json,
built by build_b1_splits.py), a READ-ONLY COPY of the shipped training module (so Kaggle needs no
second dataset for it - "ADDING EOG VERSION - ADVANCED STUDENT/" is not itself attachable as a Kaggle
dataset), and READ-ONLY COPIES of the two corrected-labels files it needs (together ~600 KB - small
enough to bundle directly, rather than asking for a third Kaggle dataset just for two files). B1's
10-fold split covers all 78 SC subjects, which includes subjects in the project's locked test split, so
BOTH files are needed (see TEST_ACCESS_LOG.md in this folder for why that is deliberate, not a leak).
Nothing in either source folder is edited; pack_b1.py only copies the files it needs.
"""
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER_SRC = HERE.parent / "ADDING EOG VERSION - ADVANCED STUDENT" / "kaggle_train_student_advanced.py"
ST_NIGHTS = HERE.parent / "ADDING EOG VERSION - ST NIGHTS" / "results"
BUNDLE = HERE / "b1_bundle.zip"
SPLITS = HERE / "results" / "b1_splits.json"


def main() -> int:
    if not SPLITS.exists():
        raise SystemExit("run build_b1_splits.py first")
    members = {"run_b1_benchmark.py": HERE / "run_b1_benchmark.py",
               "kaggle_train_student_advanced.py": TRAINER_SRC,
               "corrected_labels_train_val.json": ST_NIGHTS / "corrected_labels_train_val.json",
               "corrected_labels_test.json": ST_NIGHTS / "corrected_labels_test.json",
               "b1_splits.json": SPLITS}
    missing = [n for n, p in members.items() if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for arc, src in members.items():
            z.write(src, arcname=arc)
    print(f"wrote {BUNDLE.name} ({BUNDLE.stat().st_size / 1024:.0f} KiB, {len(members)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
