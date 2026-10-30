"""B1: the standard Sleep-EDF-78 10-fold subject-wise split (SC only, 78 subjects).

    python "ROADMAP B1 - BENCHMARK/build_b1_splits.py"

Runs locally, CPU only, from the project's own processed_sleepedf_mc/index.csv (no raw EDFs needed).
Deterministic: a fixed-seed shuffle of the 78 SC subject IDs, then chunked into 10 folds as evenly as
possible (8 folds of 8 subjects, 2 of 7 = 78). Each fold is one CV split: that fold is held out as
validation, the other 9 are training. Writes results/b1_splits.json, which pack_b1.py bundles for Kaggle.

This is a SEPARATE 10-fold split over the full 78 SC subjects, not the project's existing 5-fold split
(which pools 85 SC+ST subjects and excludes the 15 locked test subjects). The 29 test-set recordings'
subjects (some of which are SC) ARE included here - deliberately: this is a different evaluation, run
once and frozen before any result exists, not a model-selection decision on test subjects, per the
pre-registered protocol (ROADMAP.md R4A/B1 and GPU_PLAN_AND_BENCHMARK.md section 2.3).
"""
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEED = 0
N_FOLDS = 10


def main() -> int:
    df = pd.read_csv(REPO / "processed_sleepedf_mc" / "index.csv")
    df["rec"] = [re.split(r"[\/]", p)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    sc = df[df["rec"].str.startswith("SC")]
    subjects = sorted(sc["subject"].unique())
    if len(subjects) != 78:
        raise SystemExit(f"expected 78 SC subjects, found {len(subjects)}")

    import random
    rng = random.Random(SEED)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    folds = [shuffled[i::N_FOLDS] for i in range(N_FOLDS)]   # near-even: 8x8 + 2x7 = 78
    assert sorted(s for f in folds for s in f) == subjects   # every subject exactly once
    assert len({len(f) for f in folds}) <= 2

    recs_by_subject = sc.groupby("subject")["rec"].apply(list).to_dict()
    out = {"seed": SEED, "n_folds": N_FOLDS, "n_subjects": 78, "n_recordings": len(sc),
           "folds": [{"fold": i, "val_subjects": f, "val_recordings": sorted(
               r for s in f for r in recs_by_subject[s])} for i, f in enumerate(folds)]}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "b1_splits.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for f in out["folds"]:
        print(f"fold {f['fold']}: {len(f['val_subjects'])} subjects, {len(f['val_recordings'])} recordings")
    print(f"wrote results/b1_splits.json ({sum(len(f['val_recordings']) for f in out['folds'])} recordings total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
