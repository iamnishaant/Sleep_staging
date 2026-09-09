"""Roadmap item 0.1: 5-fold subject-level CV folds over the 85 non-test subjects.

The folds are drawn at SUBJECT level, stratified by cohort, and the 15 held-out
test subjects are excluded entirely. This is NOT nested CV: it is one 5-fold CV
for model selection, plus the frozen test split for the final report.

Why subject level, restated because the project got this wrong once already:
Sleep-EDFx records two nights per person. A recording-level fold would put one
of a subject's nights in train and the other in validation, which is the same
leakage that invalidated the inherited model - just at a smaller scale.

Writes distillation/cv_folds.json. Deterministic given SEED.
"""
import argparse
import collections
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = HERE / "splits.json"
OUT = HERE / "cv_folds.json"
SEED = 42
K = 5


def build(k=K, seed=SEED):
    sp = json.loads(SPLITS.read_text())
    rbs = sp["recordings_by_subject"]

    test = set(sp["splits"]["test"])
    pool = sorted(set(sp["splits"]["train"]) | set(sp["splits"]["val"]))
    assert not (set(pool) & test), "test subjects must not enter the CV pool"
    assert len(pool) + len(test) == len(rbs), "pool + test must cover every subject"

    # stratify by cohort so each fold carries a representative ST share
    by_cohort = collections.defaultdict(list)
    for s in pool:
        by_cohort[s[:2]].append(s)

    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    # One continuous round-robin counter ACROSS cohorts. Dealing each cohort
    # from its own zero leaves each cohort's remainder on the low folds, which
    # compounds: 66 SC + 19 ST that way gives fold sizes [18,17,17,16,17].
    # Carrying the counter makes every fold exactly ceil/floor of 85/5.
    pos = 0
    for cohort in sorted(by_cohort):
        subs = sorted(by_cohort[cohort])
        rng.shuffle(subs)
        for s in subs:
            folds[pos % k].append(s)
            pos += 1

    folds = [sorted(f) for f in folds]

    # ---- invariants ----
    flat = [s for f in folds for s in f]
    assert len(flat) == len(set(flat)) == len(pool), "folds must partition the pool exactly"
    assert not (set(flat) & test), "no test subject may appear in any fold"
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1, f"folds unbalanced: {sizes}"

    out = {
        "purpose": "Roadmap item 0.1 - 5-fold subject-level CV for model selection.",
        "not_nested_cv": (
            "One 5-fold CV over the non-test subjects for selection, plus the frozen "
            "15-subject test split for the final report. Nested CV would run a CV loop "
            "inside each fold of an outer CV loop; this does not."),
        "unit": "subject",
        "why_subject_level": (
            "Sleep-EDFx records two nights per person. Recording-level folds would split "
            "a subject across train and validation - the same leakage that invalidated "
            "the inherited model, at smaller scale. It is also why the project's "
            "confidence intervals were corrected on 29 Aug 2026 to cluster by subject."),
        "k": k, "seed": seed,
        "n_subjects": len(pool),
        "n_recordings": sum(len(rbs[s]) for s in pool),
        "test_subjects_excluded": sorted(test),
        "folds": [],
    }
    for i, f in enumerate(folds):
        c = collections.Counter(s[:2] for s in f)
        out["folds"].append({
            "fold": i,
            "n_subjects": len(f),
            "n_recordings": sum(len(rbs[s]) for s in f),
            "cohort": {"SC": c["SC"], "ST": c["ST"]},
            "val_subjects": f,
            "train_subjects": sorted(s for j, g in enumerate(folds) if j != i for s in g),
        })
    return out, rbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    a = ap.parse_args()

    out, rbs = build(a.k, a.seed)

    print(f"{a.k}-fold subject-level CV over {out['n_subjects']} non-test subjects "
          f"({out['n_recordings']} recordings)")
    print(f"held-out test subjects excluded: {len(out['test_subjects_excluded'])}\n")
    print(f"  {'fold':<6}{'val subj':>9}{'val rec':>9}{'SC':>5}{'ST':>5}{'train subj':>12}")
    for f in out["folds"]:
        print(f"  {f['fold']:<6}{f['n_subjects']:>9}{f['n_recordings']:>9}"
              f"{f['cohort']['SC']:>5}{f['cohort']['ST']:>5}{len(f['train_subjects']):>12}")

    st_share = [f["cohort"]["ST"] / f["n_subjects"] for f in out["folds"]]
    print(f"\n  ST share per fold: {', '.join(f'{x:.1%}' for x in st_share)}")
    print(f"  pool ST share:     "
          f"{sum(f['cohort']['ST'] for f in out['folds'])/out['n_subjects']:.1%}")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
