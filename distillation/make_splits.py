"""
Phase 1, task 3 - subject-level train/val/test splits.

WHY SUBJECT-LEVEL, NOT RECORDING-LEVEL
--------------------------------------
Sleep-EDFx records most subjects on two consecutive nights:
    SC4001E0 / SC4002E0  -> SC subject 00, nights 1 and 2
    ST7011J0 / ST7012J0  -> ST subject 01, nights 1 and 2
Splitting per recording puts both nights of one subject on opposite sides of
the split. The subject key is the first 5 characters, which works for both
cohorts because each is <3-char prefix><2-digit subject><night><suffix>.

COHORT STRATIFICATION
---------------------
SC (Sleep Cassette, 78 subjects) is a healthy-ageing study; ST (Sleep
Telemetry, 22 subjects) is a temazepam trial with different hardware and a
different population. With only 22 ST subjects an unstratified 15% test split
could contain anywhere from 0 to 7 of them. Splits are therefore stratified on
cohort so each split holds a representative mix.

TEACHER PROVENANCE
------------------
The teacher was trained with a *recording-level* split:
    train_test_split(range(197), test_size=0.2, random_state=42)
which leaked 30 of its 35 validation subjects into training. This script
reconstructs that split and annotates every subject with whether the teacher
ever saw it, so Phase 2 can quantify how much of the reported kappa=0.6663 is
leakage. See README_V2.md section 4.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_STAGES = ["W", "N1", "N2", "N3", "REM"]


def recording_id(path_value: str) -> str:
    leaf = str(path_value).replace("\\", "/").rsplit("/", 1)[-1]
    return leaf[:-3] if leaf.endswith(".pt") else leaf


def subject_of(rec: str) -> str:
    """First 5 chars. 'SC4001E0-PSG' -> 'SC400'; 'ST7011J0-PSG' -> 'ST701'."""
    return rec[:5]


def cohort_of(rec: str) -> str:
    return rec[:2]


def reconstruct_teacher_split(n_recordings: int, seed: int = 42):
    """Reproduce the teacher's recording-level 80/20 split exactly."""
    idx = list(range(n_recordings))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=seed)
    return set(train_idx), set(val_idx)


def summarise(subjects, rec_by_subject, stages_by_rec):
    recs = [r for s in subjects for r in rec_by_subject[s]]
    counter: Counter = Counter()
    for r in recs:
        counter.update(stages_by_rec[r])
    total = sum(counter.values())
    return {
        "n_subjects": len(subjects),
        "n_recordings": len(recs),
        "n_epochs": total,
        "cohorts": dict(Counter(cohort_of(r) for r in recs)),
        "class_counts": {s: counter.get(s, 0) for s in CANONICAL_STAGES},
        "class_fractions": {s: round(counter.get(s, 0) / total, 5) if total else 0.0
                            for s in CANONICAL_STAGES},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    recs = [recording_id(p) for p in df["tensor_path"]]
    stages_by_rec = {r: str(s).split()
                     for r, s in zip(recs, df["stage_sequence"])}

    rec_by_subject: dict[str, list[str]] = defaultdict(list)
    for r in recs:
        rec_by_subject[subject_of(r)].append(r)
    for s in rec_by_subject:
        rec_by_subject[s].sort()

    subjects = sorted(rec_by_subject)
    cohorts = [s[:2] for s in subjects]
    print(f"{len(recs)} recordings -> {len(subjects)} subjects "
          f"({Counter(cohorts)})")
    multi = sum(1 for s in subjects if len(rec_by_subject[s]) > 1)
    print(f"  subjects with >1 night: {multi}  "
          f"(these are exactly what a recording-level split would leak)")

    # ---- stratified subject-level split -------------------------------------
    train_val, test = train_test_split(
        subjects, test_size=args.test_frac, random_state=args.seed, stratify=cohorts
    )
    tv_cohorts = [s[:2] for s in train_val]
    val_adj = args.val_frac / (1.0 - args.test_frac)
    train, val = train_test_split(
        train_val, test_size=val_adj, random_state=args.seed, stratify=tv_cohorts
    )
    train, val, test = sorted(train), sorted(val), sorted(test)

    # ---- assertions on SUBJECT ids, not recordings --------------------------
    assert not (set(train) & set(val)), "subject overlap: train/val"
    assert not (set(train) & set(test)), "subject overlap: train/test"
    assert not (set(val) & set(test)), "subject overlap: val/test"
    assert set(train) | set(val) | set(test) == set(subjects), "subjects lost"
    all_recs = {r for s in train + val + test for r in rec_by_subject[s]}
    assert all_recs == set(recs), "recordings lost"
    print("  assertions passed: no subject appears in two splits\n")

    # ---- teacher provenance -------------------------------------------------
    t_train_idx, _ = reconstruct_teacher_split(len(recs), seed=42)
    teacher_train_recs = {recs[i] for i in t_train_idx}
    teacher_seen_subject = {
        s: any(r in teacher_train_recs for r in rec_by_subject[s]) for s in subjects
    }

    splits = {"train": train, "val": val, "test": test}
    summary = {name: summarise(subs, rec_by_subject, stages_by_rec)
               for name, subs in splits.items()}

    for name in ("train", "val", "test"):
        s = summary[name]
        clean = [x for x in splits[name] if not teacher_seen_subject[x]]
        s["subjects_unseen_by_teacher"] = len(clean)
        print(f"{name.upper():<6} {s['n_subjects']:>3} subj | "
              f"{s['n_recordings']:>3} rec | {s['n_epochs']:>7,} epochs | "
              f"{s['cohorts']} | teacher-unseen subj: {len(clean)}")
        print("        class %: " + "  ".join(
            f"{k}={100 * v:.1f}" for k, v in s["class_fractions"].items()))

    n_clean_test = summary["test"]["subjects_unseen_by_teacher"]
    print(f"\nTEACHER LEAKAGE INTO TEST SPLIT")
    print(f"  test subjects the teacher trained on : "
          f"{summary['test']['n_subjects'] - n_clean_test} / {summary['test']['n_subjects']}")
    print(f"  genuinely unseen by teacher          : {n_clean_test}")
    if n_clean_test < summary["test"]["n_subjects"]:
        print("  -> teacher metrics on this split are OPTIMISTIC. Phase 2 must "
              "also report the teacher-clean subset separately.")

    globally_clean = sorted(s for s in subjects if not teacher_seen_subject[s])
    print(f"\n  subjects never seen by teacher, repo-wide: {len(globally_clean)} "
          f"({sum(len(rec_by_subject[s]) for s in globally_clean)} recordings)")
    print(f"    {globally_clean}")

    payload = {
        "seed": args.seed,
        "granularity": "subject",
        "subject_key": "recording_id[:5]",
        "stratified_by": "cohort (SC/ST)",
        "fractions": {"train": round(1 - args.val_frac - args.test_frac, 4),
                      "val": args.val_frac, "test": args.test_frac},
        "splits": splits,
        "recordings_by_subject": {s: rec_by_subject[s] for s in subjects},
        "summary": summary,
        "teacher_provenance": {
            "teacher_split": "train_test_split(range(197), test_size=0.2, random_state=42)",
            "teacher_split_granularity": "recording (LEAKY - see README_V2.md s4)",
            "subject_seen_by_teacher": teacher_seen_subject,
            "subjects_never_seen_by_teacher": globally_clean,
            "n_recordings_never_seen": sum(len(rec_by_subject[s]) for s in globally_clean),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
