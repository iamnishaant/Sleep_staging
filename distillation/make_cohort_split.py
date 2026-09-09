"""Roadmap item 3.1: the leave-one-cohort-out split.

Train on the SC cohort, test on the whole ST cohort. Sleep-EDFx is two studies,
not one:

    SC  sleep-cassette   78 subjects, 153 recordings   healthy, ageing study
    ST  sleep-telemetry  22 subjects,  44 recordings   mild difficulty falling
                                                       asleep, temazepam,
                                                       different recorder

That is a genuine held-out-population test - different people, different
pharmacology, different hardware - and it is the strongest generalisation claim
available without leaving this dataset.

WHAT THIS IS NOT
----------------
It is NOT a re-run of the headline number. The main test split (15 subjects) is
a subset of both cohorts, so its kappa and this one are measured on different
populations and must never be compared directly. Expect this to be LOWER; the
size of the drop is the finding.

WHAT NOT TO DO WITH IT
----------------------
Do not re-weight SC's class distribution to match the dataset overall. The
"overall" includes ST - the test cohort - so fitting to it leaks the answer into
training. The prior shift is also part of what is being measured: ST has twice
the N3 and a third of the wake (see results/cohort_sc_vs_st.json). Erasing it
would produce a number that looks better and means less. Train on SC as SC is,
and report the shift.

Writes distillation/cohort_split.json. Deterministic given SEED.
"""
import argparse
import collections
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = HERE / "splits.json"
OUT = HERE / "cohort_split.json"
SEED = 42
VAL_FRACTION = 0.20


def build(seed=SEED, val_fraction=VAL_FRACTION):
    sp = json.loads(SPLITS.read_text())
    rbs = sp["recordings_by_subject"]

    sc = sorted(s for s in rbs if s.startswith("SC"))
    st = sorted(s for s in rbs if s.startswith("ST"))
    assert len(sc) + len(st) == len(rbs), "every subject must be SC or ST"
    assert sc and st

    # The SC pool still needs a validation slice, for early stopping only.
    # Subject level, like everything else here.
    rng = random.Random(seed)
    pool = list(sc)
    rng.shuffle(pool)
    n_val = max(1, round(len(pool) * val_fraction))
    val = sorted(pool[:n_val])
    train = sorted(pool[n_val:])

    assert not (set(train) & set(val))
    assert set(train) | set(val) == set(sc)
    assert not ((set(train) | set(val)) & set(st)), "no ST subject may be trained on"

    out = {
        "purpose": "Roadmap item 3.1 - leave-one-cohort-out: train SC, test ST.",
        "unit": "subject",
        "not_comparable_to_headline": (
            "The main 15-subject test split is a subset of both cohorts. This "
            "experiment measures a different population and its kappa must never "
            "be compared directly to the headline 0.7001."),
        "do_not_rebalance": (
            "Do NOT re-weight SC's class distribution toward the dataset overall: "
            "the overall includes ST, the test cohort, so it leaks. The prior shift "
            "IS part of the measurement - ST has ~2x the N3 and ~1/3 the wake."),
        "report": ("kappa AND per-class F1 AND the confusion matrix. Kappa is "
                   "prevalence-sensitive, so alone it cannot separate prior shift "
                   "from transfer failure."),
        "bootstrap": "cluster by SUBJECT over the 22 ST subjects, not the 44 recordings",
        "seed": seed, "val_fraction": val_fraction,
        "train": train,
        "val": val,
        "test": st,
        "counts": {
            "train": {"subjects": len(train),
                      "recordings": sum(len(rbs[s]) for s in train)},
            "val": {"subjects": len(val),
                    "recordings": sum(len(rbs[s]) for s in val)},
            "test": {"subjects": len(st),
                     "recordings": sum(len(rbs[s]) for s in st)},
        },
    }
    return out, rbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out, rbs = build(a.seed, a.val_fraction)
    c = out["counts"]
    print("Leave-one-cohort-out: train on SC, test on ST\n")
    print(f"  {'split':<8}{'cohort':<10}{'subjects':>10}{'recordings':>13}")
    print(f"  {'train':<8}{'SC':<10}{c['train']['subjects']:>10}{c['train']['recordings']:>13}")
    print(f"  {'val':<8}{'SC':<10}{c['val']['subjects']:>10}{c['val']['recordings']:>13}")
    print(f"  {'TEST':<8}{'ST':<10}{c['test']['subjects']:>10}{c['test']['recordings']:>13}")
    print(f"\n  test set is {c['test']['recordings']} recordings from "
          f"{c['test']['subjects']} subjects — larger than the main test split's 29/15,")
    print("  and a genuinely different population.")
    print("\n  Expect kappa to FALL. The size of the drop is the finding.")
    print("  Report per-class F1 and the confusion matrix alongside it: the cohorts")
    print("  differ enough in stage architecture that pooled kappa alone would")
    print("  confound prior shift with transfer failure.")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
