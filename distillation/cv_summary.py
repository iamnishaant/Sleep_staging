"""Roadmap item 0.1: summarise the 5-fold CV runs.

Reads distillation/results/students/student_<EXP>_cv<k>/training_metrics.jsonl
and reports the CV mean and across-fold spread.

WHAT THIS DELIBERATELY DOES NOT REPORT
--------------------------------------
A confidence interval. CV folds share training data - each model sees 80% of the
same subjects - so the fold estimates are correlated and the independence that
1/sqrt(n) assumes does not hold. There is in fact no unbiased estimator of
k-fold CV variance. The across-fold sd below is a descriptive spread, NOT a
standard error, and must not be turned into one. An earlier draft of the roadmap
promised the selection interval would tighten to a specific number by 1/sqrt(n);
that claim was withdrawn, and this script exists partly so it cannot come back.

Each fold's reported epoch is the one the trainer itself selected (best
validation macro-F1), so this summarises exactly what early stopping kept.
"""
import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDENTS = HERE / "results" / "students"
FOLDS = HERE / "cv_folds.json"


def read_fold(exp, k, root):
    d = root / f"student_{exp}_cv{k}"
    f = d / "training_metrics.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    best = max(rows, key=lambda r: r["macro_f1"])
    return {"fold": k, "dir": d.name, "epochs_run": len(rows),
            "best_epoch": best["epoch"], "macro_f1": best["macro_f1"],
            "kappa": best["kappa"], "accuracy": best.get("accuracy"),
            "per_class_f1": best.get("per_class_f1", {}),
            "cv_fold_in_log": best.get("cv_fold", "not recorded")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="N4kd",
                    help="base EXPERIMENT name, without the _cv<k> suffix")
    ap.add_argument("--root", default=str(STUDENTS))
    ap.add_argument("--out", default=str(HERE / "results" / "cv_summary.json"))
    a = ap.parse_args()
    root = Path(a.root)

    cv = json.loads(FOLDS.read_text())
    k = cv["k"]

    got = [r for r in (read_fold(a.experiment, i, root) for i in range(k)) if r]
    missing = [i for i in range(k) if not any(r["fold"] == i for r in got)]

    print(f"5-fold subject-level CV  -  experiment {a.experiment}")
    print(f"reading {root}\n")
    if missing:
        print(f"  MISSING folds {missing} - run them, or this summary is partial\n")
    if not got:
        print("  nothing to summarise yet.")
        print(f"  expected: {root}/student_{a.experiment}_cv0/training_metrics.jsonl  (etc)")
        return

    print(f"  {'fold':<6}{'subj':>6}{'best ep':>9}{'epochs':>8}{'macro-F1':>11}{'kappa':>9}")
    for r in got:
        n = cv["folds"][r["fold"]]["n_subjects"]
        print(f"  {r['fold']:<6}{n:>6}{r['best_epoch']:>9}{r['epochs_run']:>8}"
              f"{r['macro_f1']:>11.4f}{r['kappa']:>9.4f}")

    out = {"experiment": a.experiment, "k": k, "folds_present": [r["fold"] for r in got],
           "folds_missing": missing, "per_fold": got}

    if len(got) >= 2:
        print()
        for metric in ("macro_f1", "kappa"):
            vals = [r[metric] for r in got]
            m, sd = st.fmean(vals), st.stdev(vals)
            out[metric] = {"mean": round(m, 4), "across_fold_sd": round(sd, 4),
                           "min": round(min(vals), 4), "max": round(max(vals), 4),
                           "n_folds": len(vals)}
            print(f"  {metric:<9} mean {m:.4f}   across-fold sd {sd:.4f}   "
                  f"range [{min(vals):.4f}, {max(vals):.4f}]")

        print()
        print("  The sd above is a DESCRIPTIVE SPREAD, not a standard error.")
        print("  Folds share training data, so they are correlated and there is no")
        print("  unbiased estimator of k-fold CV variance. Do not build a CI from it.")

        if len(got) == k:
            sd = out["kappa"]["across_fold_sd"]
            print()
            print(f"  Practical bar: a change is worth believing if it moves the CV mean")
            print(f"  by more than one across-fold sd ({sd:.4f} kappa). Anything smaller")
            print(f"  is not distinguishable from fold-to-fold noise.")
    else:
        print("\n  (need at least 2 folds for a spread)")

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
