"""
Summarise the protocol-A cross-validation runs: does EOG help?

Usage:
    python summarise_eog_cv.py <folder>

<folder> holds the result_*.json files downloaded from the Kaggle runs of
kaggle_train_student_kd_eog.py (any nesting). Runs are PAIRED by (fold, seed):
the "eog" and "eegonly" arms of the same fold and seed see the same subjects
and the same initialisation seed, so their difference isolates EOG.

The success bar is fixed here, before any run exists (see README.md):
    mean delta REM F1    >= +0.02
    mean delta kappa     >= +0.01
    95% CI of delta kappa excludes zero
All three must hold. Anything else is reported as "not shown", not as a
partial win.

Writes eog_cv_summary.json next to the inputs. Reads nothing else.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

BAR = {"delta_rem_f1": 0.02, "delta_kappa": 0.01}
MEASURES = ("kappa", "macro_f1", "accuracy", "N1_f1", "REM_f1", "W_f1", "N2_f1", "N3_f1")


def t_crit_975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:                                   # small fallback table
        table = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
                 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14}
        return table.get(df, 1.96)


def flatten(val):
    out = {"kappa": val["kappa"], "macro_f1": val["macro_f1"], "accuracy": val["accuracy"]}
    for s, v in val["per_class_f1"].items():
        out[f"{s}_f1"] = v
    return out


def load(folder):
    runs = {}
    for p in sorted(Path(folder).rglob("result_*.json")):
        r = json.loads(p.read_text())
        if r.get("cv_fold") is None:
            print(f"  skip {p.name}: not a CV run (protocol B)")
            continue
        key = (r["arm"], r["cv_fold"], r["seed"])
        if key in runs:
            raise SystemExit(f"duplicate run {key}: {p}")
        runs[key] = flatten(r["val"])
    return runs


def summarise(runs):
    pairs = defaultdict(dict)
    for (arm, fold, seed), m in runs.items():
        pairs[(fold, seed)][arm] = m
    complete = {k: v for k, v in pairs.items() if "eog" in v and "eegonly" in v}
    unpaired = sorted(k for k, v in pairs.items() if k not in complete)
    if not complete:
        raise SystemExit("no complete (eog, eegonly) pairs found")

    deltas = {m: [complete[k]["eog"][m] - complete[k]["eegonly"][m] for k in sorted(complete)]
              for m in MEASURES}
    n = len(complete)
    stats = {}
    for m, d in deltas.items():
        mean = sum(d) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
        half = t_crit_975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
        stats[m] = {"mean": mean, "sd": sd, "ci95": [mean - half, mean + half],
                    "eog_better": sum(x > 0 for x in d), "n": n}

    arms = {}
    for arm in ("eegonly", "eog"):
        for m in MEASURES:
            v = [complete[k][arm][m] for k in complete]
            arms.setdefault(arm, {})[m] = sum(v) / len(v)

    folds = sorted({f for f, _ in complete})
    passed = (stats["REM_f1"]["mean"] >= BAR["delta_rem_f1"]
              and stats["kappa"]["mean"] >= BAR["delta_kappa"]
              and stats["kappa"]["ci95"][0] > 0)
    return {"n_pairs": n, "folds": folds, "unpaired": unpaired, "arm_means": arms,
            "deltas_eog_minus_eegonly": stats, "bar": BAR,
            "verdict": "EOG HELPS (bar met)" if passed else "EOG BENEFIT NOT SHOWN (bar not met)",
            "complete_cv": folds == [0, 1, 2, 3, 4]}


def main(argv):
    if len(argv) != 1:
        raise SystemExit(__doc__)
    folder = Path(argv[0])
    s = summarise(load(folder))
    print(f"\n{s['n_pairs']} paired runs over folds {s['folds']}"
          + ("" if s["complete_cv"] else "   *** INCOMPLETE - not all 5 folds ***"))
    if s["unpaired"]:
        print(f"  unpaired (ignored): {s['unpaired']}")
    print(f"\n{'measure':<10}{'eegonly':>9}{'eog':>9}{'delta':>9}   95% CI            eog better")
    for m in MEASURES:
        d = s["deltas_eog_minus_eegonly"][m]
        print(f"{m:<10}{s['arm_means']['eegonly'][m]:>9.4f}{s['arm_means']['eog'][m]:>9.4f}"
              f"{d['mean']:>+9.4f}   [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]   {d['eog_better']}/{d['n']}")
    print(f"\nBar: delta REM F1 >= +{BAR['delta_rem_f1']}, delta kappa >= +{BAR['delta_kappa']}, "
          "kappa CI excludes 0")
    print(f"VERDICT: {s['verdict']}")
    out = folder / "eog_cv_summary.json"
    out.write_text(json.dumps(s, indent=1))
    print(f"written {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
