"""
Summarise protocol-A cross-validation runs of kaggle_train_student_advanced.py.

    python summarise_cv.py <folder with result_*.json files>

Every arm is compared with "base" (the N4kd architecture and recipe), PAIRED by
(fold, seed): same subjects, same seed, so the difference isolates the
components. The bar is fixed here, before any run exists:

    an arm HELPS if  mean delta kappa >= +0.01  and its 95% CI excludes zero.
    REM and N1 F1 deltas are reported for every arm; they do not change the verdict.

Writes cv_summary.json next to the inputs.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

BAR_KAPPA = 0.01
MEASURES = ("kappa", "accuracy", "macro_f1", "W_f1", "N1_f1", "N2_f1", "N3_f1", "REM_f1")


def t975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except Exception:
        return {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
                9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14}.get(df, 1.96)


def flatten(val):
    out = {"kappa": val["kappa"], "accuracy": val["accuracy"], "macro_f1": val["macro_f1"]}
    out.update({f"{s}_f1": v for s, v in val["per_class_f1"].items()})
    return out


def variant(r):
    """Runs on different data variants are only ever compared with a base of the same variant."""
    return ("_clean" if r.get("exclude_misaligned") else "") + ("_fixlabels" if r.get("corrected_labels") else "")


def load(folder):
    runs = {}
    for p in sorted(Path(folder).rglob("result_*.json")):
        r = json.loads(p.read_text())
        if r.get("cv_fold") is None:
            continue
        key = (r["arm"] + variant(r), r["cv_fold"], r["seed"])
        if key in runs:
            raise SystemExit(f"duplicate run {key}: {p}")
        runs[key] = flatten(r["val"])
    return runs


def summarise(runs):
    by_pair = defaultdict(dict)
    for (arm, fold, seed), m in runs.items():
        by_pair[(fold, seed)][arm] = m
    arms = sorted(a for a in {a for a, _, _ in runs} if not a.startswith("base"))
    out = {"bar_delta_kappa": BAR_KAPPA, "arms": {}}
    for arm in arms:
        suffix = arm[len(arm.split("_clean")[0].split("_fixlabels")[0]):]
        ref = "base" + suffix                                        # same data variant only
        pairs = sorted(k for k, v in by_pair.items() if arm in v and ref in v)
        if not pairs:
            continue
        n = len(pairs)
        stats = {}
        for meas in MEASURES:
            d = [by_pair[k][arm][meas] - by_pair[k][ref][meas] for k in pairs]
            mean = sum(d) / n
            sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
            half = t975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
            stats[meas] = {"mean": mean, "ci95": [mean - half, mean + half], "better": sum(x > 0 for x in d)}
        folds = sorted({f for f, _ in pairs})
        helps = stats["kappa"]["mean"] >= BAR_KAPPA and stats["kappa"]["ci95"][0] > 0
        out["arms"][arm] = {"n_pairs": n, "folds": folds, "complete_cv": folds == [0, 1, 2, 3, 4],
                            "reference": ref, "base_mean_kappa": sum(by_pair[k][ref]["kappa"] for k in pairs) / n,
                            "arm_mean_kappa": sum(by_pair[k][arm]["kappa"] for k in pairs) / n,
                            "delta": stats, "verdict": "HELPS" if helps else "not shown"}
    return out


def main(argv):
    if len(argv) != 1:
        raise SystemExit(__doc__)
    folder = Path(argv[0])
    s = summarise(load(folder))
    if not s["arms"]:
        raise SystemExit("no arm has a paired 'base' run")
    print(f"\n{'arm':<10}{'pairs':>6}{'base k':>8}{'arm k':>8}{'d kappa':>9}   95% CI            "
          f"{'d acc':>7}{'d N1':>7}{'d REM':>7}  verdict")
    for arm, a in s["arms"].items():
        d = a["delta"]
        print(f"{arm:<10}{a['n_pairs']:>6}{a['base_mean_kappa']:>8.4f}{a['arm_mean_kappa']:>8.4f}"
              f"{d['kappa']['mean']:>+9.4f}   [{d['kappa']['ci95'][0]:+.4f}, {d['kappa']['ci95'][1]:+.4f}]"
              f"{100*d['accuracy']['mean']:>+7.2f}{100*d['N1_f1']['mean']:>+7.2f}{100*d['REM_f1']['mean']:>+7.2f}"
              f"  {a['verdict']}{'' if a['complete_cv'] else '  (incomplete CV)'}")
    print(f"\nBar: delta kappa >= +{BAR_KAPPA} with the 95% CI above zero. Accuracy/N1/REM in percentage points.")
    (folder / "cv_summary.json").write_text(json.dumps(s, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
