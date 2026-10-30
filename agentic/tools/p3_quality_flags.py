"""V3 (PENDING.md): formal test that the signal-quality flags pick out worse nights.

    python -m agentic.tools.p3_quality_flags

The P2 review could only report "kappa 0.587 flagged vs 0.689 unflagged, n = 5". This turns that into a
test with an interval, on both splits that have cached predictions:

  * train (137 recordings) - pass 1 on the reference span, flagged by the frozen quality thresholds;
  * dev   (31 recordings)  - pass 2 on the estimated span (the deployed path), flagged at runtime.

Three statistics, all bootstrapped or permuted over RECORDINGS (not epochs):
  1. mean kappa difference, unflagged minus flagged, with a 95% percentile CI;
  2. a permutation p-value for "flagged nights are worse";
  3. Spearman rho between the continuous unusable fraction and kappa (the flag is only a threshold on it).

Writes agentic/reports/v3_quality_flags.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import cohen_kappa_score

from .. import PACKAGE_DIR
from ..core import atomic
from ..tools.p2_calibrate import CACHE, ref_span, splits

SEED, N_BOOT, N_PERM = 42, 10000, 10000


def mean_difference(kappa: np.ndarray, flagged: np.ndarray) -> float | None:
    if not flagged.any() or flagged.all():
        return None
    return float(kappa[~flagged].mean() - kappa[flagged].mean())


def bootstrap(kappa: np.ndarray, flagged: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> list[float] | None:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(kappa), len(kappa))
        d = mean_difference(kappa[i], flagged[i])
        if d is not None:
            diffs.append(d)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else None


def permutation_p(kappa: np.ndarray, flagged: np.ndarray, n: int = N_PERM, seed: int = SEED) -> float | None:
    observed = mean_difference(kappa, flagged)
    if observed is None:
        return None
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        d = mean_difference(kappa, rng.permutation(flagged))
        if d is not None and d >= observed:
            hits += 1
    return float((hits + 1) / (n + 1))


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": int(len(x))}


def summarise(name: str, kappa: np.ndarray, flagged: np.ndarray, quality: np.ndarray) -> dict:
    kappa, flagged, quality = map(np.asarray, (kappa, flagged, quality))
    return {"split": name, "n": int(len(kappa)), "n_flagged": int(flagged.sum()),
            "kappa_flagged": float(kappa[flagged].mean()) if flagged.any() else None,
            "kappa_unflagged": float(kappa[~flagged].mean()) if (~flagged).any() else None,
            "mean_difference_unflagged_minus_flagged": mean_difference(kappa, flagged),
            "bootstrap_95_ci": bootstrap(kappa, flagged),
            "permutation_p_flagged_worse": permutation_p(kappa, flagged),
            "spearman_quality_vs_kappa": spearman(quality, kappa)}


def from_cache(recs: list[str], warn: float, review: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train split: pass-1 argmax on the reference span, flagged by the frozen thresholds."""
    kappa, flagged, quality = [], [], []
    for rec in recs:
        d = np.load(CACHE / f"{rec}.npz")
        labels, p1, unusable = d["labels"], d["pass1"].argmax(1), d["unusable"]
        s, e = ref_span(labels)
        y, p = labels[s:e], p1[s:e]
        m = y >= 0
        kappa.append(cohen_kappa_score(y[m], p[m], labels=range(5)))
        frac = float(np.mean(unusable[s:e]))
        quality.append(frac)
        flagged.append(frac > warn or frac > review)
    return np.array(kappa), np.array(flagged), np.array(quality)


def from_calibration_dev() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = json.loads((PACKAGE_DIR / "reports" / "p2_calibration.json").read_text())["rows"]
    rows = [r for r in rows if r.get("code") != "R04" and "kappa_est" in r]
    return (np.array([r["kappa_est"] for r in rows]),
            np.array([r["quality_flag"] is not None for r in rows]),
            np.array([r["quality"]["unusable_fraction"] for r in rows]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PACKAGE_DIR / "reports" / "v3_quality_flags.json"))
    a = ap.parse_args(argv)
    calibration = json.loads((PACKAGE_DIR / "reports" / "p2_calibration.json").read_text())
    q = calibration["quality"]
    train, _dev, _test = splits()

    train_stats = summarise("train", *from_cache(train, q["warn_fraction"], q["review_fraction"]))
    dev_stats = summarise("dev", *from_calibration_dev())
    result = {
        "question": "do the frozen signal-quality flags pick out worse nights?",
        "thresholds": {"warn_fraction": q["warn_fraction"], "review_fraction": q["review_fraction"],
                       "calibrated_on": q["calibrated_on"]},
        "method": {"unit": "recording", "bootstrap": N_BOOT, "permutation": N_PERM, "seed": SEED,
                   "kappa": "Cohen's kappa against expert labels, scored epochs only"},
        "caveat": "train uses pass 1 on the reference span (no Path B pass 2); dev uses the deployed estimated span.",
        "results": {"train": train_stats, "dev": dev_stats},
        "wall_s": None,
    }
    atomic.write_json(a.out, result)
    print(json.dumps(result["results"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
