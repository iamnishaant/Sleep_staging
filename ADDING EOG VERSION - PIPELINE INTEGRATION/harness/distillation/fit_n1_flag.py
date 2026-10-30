"""Roadmap 4.1 / track B1: fit a per-epoch confidence flag for N1.

N1 is the model's weakest class and it is representation-bound - no decision
threshold improves its F1 by more than +0.0086, and the human-scorer ceiling is
itself low (scorers agree on N1 roughly 25-45% of the time). Chasing the F1 is
not the move.

But the model already knows where it is unsure. Accuracy when predicting N1
rises monotonically with its own confidence, and that turns the weakest class
from a limitation into something a reader can act on: not "N1 is unreliable" in
general, but "these particular epochs are the unreliable ones, review them".

FITTED ON VALIDATION. The threshold is a decision rule, and a decision rule
fitted on the test split would make the test split a selection set. Validation
chooses it; test only reports what it does.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
STAGES = ["W", "N1", "N2", "N3", "REM"]
N1 = 1


def load(tag: str, split: str):
    d = RES / (f"probs_{tag}_val" if split == "val" else f"probs_{tag}")
    out = {}
    for f in sorted(glob.glob(str(d / "*.npz"))):
        z = np.load(f)
        p, y = z["probs"].astype(np.float64), z["labels"]
        m = y >= 0
        out[Path(f).stem] = (p[m], y[m])
    return out


def curve(P, Y, edges):
    """Accuracy of N1 predictions, binned by the model's own confidence."""
    pred, conf = P.argmax(1), P.max(1)
    sel = pred == N1
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = sel & (conf > lo) & (conf <= hi)
        n = int(m.sum())
        rows.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": n,
                     "accuracy": round(float((Y[m] == N1).mean()), 4) if n else None})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="student_N4kd")
    ap.add_argument("--target-accuracy", type=float, default=0.60,
                    help="required accuracy of the UNFLAGGED N1 predictions on "
                         "validation; the threshold is the lowest confidence that "
                         "achieves it while still flagging a non-trivial share")
    ap.add_argument("--min-flag-fraction", type=float, default=0.10,
                    help="a flag that fires on almost nothing is not a flag")
    ap.add_argument("--out", default=str(RES / "n1_flag.json"))
    a = ap.parse_args()

    val = load(a.model, "val")
    if not val:
        raise SystemExit(f"no cached validation probabilities for {a.model}")
    P = np.concatenate([v[0] for v in val.values()])
    Y = np.concatenate([v[1] for v in val.values()])
    pred, conf = P.argmax(1), P.max(1)
    sel = pred == N1

    print(f"model {a.model}  -  {len(val)} validation recordings, {len(Y):,} epochs")
    print(f"  predicted N1: {int(sel.sum()):,} epochs "
          f"({sel.mean():.1%} of all)   true N1: {(Y == N1).mean():.1%}")
    print(f"  overall accuracy when predicting N1: {(Y[sel] == N1).mean():.1%}\n")

    edges = [0.0, 0.4, 0.6, 0.8, 1.0]
    rows = curve(P, Y, edges)
    print(f"  {'confidence':<14}{'epochs':>9}{'accuracy':>11}")
    for r in rows:
        acc = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "-"
        print(f"  {r['lo']:.1f} - {r['hi']:.1f}     {r['n']:>9,}{acc:>11}")

    # ---- choose the threshold on VALIDATION, and show the trade-off ----
    # The objective is about the epochs a reader is asked to TRUST: unflagged
    # N1 calls must reach the target accuracy, while the flag still fires on a
    # meaningful share. An earlier version targeted accuracy over ALL retained
    # epochs, which is satisfied at the lowest threshold because overall N1
    # accuracy already sits at the target - it flagged 0.4% of epochs and meant
    # nothing.
    grid = np.arange(0.35, 0.901, 0.025)
    trade = []
    for t in grid:
        keep_m, flag_m = sel & (conf >= t), sel & (conf < t)
        if keep_m.sum() < 100 or flag_m.sum() < 30:
            continue
        trade.append({
            "threshold": round(float(t), 3),
            "flagged_fraction": round(float(flag_m.sum() / sel.sum()), 4),
            "accuracy_unflagged": round(float((Y[keep_m] == N1).mean()), 4),
            "accuracy_flagged": round(float((Y[flag_m] == N1).mean()), 4),
        })

    print(f"\n  trade-off on validation (the choice, not an assertion):")
    print(f"    {'thr':<7}{'flagged':>10}{'unflagged acc':>16}{'flagged acc':>14}"
          f"{'separation':>13}")
    for r in trade:
        sep = r["accuracy_unflagged"] - r["accuracy_flagged"]
        print(f"    {r['threshold']:<7.3f}{r['flagged_fraction']:>9.1%}"
              f"{r['accuracy_unflagged']:>16.1%}{r['accuracy_flagged']:>14.1%}"
              f"{sep:>13.1%}")

    ok = [r for r in trade
          if r["accuracy_unflagged"] >= a.target_accuracy
          and r["flagged_fraction"] >= a.min_flag_fraction]
    if ok:
        thr = min(r["threshold"] for r in ok)      # least aggressive that qualifies
        print(f"\n  lowest threshold reaching {a.target_accuracy:.0%} unflagged accuracy "
              f"while flagging at least {a.min_flag_fraction:.0%}: {thr:.3f}")
    else:
        thr = max(trade, key=lambda r: r["accuracy_unflagged"] - r["accuracy_flagged"])["threshold"]
        print(f"\n  no threshold reaches {a.target_accuracy:.0%} unflagged accuracy while "
              f"flagging at least {a.min_flag_fraction:.0%}.")
        print(f"  Falling back to maximum separation: {thr:.3f}. Report the target as "
              f"unmet rather than lowering it silently.")

    kept = sel & (conf >= thr)
    flagged = sel & (conf < thr)
    print(f"\n  threshold {thr:.2f}  (fitted on validation for "
          f"{a.target_accuracy:.0%} accuracy above it)")
    print(f"    unflagged N1: {int(kept.sum()):>6,} epochs, accuracy "
          f"{float((Y[kept] == N1).mean()):.1%}")
    print(f"    FLAGGED N1:   {int(flagged.sum()):>6,} epochs, accuracy "
          f"{float((Y[flagged] == N1).mean()):.1%}")
    print(f"    {flagged.sum() / max(sel.sum(), 1):.1%} of N1 predictions get flagged")

    # what the flagged epochs actually are, when the model is wrong about them
    wrong = flagged & (Y != N1)
    if wrong.sum():
        vc = np.bincount(Y[wrong], minlength=5)
        print(f"\n  when a FLAGGED N1 is wrong, the truth is:")
        for i in np.argsort(-vc):
            if vc[i]:
                print(f"    {STAGES[i]:<4} {vc[i]:>6,}  ({vc[i]/wrong.sum():.1%})")

    out = {
        "purpose": "Roadmap 4.1 - per-epoch confidence flag for N1 predictions.",
        "model": a.model,
        "fitted_on": f"validation split, {len(val)} recordings, {int(len(Y)):,} epochs",
        "never_touched": "the held-out test split - this is a decision rule, and "
                         "fitting one on test would make test a selection set",
        "rule": f"flag an epoch when the predicted stage is N1 and max probability < {thr:.4f}",
        "threshold": round(thr, 4),
        "target_accuracy_unflagged": a.target_accuracy,
        "min_flag_fraction": a.min_flag_fraction,
        "tradeoff_on_validation": trade,
        "objective": ("lowest threshold whose UNFLAGGED N1 predictions reach the target accuracy while the flag still fires on at least min_flag_fraction. An earlier objective targeted accuracy over all retained epochs and was degenerate - overall N1 accuracy already equals the target, so it selected the lowest grid point and flagged 0.4% of epochs."),
        "validation": {
            "n1_predictions": int(sel.sum()),
            "accuracy_all_n1": round(float((Y[sel] == N1).mean()), 4),
            "accuracy_unflagged": round(float((Y[kept] == N1).mean()), 4),
            "accuracy_flagged": round(float((Y[flagged] == N1).mean()), 4),
            "fraction_flagged": round(float(flagged.sum() / max(sel.sum(), 1)), 4),
            "confidence_curve": rows,
        },
        "interpretation": (
            "N1 cannot be fixed by a decision rule - it is representation-bound. "
            "What the model can do is say which of its N1 calls are the doubtful "
            "ones, which is a defensible reason to review specific epochs rather "
            "than a claim that N1 was solved."),
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {a.out}")
    print("  test split NOT touched. Report what this rule does there once, afterwards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
