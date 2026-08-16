"""
Prior correction, fitted on VALIDATION and applied to test.

Training used focal loss with inverse-frequency alpha. That shifts the model's
implicit class prior: on the held-out test split the retrained teacher predicts
N1 2.61x more often than N1 occurs, and 30% of true N2 epochs are called N1.

Dividing the softmax by alpha**g and re-taking the argmax undoes that shift.
g=0 is no correction, g=1 fully undoes the training weights, g=0.5 is the
principled "sqrt-alpha" midpoint.

METHOD NOTE
-----------
g is fitted on the VALIDATION split only, then applied unchanged to test. An
earlier exploratory sweep on test suggested g=0.55 reaches macro-F1 0.6342 -
that number is an upper bound for reporting only and is never used, because
tuning on test is the exact error this project exists to catch.

This is a post-processing step, not retraining. It costs no GPU time and should
be applied to every model downstream, including the distilled student.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).parent))
from teacher_model import STAGE_TO_IDX  # noqa: E402
from eval_final import load_any, recording_id  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
WINDOW_SIZE = 256
LAB = list(range(5))


@torch.no_grad()
def probs_for(model, t_path, s_path, stages):
    t = torch.load(t_path, map_location="cpu").float()
    s = torch.load(s_path, map_location="cpu").float()
    y = np.array([STAGE_TO_IDX[x] for x in stages], dtype=np.int64)
    n = min(t.shape[0], s.shape[0], len(y))
    P = np.empty((n, 5), dtype=np.float32)
    for st in range(0, n, WINDOW_SIZE):
        en = min(st + WINDOW_SIZE, n)
        P[st:en] = F.softmax(model(t[st:en].unsqueeze(0),
                                   s[st:en].unsqueeze(0)).float(), -1).squeeze(0).numpy()
    del t, s
    return P, y[:n]


def collect(model, df, row_of, recs, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    Ps, ys = [], []
    t0 = time.time()
    for i, r in enumerate(recs):
        f = cache / f"{r}.npz"
        if f.exists():
            d = np.load(f)
            P, y = d["probs"], d["labels"]
        else:
            row = df.iloc[row_of[r]]
            P, y = probs_for(model, REPO_ROOT / row["tensor_path"],
                             REPO_ROOT / row["spectral"],
                             str(row["stage_sequence"]).split())
            np.savez_compressed(f, probs=P, labels=y)
            print(f"    [{i+1:>2}/{len(recs)}] {r} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
        Ps.append(P); ys.append(y)
    return np.concatenate(Ps), np.concatenate(ys)


def metrics(P, y, w, g):
    pred = (P / (w ** g)).argmax(1)
    _, _, f1, _ = precision_recall_fscore_support(y, pred, labels=LAB, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred, labels=LAB)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=LAB, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=LAB, zero_division=0)),
        "per_class_f1": {STAGES[i]: float(f1[i]) for i in LAB},
        "prediction_ratio": {STAGES[i]: float((pred == i).sum() / max((y == i).sum(), 1))
                             for i in LAB},
        "confusion_matrix": confusion_matrix(y, pred, labels=LAB).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = Path(__file__).parent / "results"
    ap.add_argument("--checkpoint", default=str(res / "retrained" / "teacher_best.pt"))
    ap.add_argument("--tag", default="retrained",
                    help="Cache/output tag so different models do not collide.")
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--select-on", default="macro_f1",
                    choices=["macro_f1", "kappa", "accuracy"])
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    recs = [recording_id(p) for p in df["tensor_path"]]
    row_of = {r: i for i, r in enumerate(recs)}
    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    rbs = sp["recordings_by_subject"]
    val_recs = sorted(r for s in sp["splits"]["val"] for r in rbs[s])
    test_recs = sorted(r for s in sp["splits"]["test"] for r in rbs[s])

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    alpha = np.array(ck["alpha"], dtype=np.float64)
    print(f"checkpoint : {Path(args.checkpoint).name} (epoch {ck.get('epoch')})")
    print(f"focal alpha: " + ", ".join(f"{s}={a:.4f}" for s, a in zip(STAGES, alpha)))

    model = load_any(Path(args.checkpoint))
    torch.set_grad_enabled(False)

    print(f"\ncollecting VALIDATION probabilities ({len(val_recs)} recordings)")
    Pv, yv = collect(model, df, row_of, val_recs, res / f"probs_{args.tag}_val")
    print(f"  {len(yv):,} epochs")
    print(f"collecting TEST probabilities ({len(test_recs)} recordings)")
    Pt, yt = collect(model, df, row_of, test_recs, res / f"probs_{args.tag}")
    print(f"  {len(yt):,} epochs")

    # ---- fit g on VALIDATION ONLY -----------------------------------------
    grid = np.round(np.linspace(0.0, 1.5, 61), 3)
    val_curve = [(float(g), metrics(Pv, yv, alpha, g)) for g in grid]
    best_g, best_val = max(val_curve, key=lambda kv: kv[1][args.select_on])
    print(f"\nfitted on VALIDATION, selecting on {args.select_on}")
    print(f"  best g = {best_g:.3f}   (val {args.select_on} = {best_val[args.select_on]:.4f})")
    print(f"  note: g=0.5 is the principled sqrt-alpha choice; "
          f"val {args.select_on} there = "
          f"{dict(val_curve)[0.5][args.select_on]:.4f}")

    # ---- apply to TEST -----------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'setting':<34}{'acc':>8}{'kappa':>8}{'mF1':>8}{'N1 F1':>8}{'N2 F1':>8}{'N1 ratio':>10}")
    print("-" * 78)
    rows = {}
    for tag, g in [("as-trained (g=0)", 0.0),
                   ("sqrt-alpha (g=0.5, principled)", 0.5),
                   (f"val-fitted (g={best_g:.3f})", best_g)]:
        m = metrics(Pt, yt, alpha, g)
        rows[tag] = {"g": g, **m}
        print(f"{tag:<34}{m['accuracy']:>8.4f}{m['kappa']:>8.4f}{m['macro_f1']:>8.4f}"
              f"{m['per_class_f1']['N1']:>8.4f}{m['per_class_f1']['N2']:>8.4f}"
              f"{m['prediction_ratio']['N1']:>9.2f}x")
    print("=" * 78)

    chosen = rows[f"val-fitted (g={best_g:.3f})"]
    base = rows["as-trained (g=0)"]
    print(f"\nDELTA from validation-fitted correction (test split, honest):")
    for k in ("accuracy", "kappa", "macro_f1"):
        print(f"  {k:<12} {base[k]:.4f} -> {chosen[k]:.4f}   ({chosen[k]-base[k]:+.4f})")
    print("  per-class F1:")
    for s in STAGES:
        a, b = base["per_class_f1"][s], chosen["per_class_f1"][s]
        print(f"    {s:<5} {a:.4f} -> {b:.4f}   ({b-a:+.4f})")
    print("  prediction ratio (1.00x = calibrated frequency):")
    for s in STAGES:
        a, b = base["prediction_ratio"][s], chosen["prediction_ratio"][s]
        print(f"    {s:<5} {a:.2f}x -> {b:.2f}x")

    out = {
        "method": "divide softmax by alpha**g, argmax",
        "alpha": alpha.tolist(),
        "g_fitted_on": "validation split only",
        "selection_metric": args.select_on,
        "g_selected": best_g,
        "validation_curve": {str(g): {"macro_f1": m["macro_f1"], "kappa": m["kappa"],
                                      "accuracy": m["accuracy"]}
                             for g, m in val_curve},
        "validation_at_selected_g": best_val,
        "test_results": rows,
        "warning": ("g was fitted on validation and applied unchanged to test. "
                    "An exploratory test-tuned sweep gave g=0.55 / macro-F1 0.6342; "
                    "that is an upper bound for reporting only and must not be used."),
    }
    p = res / f"prior_correction_{args.tag}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
