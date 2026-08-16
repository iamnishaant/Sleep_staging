"""
Phase 6 - evaluate the distilled and baseline students.

Two jobs:

1. REPRODUCE the validation numbers Kaggle reported. If a locally computed
   val macro-F1 does not match the training log, something is wrong with the
   checkpoint, the model class, or the split - and every downstream number
   would be untrustworthy. This runs first and is checked explicitly.

2. Produce the held-out TEST numbers, which are the honest figures.

Per-class F1 is reported on UNSEEN subjects only (project convention).
Per-recording kappa and mean prediction entropy are saved for the night-level
confidence question and for the reliability table.

The student class is imported by exec-ing kaggle_train_student.py rather than
re-declared here, so the architecture is guaranteed identical to the one that
produced the checkpoints - a re-declaration could drift.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}
LAB = list(range(5))
WINDOW_SIZE = 256


def load_student_class():
    """Import the exact class used for training, by exec-ing the Kaggle script."""
    src = (Path(__file__).parent / "kaggle_train_student.py").read_text(encoding="utf-8")
    src = src.replace("\nmain()", "")
    ns: dict = {}
    exec(compile(src, "kaggle_train_student.py", "exec"), ns)
    return ns["StudentSleepStagingModel"]


def load_student(ckpt_path: Path):
    Student = load_student_class()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    model = Student(
        embed=args.get("embed_dim", 64), heads=args.get("heads", 2),
        layers=args.get("layers", 2), dropout=args.get("dropout", 0.2),
        n_tokens=args.get("n_tokens", 1),
        use_spectral=ck.get("use_spectral", True),
    )
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{ckpt_path.name} mismatch:\n  missing {sorted(missing)}"
                           f"\n  unexpected {sorted(unexpected)}")
    model.eval()
    return model, ck


@torch.no_grad()
def predict(model, t_path: Path, s_path: Path, stages: list[str]):
    """Non-overlapping 256-epoch windows, matching how validation ran in training."""
    t = torch.load(t_path, map_location="cpu").float()
    s = torch.load(s_path, map_location="cpu").float()
    y = np.array([STAGE_TO_IDX[x] for x in stages], dtype=np.int64)
    n = min(t.shape[0], s.shape[0], len(y))
    P = np.empty((n, 5), dtype=np.float32)
    for st in range(0, n, WINDOW_SIZE):
        en = min(st + WINDOW_SIZE, n)
        lg = model(t[st:en].unsqueeze(0), s[st:en].unsqueeze(0))
        P[st:en] = F.softmax(lg.float(), -1).squeeze(0).numpy()
    del t, s
    return P, y[:n]


def metrics_from(P, y, weights=None, g=0.0):
    pred = (P / (weights ** g)).argmax(1) if (weights is not None and g) else P.argmax(1)
    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=LAB, zero_division=0)
    ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
    return {
        "n_epochs": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred, labels=LAB)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=LAB, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=LAB, zero_division=0)),
        "per_class": {STAGES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f1[i]), "support": int(sup[i])} for i in LAB},
        "prediction_ratio": {STAGES[i]: float((pred == i).sum() / max((y == i).sum(), 1))
                             for i in LAB},
        "confusion_matrix": confusion_matrix(y, pred, labels=LAB).tolist(),
        "mean_entropy_nats": float(ent.mean()),
    }


def collect(model, df, row_of, recs, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    per_rec, Ps, ys = {}, [], []
    t0 = time.time()
    for i, r in enumerate(recs):
        f = cache / f"{r}.npz"
        if f.exists():
            d = np.load(f); P, y = d["probs"], d["labels"]
        else:
            row = df.iloc[row_of[r]]
            P, y = predict(model, REPO_ROOT / row["tensor_path"],
                           REPO_ROOT / row["spectral"],
                           str(row["stage_sequence"]).split())
            np.savez_compressed(f, probs=P, labels=y)
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(recs)}] ({(time.time()-t0)/60:.1f} min)", flush=True)
        pred = P.argmax(1)
        ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
        per_rec[r] = {"subject": r[:5], "cohort": r[:2], "n_epochs": int(len(y)),
                      "kappa": float(cohen_kappa_score(y, pred, labels=LAB)),
                      "accuracy": float(accuracy_score(y, pred)),
                      "mean_entropy_nats": float(ent.mean())}
        Ps.append(P); ys.append(y)
    return np.concatenate(Ps), np.concatenate(ys), per_rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = Path(__file__).parent / "results"
    ap.add_argument("--students", nargs="+",
                    default=["student_distilled_E0", "student_baseline_E0"])
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    row_of = {r: i for i, r in enumerate(df["rec"])}
    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    rbs = sp["recordings_by_subject"]
    val_recs = sorted(r for s in sp["splits"]["val"] for r in rbs[s])
    test_recs = sorted(r for s in sp["splits"]["test"] for r in rbs[s])

    report = {}
    torch.set_grad_enabled(False)
    for name in args.students:
        d = res / "students" / name
        model, ck = load_student(d / "student_best.pt")
        npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logged = ck["best_macro_f1"]
        print(f"\n{'='*74}\n{name}\n{'='*74}")
        print(f"  epoch {ck['epoch']} | alpha={ck['alpha']} T={ck['T']} "
              f"cw={ck['class_weight_power']} spectral={ck['use_spectral']}")
        print(f"  {npar:,} params (teacher 649,229 -> {649229/npar:.2f}x)")

        # ---- 1. REPRODUCE VALIDATION -------------------------------------
        print(f"\n  reproducing validation ({len(val_recs)} recordings)...")
        Pv, yv, _ = collect(model, df, row_of, val_recs, res / f"probs_{name}_val")
        mv = metrics_from(Pv, yv)
        delta = mv["macro_f1"] - logged
        agree = abs(delta) < 5e-3
        print(f"  logged in training : macro-F1 {logged:.4f}")
        print(f"  recomputed locally : macro-F1 {mv['macro_f1']:.4f}   "
              f"(delta {delta:+.4f})  {'MATCH' if agree else '*** MISMATCH ***'}")
        if not agree:
            print("  -> checkpoint, model class or split disagree. Downstream numbers"
                  " would be untrustworthy.")

        # ---- 2. HELD-OUT TEST ---------------------------------------------
        print(f"\n  held-out test ({len(test_recs)} recordings)...")
        Pt, yt, per_rec = collect(model, df, row_of, test_recs, res / f"probs_{name}")
        mt = metrics_from(Pt, yt)
        print(f"  acc {mt['accuracy']:.4f} | kappa {mt['kappa']:.4f} | "
              f"macro-F1 {mt['macro_f1']:.4f} | entropy {mt['mean_entropy_nats']:.3f} nats")
        print("  per-class F1 (UNSEEN subjects only):")
        for s in STAGES:
            c = mt["per_class"][s]
            print(f"     {s:<5} f1={c['f1']:.4f}  prec={c['precision']:.4f}  "
                  f"rec={c['recall']:.4f}  n={c['support']:,}  "
                  f"ratio={mt['prediction_ratio'][s]:.2f}x")

        per_subj = {}
        by_s = defaultdict(list)
        for r in test_recs:
            by_s[r[:5]].append(r)
        for s, rl in sorted(by_s.items()):
            ks = [per_rec[r]["kappa"] for r in rl]
            ns = [per_rec[r]["n_epochs"] for r in rl]
            es = [per_rec[r]["mean_entropy_nats"] for r in rl]
            w = np.array(ns) / sum(ns)
            per_subj[s] = {"cohort": s[:2], "n_epochs": int(sum(ns)),
                           "kappa": float(np.dot(w, ks)),
                           "mean_entropy_nats": float(np.dot(w, es))}

        report[name] = {
            "checkpoint": str((d / "student_best.pt").relative_to(REPO_ROOT)),
            "epoch": ck["epoch"], "alpha": ck["alpha"], "T": ck["T"],
            "class_weight_power": ck["class_weight_power"],
            "use_spectral": ck["use_spectral"], "n_parameters": npar,
            "compression_vs_teacher": round(649229 / npar, 3),
            "validation_logged_macro_f1": logged,
            "validation_recomputed": mv,
            "validation_reproduces": bool(agree),
            "test_heldout": mt,
            "per_recording_test": per_rec,
            "per_subject_test": per_subj,
        }

    outp = res / "eval_students.json"
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- comparison ------------------------------------------------------
    if len(args.students) == 2:
        a, b = args.students
        A, B = report[a]["test_heldout"], report[b]["test_heldout"]
        print(f"\n{'='*74}\nHELD-OUT TEST COMPARISON\n{'='*74}")
        print(f"  {'metric':<14}{a:>26}{b:>26}")
        for k in ("accuracy", "kappa", "macro_f1"):
            print(f"  {k:<14}{A[k]:>26.4f}{B[k]:>26.4f}")
        print(f"\n  distillation gain (distilled - baseline):")
        for k in ("accuracy", "kappa", "macro_f1"):
            print(f"     {k:<10} {A[k]-B[k]:+.4f}")

    print(f"\nWrote {outp.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
