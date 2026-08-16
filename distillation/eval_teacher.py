"""
Phase 2 (GATE) - evaluate the teacher and quantify how much of its reported
kappa = 0.6663 is leakage.

The teacher was trained with a RECORDING-level split
(train_test_split(range(197), test_size=0.2, random_state=42)), which put both
nights of 30 of its 35 validation subjects into training. Only 5 subjects
(10 recordings) were never seen. This script separates:

  teacher_train        recordings the teacher trained on         -> memorisation
  teacher_val_leaked   its val recordings whose subject IS in train -> leaked
  teacher_val_clean    its val recordings whose subject is NOT     -> honest (n=5)
  new_test             the Phase 1 subject-level test split        -> mixed

If train ~= val_leaked ~= 0.666 while val_clean is materially lower, that is the
memorisation signature and the reported number cannot be trusted.

Runs one forward pass per recording and caches per-recording predictions, so the
run is resumable and every grouping is computed by slicing afterwards.
Recordings are visited in decision-priority order.
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
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from teacher_model import (DEFAULT_TEACHER_CKPT, STAGE_TO_IDX, count_parameters,  # noqa: E402
                           load_teacher)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
WINDOW_SIZE = 256          # matches the teacher's training config (notebook cell 12)


def recording_id(p: str) -> str:
    leaf = str(p).replace("\\", "/").rsplit("/", 1)[-1]
    return leaf[:-3] if leaf.endswith(".pt") else leaf


def subject_of(rec: str) -> str:
    return rec[:5]


@torch.no_grad()
def predict_recording(model, t_path: Path, s_path: Path, stages: list[str]):
    """Non-overlapping 256-epoch windows, batch of 1, no padding needed."""
    t = torch.load(t_path, map_location="cpu").float()
    s = torch.load(s_path, map_location="cpu").float()
    y = np.array([STAGE_TO_IDX[x] for x in stages], dtype=np.int64)

    n = min(t.shape[0], s.shape[0], len(y))
    preds = np.empty(n, dtype=np.int64)

    for start in range(0, n, WINDOW_SIZE):
        end = min(start + WINDOW_SIZE, n)
        logits = model(t[start:end].unsqueeze(0), s[start:end].unsqueeze(0))
        preds[start:end] = logits.argmax(-1).squeeze(0).numpy()

    del t, s
    return preds, y[:n]


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {}
    labels = list(range(len(STAGES)))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n_epochs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      labels=labels, zero_division=0)),
        "per_class": {
            STAGES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                        "f1": float(f1[i]), "support": int(sup[i])}
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / DEFAULT_TEACHER_CKPT))
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--cache", default=str(Path(__file__).parent / "results" / "teacher_preds"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "eval_teacher.json"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N recordings in priority order.")
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    recs = [recording_id(p) for p in df["tensor_path"]]
    row_of = {r: i for i, r in enumerate(recs)}
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))

    # --- reconstruct the teacher's own recording-level split -----------------
    tr_idx, va_idx = train_test_split(list(range(len(recs))), test_size=0.2,
                                      random_state=42)
    teacher_train = {recs[i] for i in tr_idx}
    teacher_val = {recs[i] for i in va_idx}
    tr_subj = {subject_of(r) for r in teacher_train}
    clean_subjects = sorted({subject_of(r) for r in teacher_val} - tr_subj)
    val_clean = sorted(r for r in teacher_val if subject_of(r) in clean_subjects)
    val_leaked = sorted(r for r in teacher_val if subject_of(r) not in clean_subjects)
    new_test = sorted(r for s in splits["splits"]["test"]
                      for r in splits["recordings_by_subject"][s])

    # --- priority order: decision-critical groups first -----------------------
    order, seen = [], set()
    for group in (val_clean, sorted(teacher_train)[:30], val_leaked, new_test, recs):
        for r in group:
            if r not in seen:
                seen.add(r)
                order.append(r)
    if args.limit:
        order = order[:args.limit]

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Teacher : {Path(args.checkpoint).name}")
    model = load_teacher(args.checkpoint, device="cpu", strict=True)
    print(f"          {count_parameters(model):,} trainable parameters")
    print(f"Groups  : clean={len(val_clean)} leaked_val={len(val_leaked)} "
          f"train={len(teacher_train)} new_test={len(new_test)}")
    print(f"Clean subjects (never seen): {clean_subjects}")
    print(f"Evaluating {len(order)} recordings in priority order\n")

    torch.set_grad_enabled(False)
    t_start = time.time()
    for i, rec in enumerate(order):
        npz = cache_dir / f"{rec}.npz"
        if npz.exists():
            continue
        row = df.iloc[row_of[rec]]
        preds, y = predict_recording(
            model, REPO_ROOT / row["tensor_path"], REPO_ROOT / row["spectral"],
            str(row["stage_sequence"]).split())
        np.savez_compressed(npz, preds=preds, labels=y)
        el = time.time() - t_start
        print(f"  [{i+1:>3}/{len(order)}] {rec}  n={len(y):>5}  "
              f"kappa={cohen_kappa_score(y, preds, labels=list(range(5))):.4f}  "
              f"({el/60:.1f} min elapsed)", flush=True)

    # --- assemble -------------------------------------------------------------
    def load(rec):
        f = cache_dir / f"{rec}.npz"
        if not f.exists():
            return None
        d = np.load(f)
        return d["labels"], d["preds"]

    def group_metrics(rec_list):
        ys, ps = [], []
        for r in rec_list:
            got = load(r)
            if got is None:
                continue
            ys.append(got[0])
            ps.append(got[1])
        if not ys:
            return {}
        return metrics_for(np.concatenate(ys), np.concatenate(ps))

    groups = {
        "teacher_train": sorted(teacher_train),
        "teacher_val_leaked": val_leaked,
        "teacher_val_clean": val_clean,
        "new_test_split": new_test,
    }
    results = {name: group_metrics(rl) for name, rl in groups.items()}

    # per-subject for the clean cohort - the spread matters at n=5
    per_subject = {}
    by_subj = defaultdict(list)
    for r in val_clean:
        by_subj[subject_of(r)].append(r)
    for s, rl in sorted(by_subj.items()):
        m = group_metrics(rl)
        if m:
            per_subject[s] = {"recordings": rl, "n_epochs": m["n_epochs"],
                              "kappa": m["kappa"], "accuracy": m["accuracy"],
                              "macro_f1": m["macro_f1"],
                              "per_class_f1": {k: v["f1"] for k, v in m["per_class"].items()}}

    # cohort breakdown - no ST subject is clean, so this is load-bearing
    cohort_clean = sorted({s[:2] for s in clean_subjects})

    payload = {
        "checkpoint": str(Path(args.checkpoint).relative_to(REPO_ROOT)),
        "commit": "b00a428",
        "trainable_parameters": count_parameters(model),
        "window_size": WINDOW_SIZE,
        "reported_validation_kappa": 0.6663,
        "teacher_split": "train_test_split(range(197), test_size=0.2, random_state=42) [RECORDING-level]",
        "clean_subjects": clean_subjects,
        "clean_cohorts_represented": cohort_clean,
        "st_subjects_clean": 0,
        "groups": results,
        "per_subject_clean": per_subject,
        "n_evaluated": len(list(cache_dir.glob("*.npz"))),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- print ---------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"{'group':<22}{'n_epochs':>10}{'acc':>9}{'kappa':>9}{'macroF1':>10}")
    print("-" * 74)
    for name in ("teacher_train", "teacher_val_leaked", "teacher_val_clean", "new_test_split"):
        m = results.get(name)
        if m:
            print(f"{name:<22}{m['n_epochs']:>10,}{m['accuracy']:>9.4f}"
                  f"{m['kappa']:>9.4f}{m['macro_f1']:>10.4f}")
    print("-" * 74)
    print(f"{'reported (val, leaky)':<22}{'-':>10}{0.7575:>9.4f}{0.6663:>9.4f}{0.6814:>10.4f}")
    print("=" * 74)

    if per_subject:
        print("\nPER-SUBJECT, teacher-clean cohort (never seen in training)")
        print(f"  {'subject':<10}{'cohort':>8}{'epochs':>9}{'kappa':>9}{'acc':>9}{'macroF1':>10}")
        ks = []
        for s, m in per_subject.items():
            ks.append(m["kappa"])
            print(f"  {s:<10}{s[:2]:>8}{m['n_epochs']:>9,}{m['kappa']:>9.4f}"
                  f"{m['accuracy']:>9.4f}{m['macro_f1']:>10.4f}")
        ks = np.array(ks)
        print(f"\n  mean={ks.mean():.4f}  sd={ks.std(ddof=1):.4f}  "
              f"min={ks.min():.4f}  max={ks.max():.4f}")
        print(f"  all 5 below reported 0.6663? {bool((ks < 0.6663).all())}")
        print(f"  straddle 0.6663?             {bool((ks < 0.6663).any() and (ks >= 0.6663).any())}")

    print(f"\n  clean cohorts represented: {cohort_clean}  "
          f"-> ZERO ST subjects are teacher-clean.")
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
