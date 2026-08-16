"""
Final evaluation: inherited teacher vs retrained teacher.

Produces, per model:
  - held-out TEST split metrics (the first honest number for this project)
  - per-recording kappa AND mean prediction entropy (night-level confidence)
  - per-subject aggregation
  - per-class F1 on UNSEEN subjects only (project convention)
  - exposure-gradient groups, so before/after flattening can be computed

Exposure groups differ by model, because their training sets differ:
  inherited : train / val_leaked / val_clean   (recording-level split, leaky)
  retrained : train / val / test               (subject-level split, clean)

Predictions are cached per (model, recording), so runs are resumable and
groups are computed by slicing afterwards.
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from teacher_model import FusedSleepStagingModel, STAGE_TO_IDX  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
WINDOW_SIZE = 256


def recording_id(p: str) -> str:
    leaf = str(p).replace("\\", "/").rsplit("/", 1)[-1]
    return leaf[:-3] if leaf.endswith(".pt") else leaf


def subject_of(rec: str) -> str:
    return rec[:5]


def load_any(checkpoint: Path) -> FusedSleepStagingModel:
    """Load either the inherited bare state_dict (module. prefix) or the
    retrained checkpoint dict, into the same class. Strict."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    model = FusedSleepStagingModel()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{checkpoint.name} mismatch:\n  missing {sorted(missing)}"
                           f"\n  unexpected {sorted(unexpected)}")
    model.eval()
    return model


@torch.no_grad()
def predict_recording(model, t_path: Path, s_path: Path, stages: list[str]):
    """Non-overlapping 256-epoch windows. Returns preds, labels, per-epoch entropy."""
    t = torch.load(t_path, map_location="cpu").float()
    s = torch.load(s_path, map_location="cpu").float()
    y = np.array([STAGE_TO_IDX[x] for x in stages], dtype=np.int64)
    n = min(t.shape[0], s.shape[0], len(y))

    preds = np.empty(n, dtype=np.int64)
    ent = np.empty(n, dtype=np.float32)
    for start in range(0, n, WINDOW_SIZE):
        end = min(start + WINDOW_SIZE, n)
        logits = model(t[start:end].unsqueeze(0), s[start:end].unsqueeze(0))
        p = F.softmax(logits.float(), dim=-1).squeeze(0)
        preds[start:end] = p.argmax(-1).numpy()
        # Shannon entropy in nats; ln(5)=1.609 is maximal (uniform) uncertainty.
        ent[start:end] = (-(p * torch.log(p.clamp_min(1e-12))).sum(-1)).numpy()
    del t, s
    return preds, y[:n], ent


def metrics_for(y_true, y_pred, ent=None) -> dict:
    if len(y_true) == 0:
        return {}
    lab = list(range(len(STAGES)))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=lab,
                                                    zero_division=0)
    out = {
        "n_epochs": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=lab)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=lab,
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=lab,
                                      zero_division=0)),
        "per_class": {STAGES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f1[i]), "support": int(sup[i])}
                      for i in lab},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=lab).tolist(),
    }
    if ent is not None and len(ent):
        out["mean_entropy_nats"] = float(np.mean(ent))
        out["max_possible_entropy_nats"] = float(np.log(len(STAGES)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = Path(__file__).parent / "results"
    ap.add_argument("--model", required=True,
                    choices=["inherited", "retrained", "E1a", "E1b", "E4"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--train-sample", type=int, default=30,
                    help="Recordings sampled from the model's train set for the "
                         "exposure gradient (full set is unnecessary and slow).")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif args.model == "retrained":
        ckpt = res / "retrained" / "teacher_best.pt"
    elif args.model in ("E1a", "E1b", "E4"):
        ckpt = res / args.model / "teacher_best.pt"
    else:
        ckpt = REPO_ROOT / ("results/sleep staging/temporal spectral fusion/"
                            "concat/best_model_fusion.pt")

    df = pd.read_csv(args.index)
    recs = [recording_id(p) for p in df["tensor_path"]]
    row_of = {r: i for i, r in enumerate(recs)}
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    rbs = splits["recordings_by_subject"]

    def recs_of(split):
        return sorted(r for s in splits["splits"][split] for r in rbs[s])

    test_recs, val_recs, train_recs = recs_of("test"), recs_of("val"), recs_of("train")

    if args.model != "inherited":
        groups = {"test_heldout": test_recs, "val": val_recs,
                  "train_sample": train_recs[:args.train_sample]}
        exposure = [("train_sample", "same recording (trained on)"),
                    ("val", "unseen subject (selection split)"),
                    ("test_heldout", "unseen subject (held out)")]
    else:
        t_idx, v_idx = train_test_split(list(range(len(recs))), test_size=0.2,
                                        random_state=42)
        t_train = {recs[i] for i in t_idx}
        t_val = {recs[i] for i in v_idx}
        tr_subj = {subject_of(r) for r in t_train}
        clean = sorted({subject_of(r) for r in t_val} - tr_subj)
        groups = {
            "test_heldout": test_recs,
            "train_sample": sorted(t_train)[:args.train_sample],
            "val_leaked": sorted(r for r in t_val if subject_of(r) not in clean),
            "val_clean": sorted(r for r in t_val if subject_of(r) in clean),
        }
        exposure = [("train_sample", "same recording (trained on)"),
                    ("val_leaked", "same subject, different night"),
                    ("val_clean", "unseen subject")]

    # Held-out test first - it is the decision-critical number, so a truncated
    # or interrupted run still yields it.
    order, seen = [], set()
    for name in ["test_heldout"] + [e[0] for e in exposure]:
        for r in groups.get(name, []):
            if r not in seen:
                seen.add(r); order.append(r)
    if args.limit:
        order = order[:args.limit]

    cache = res / f"preds_{args.model}"
    cache.mkdir(parents=True, exist_ok=True)

    print(f"MODEL     : {args.model}")
    print(f"CHECKPOINT: {ckpt.relative_to(REPO_ROOT)}")
    model = load_any(ckpt)
    print(f"            {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params, loaded strict")
    print(f"GROUPS    : " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    print(f"Evaluating {len(order)} recordings\n")

    torch.set_grad_enabled(False)
    t0 = time.time()
    for i, rec in enumerate(order):
        f = cache / f"{rec}.npz"
        if f.exists():
            continue
        row = df.iloc[row_of[rec]]
        pr, y, en = predict_recording(model, REPO_ROOT / row["tensor_path"],
                                      REPO_ROOT / row["spectral"],
                                      str(row["stage_sequence"]).split())
        np.savez_compressed(f, preds=pr, labels=y, entropy=en)
        print(f"  [{i+1:>3}/{len(order)}] {rec} n={len(y):>5} "
              f"kappa={cohen_kappa_score(y, pr, labels=list(range(5))):.4f} "
              f"H={en.mean():.3f} ({(time.time()-t0)/60:.1f} min)", flush=True)

    def load(rec):
        f = cache / f"{rec}.npz"
        if not f.exists():
            return None
        d = np.load(f)
        return d["labels"], d["preds"], d["entropy"]

    def group_metrics(rl):
        ys, ps, es = [], [], []
        for r in rl:
            g = load(r)
            if g:
                ys.append(g[0]); ps.append(g[1]); es.append(g[2])
        if not ys:
            return {}
        return metrics_for(np.concatenate(ys), np.concatenate(ps), np.concatenate(es))

    results = {k: group_metrics(v) for k, v in groups.items()}

    # per-recording, on the held-out test split
    per_rec = {}
    for r in groups["test_heldout"]:
        g = load(r)
        if not g:
            continue
        y, p, e = g
        per_rec[r] = {
            "subject": subject_of(r), "n_epochs": int(len(y)),
            "kappa": float(cohen_kappa_score(y, p, labels=list(range(5)))),
            "accuracy": float(accuracy_score(y, p)),
            "mean_entropy_nats": float(e.mean()),
        }

    per_subj = {}
    by_s = defaultdict(list)
    for r in groups["test_heldout"]:
        by_s[subject_of(r)].append(r)
    for s, rl in sorted(by_s.items()):
        m = group_metrics(rl)
        if m:
            per_subj[s] = {"recordings": rl, "n_epochs": m["n_epochs"],
                           "kappa": m["kappa"], "accuracy": m["accuracy"],
                           "macro_f1": m["macro_f1"],
                           "mean_entropy_nats": m.get("mean_entropy_nats"),
                           "cohort": s[:2]}

    payload = {
        "model": args.model,
        "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
        "commit": "b00a428",
        "window_size": WINDOW_SIZE,
        "groups": results,
        "exposure_order": [e[0] for e in exposure],
        "exposure_labels": dict(exposure),
        "per_recording_test": per_rec,
        "per_subject_test": per_subj,
        "convention": "per-class F1 headline is groups.test_heldout (unseen subjects only)",
    }
    outp = res / f"eval_final_{args.model}.json"
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{args.model.upper()}  |  {'group':<28}{'n_ep':>8}{'acc':>8}{'kappa':>8}{'mF1':>8}{'H':>7}")
    print("-" * 78)
    for name in [e[0] for e in exposure] + (["test_heldout"] if args.model == "inherited" else []):
        m = results.get(name)
        if m:
            print(f"{'':<12}{name:<28}{m['n_epochs']:>8,}{m['accuracy']:>8.4f}"
                  f"{m['kappa']:>8.4f}{m['macro_f1']:>8.4f}"
                  f"{m.get('mean_entropy_nats', 0):>7.3f}")
    print("=" * 78)

    m = results.get("test_heldout")
    if not m:
        print("\n(no test_heldout predictions cached yet - rerun without --limit)")
        print(f"Wrote {outp.relative_to(REPO_ROOT)}")
        return 0
    print(f"\nHELD-OUT TEST ({m['n_epochs']:,} epochs, {len(per_subj)} subjects)")
    print(f"  accuracy {m['accuracy']:.4f} | kappa {m['kappa']:.4f} | "
          f"macro-F1 {m['macro_f1']:.4f} | mean entropy {m.get('mean_entropy_nats',0):.3f} nats")
    print("  per-class F1 (UNSEEN subjects only):")
    for s in STAGES:
        c = m["per_class"][s]
        print(f"     {s:<5} f1={c['f1']:.4f}  prec={c['precision']:.4f}  "
              f"rec={c['recall']:.4f}  n={c['support']:,}")

    print(f"\n  per-subject (test):")
    print(f"     {'subj':<8}{'coh':>5}{'epochs':>8}{'kappa':>8}{'acc':>8}{'H':>7}")
    for s, v in sorted(per_subj.items(), key=lambda kv: -kv[1]["kappa"]):
        print(f"     {s:<8}{v['cohort']:>5}{v['n_epochs']:>8,}{v['kappa']:>8.4f}"
              f"{v['accuracy']:>8.4f}{v['mean_entropy_nats']:>7.3f}")
    ks = np.array([v["kappa"] for v in per_subj.values()])
    print(f"     mean={ks.mean():.4f} sd={ks.std(ddof=1):.4f} "
          f"min={ks.min():.4f} max={ks.max():.4f}")
    sc = [v["kappa"] for v in per_subj.values() if v["cohort"] == "SC"]
    st = [v["kappa"] for v in per_subj.values() if v["cohort"] == "ST"]
    if sc and st:
        print(f"     SC mean={np.mean(sc):.4f} (n={len(sc)})   "
              f"ST mean={np.mean(st):.4f} (n={len(st)})")

    print(f"\nWrote {outp.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
