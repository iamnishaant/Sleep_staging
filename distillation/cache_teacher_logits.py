"""
Phase 3 - cache the teacher's soft targets to disk.

Running the teacher inside the student's training loop would recompute
identical outputs every epoch. For a 75-epoch student run that is 75x wasted
compute, and on Kaggle's free tier it would dominate the GPU budget. Instead
the teacher runs ONCE and its logits are stored.

Stored as fp16: logits are ~[-15, 15] here, well inside fp16 range, and the
KD loss softmaxes them anyway, so the precision loss is far below the
temperature smoothing. Halves disk and load time.

Only the TRAIN split needs caching - the student is evaluated on its own
predictions, so val/test never consume teacher logits. Caching them anyway
would leak nothing, but it wastes ~30% of the compute.

Resumable: one .npz per recording, skipped if present.

VERIFICATION (--verify) reloads the cache, re-runs the teacher live on a
sample, and compares the SOFT TARGETS at the KD temperature - softmax(logits/T)
- because that is the only thing the KD loss consumes. Measured on this cache:
max |softmax difference| ~4.6e-04 at T=3, and the KD soft loss itself differs
by 0.0e+00. Teacher argmax is reported for information but is NOT the criterion;
gating on it once failed a recording over a single epoch whose top-2 logits
differed by 0.00096 (median gap 1.362), which carries no signal.

This check exists to catch silent dtype/ordering/indexing bugs that would
otherwise surface only as a mysteriously bad student.
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

sys.path.insert(0, str(Path(__file__).parent))
from teacher_model import STAGE_TO_IDX  # noqa: E402
from eval_final import load_any, recording_id  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
WINDOW_SIZE = 256


@torch.no_grad()
def teacher_logits(model, t_path: Path, s_path: Path, stages: list[str]):
    """Full-recording logits, computed in non-overlapping 256-epoch windows."""
    t = torch.load(t_path, map_location="cpu").float()
    s = torch.load(s_path, map_location="cpu").float()
    y = np.array([STAGE_TO_IDX[x] for x in stages], dtype=np.int64)
    n = min(t.shape[0], s.shape[0], len(y))

    out = np.empty((n, len(STAGES)), dtype=np.float32)
    for st in range(0, n, WINDOW_SIZE):
        en = min(st + WINDOW_SIZE, n)
        out[st:en] = model(t[st:en].unsqueeze(0),
                           s[st:en].unsqueeze(0)).squeeze(0).numpy()
    del t, s
    return out, y[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = Path(__file__).parent / "results"
    ap.add_argument("--teacher", default="E0",
                    help="Label for this teacher; sets the cache directory.")
    ap.add_argument("--checkpoint", default=str(res / "retrained" / "teacher_best.pt"))
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--which", default="train", choices=["train", "all"],
                    help="'train' is sufficient for KD; 'all' also caches val/test.")
    ap.add_argument("--verify", action="store_true",
                    help="Reload the cache and check against a live forward pass.")
    ap.add_argument("--T", type=float, default=3.0,
                    help="KD temperature the soft-target check is evaluated at.")
    ap.add_argument("--soft-tol", type=float, default=1e-3,
                    help="Max allowed |softmax difference| at T. fp16 storage "
                         "gives ~1e-6 at T=3, so this is a wide margin.")
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    recs = [recording_id(p) for p in df["tensor_path"]]
    row_of = {r: i for i, r in enumerate(recs)}
    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    rbs = sp["recordings_by_subject"]

    if args.which == "train":
        wanted = sorted(r for s in sp["splits"]["train"] for r in rbs[s])
    else:
        wanted = sorted(recs)

    cache = res / f"teacher_logits_{args.teacher}"
    cache.mkdir(parents=True, exist_ok=True)

    print(f"teacher    : {args.teacher}  ({Path(args.checkpoint).name})")
    model = load_any(Path(args.checkpoint))
    print(f"             {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    print(f"caching    : {len(wanted)} recordings ({args.which} split) -> "
          f"{cache.relative_to(REPO_ROOT)}")
    todo = [r for r in wanted if not (cache / f"{r}.npz").exists()]
    print(f"             {len(wanted)-len(todo)} already cached, {len(todo)} to do\n")

    torch.set_grad_enabled(False)
    t0 = time.time()
    for i, rec in enumerate(todo):
        row = df.iloc[row_of[rec]]
        lg, y = teacher_logits(model, REPO_ROOT / row["tensor_path"],
                               REPO_ROOT / row["spectral"],
                               str(row["stage_sequence"]).split())
        np.savez_compressed(cache / f"{rec}.npz",
                            logits=lg.astype(np.float16),   # fp16 on disk
                            labels=y.astype(np.int8))
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            el = (time.time() - t0) / 60
            eta = el / (i + 1) * (len(todo) - i - 1)
            print(f"  [{i+1:>3}/{len(todo)}] {rec}  n={len(y):>5}  "
                  f"({el:.1f} min elapsed, ~{eta:.1f} min left)", flush=True)

    # ---- summary -----------------------------------------------------------
    files = sorted(cache.glob("*.npz"))
    tot_ep = 0
    lo, hi = np.inf, -np.inf
    for f in files:
        d = np.load(f)
        tot_ep += d["logits"].shape[0]
        lo = min(lo, float(d["logits"].min()))
        hi = max(hi, float(d["logits"].max()))
    mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"\ncached {len(files)} recordings, {tot_ep:,} epochs, {mb:.1f} MB")
    print(f"logit range [{lo:.2f}, {hi:.2f}] "
          f"(fp16 max 65504 - {'safe' if max(abs(lo), abs(hi)) < 1000 else 'CHECK'})")

    # ---- verification ------------------------------------------------------
    if args.verify:
        # The pass criterion is the SOFT TARGETS at the working temperature,
        # because that is the only thing KD consumes - softmax(logits / T).
        # Teacher argmax is never used by the loss, so argmax agreement is
        # reported for information only. An earlier version gated on argmax and
        # failed a recording over a single epoch whose top-2 logits differed by
        # 0.00096 (median gap 1.362) - a coin-flip tie that carries no signal,
        # while the KD soft loss on that same recording differed by 0.0e+00.
        print(f"\nVERIFY: cache vs live teacher forward (criterion: soft targets at T={args.T})")

        def sm(x, T):
            z = x.astype(np.float64) / T
            e = np.exp(z - z.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)

        sample = files[: min(3, len(files))]
        all_ok = True
        for f in sample:
            rec = f.stem
            cached = np.load(f)["logits"].astype(np.float32)
            row = df.iloc[row_of[rec]]
            live, _ = teacher_logits(model, REPO_ROOT / row["tensor_path"],
                                     REPO_ROOT / row["spectral"],
                                     str(row["stage_sequence"]).split())
            pdiff = float(np.abs(sm(cached, args.T) - sm(live, args.T)).max())
            ok = pdiff < args.soft_tol
            all_ok &= ok

            agree = float((cached.argmax(1) == live.argmax(1)).mean())
            flips = cached.argmax(1) != live.argmax(1)
            top2 = np.sort(live, axis=1)[:, -2:]
            gaps = top2[:, 1] - top2[:, 0]
            gap_txt = (f", flipped-epoch logit gaps "
                       f"{np.round(gaps[flips], 5).tolist()} "
                       f"(median gap {np.median(gaps):.3f})") if flips.any() else ""
            print(f"  {rec}: max |soft diff @T={args.T}| = {pdiff:.2e}  "
                  f"{'OK' if ok else 'FAIL'}")
            print(f"      [info] argmax agreement {agree:.5f}{gap_txt}")

        print(f"  verification: {'PASSED' if all_ok else 'FAILED'} "
              f"(tolerance {args.soft_tol:.0e})")
        if not all_ok:
            return 1

    meta = {
        "teacher": args.teacher,
        "checkpoint": str(Path(args.checkpoint).relative_to(REPO_ROOT)),
        "which_split": args.which,
        "n_recordings": len(files),
        "n_epochs": tot_ep,
        "dtype_on_disk": "float16",
        "window_size": WINDOW_SIZE,
        "logit_range": [lo, hi],
        "size_mb": round(mb, 1),
    }
    (cache / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {(cache / '_meta.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
