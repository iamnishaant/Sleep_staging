"""
Phase 5/6 - train the compact student by knowledge distillation.

Kaggle-ready. Reads CACHED teacher logits rather than running the teacher, so
the teacher's cost is paid once (see cache_teacher_logits.py) instead of once
per epoch.

THE PHASE 6 BASELINE IS THIS SAME SCRIPT WITH --alpha 1.0
---------------------------------------------------------
alpha=1.0 zeroes the soft term, leaving pure hard-label CE. Architecture, data,
splits, schedule, seed, augmentation and evaluation are then byte-identical to
the distilled run, so the ONLY difference is the training signal. A separate
baseline script could not guarantee that.

  Run A (no rebalancing, honest per-class picture):
      --alpha 0.5 --T 3.0
  Run B (with class weighting):
      --alpha 0.5 --T 3.0 --class-weight-power 0.5
  Baseline (same trainer, distillation disabled):
      --alpha 1.0

EVALUATION CONVENTION
---------------------
Validation is the selection split. The test split is never loaded here - it is
touched once, afterwards, by evaluate.py. Per-class F1 and prediction ratios
are logged every epoch; prediction ratio is the metric that exposed the N1
spray (2.61x) in the teacher and is worth watching in the student too.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from kd_loss import IGNORE_INDEX, distillation_loss  # noqa: E402
from student_model import StudentSleepStagingModel, count_parameters  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}
NUM_CLASSES = 5
LAB = list(range(NUM_CLASSES))


class KDSleepDataset(Dataset):
    """
    Sliding windows over recordings. Returns raw EEG, hard labels, and the
    teacher's cached logits for exactly the same epochs.

    teacher_dir=None yields zero logits, for the alpha=1.0 baseline where the
    soft term is unused - avoids requiring a cache to run the baseline.
    """

    def __init__(self, df, data_root, teacher_dir=None, window_size=256, overlap=0):
        self.df = df.reset_index(drop=True)
        self.root = Path(data_root)
        self.teacher_dir = Path(teacher_dir) if teacher_dir else None
        self.window_size = window_size
        self.stride = max(1, window_size - overlap)
        self.index = []
        for i, row in self.df.iterrows():
            T = len(str(row["stage_sequence"]).split())
            for s in range(0, T, self.stride):
                self.index.append((i, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, start = self.index[k]
        row = self.df.iloc[i]
        end = start + self.window_size
        stages = str(row["stage_sequence"]).split()[start:end]
        y = torch.tensor([STAGE_TO_IDX[s] for s in stages], dtype=torch.long)
        xt = torch.load(self.root / row["tensor_path"], map_location="cpu")[start:end]
        n = min(len(y), xt.shape[0])

        if self.teacher_dir is not None:
            d = np.load(self.teacher_dir / f"{row['rec']}.npz")
            tl = torch.from_numpy(d["logits"][start:end].astype(np.float32))
            n = min(n, tl.shape[0])
            tl = tl[:n]
        else:
            tl = torch.zeros(n, NUM_CLASSES)

        return xt[:n].float(), tl, y[:n]


def kd_collate_fn(batch):
    L, B = max(b[0].shape[0] for b in batch), len(batch)
    xt = torch.zeros(B, L, batch[0][0].shape[1])
    tl = torch.zeros(B, L, NUM_CLASSES)
    y = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    for i, (a, t, c) in enumerate(batch):
        T = a.shape[0]
        xt[i, :T], tl[i, :T], y[i, :T], mask[i, :T] = a, t, c, False
    return xt, tl, y, mask


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    yt, yp = [], []
    for xt, _, y, mask in loader:
        xt, mask = xt.to(device), mask.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(xt, None, mask)
        valid = ~mask
        yt.append(y.to(device)[valid].cpu().numpy())
        yp.append(logits.argmax(-1)[valid].cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    _, _, f1, _ = precision_recall_fscore_support(yt, yp, labels=LAB, zero_division=0)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa": float(cohen_kappa_score(yt, yp, labels=LAB)),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=LAB, zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", labels=LAB, zero_division=0)),
        "per_class_f1": {STAGES[i]: float(f1[i]) for i in LAB},
        "prediction_ratio": {STAGES[i]: float((yp == i).sum() / max((yt == i).sum(), 1))
                             for i in LAB},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = Path(__file__).parent / "results"
    ap.add_argument("--run-name", default=None,
                    help="Output subdirectory. Defaults to a name built from T/alpha.")
    ap.add_argument("--data-root", default=str(REPO_ROOT))
    ap.add_argument("--index", default=str(REPO_ROOT / "processed_sleepedf" / "index.csv"))
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--teacher", default="E0",
                    help="Which cached teacher to distil from (cache dir suffix).")
    # --- distillation ---
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="1.0 = hard labels only (the Phase 6 baseline).")
    ap.add_argument("--class-weight-power", type=float, default=0.0,
                    help="0 = no class weighting (Run A). 0.5 = sqrt-inverse (Run B).")
    # --- student ---
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n-tokens", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    # --- optimisation ---
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--window-size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=192)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop-patience", type=int, default=25)
    ap.add_argument("--max-recordings", type=int, default=None,
                    help="Testing only. Caps recordings per split.")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    run = args.run_name or (f"student_alpha{args.alpha:g}_T{args.T:g}"
                            f"_{args.teacher}"
                            + (f"_cw{args.class_weight_power:g}"
                               if args.class_weight_power else ""))
    out = res / "students" / run
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        sm, sup = f"sm_{cap[0]}{cap[1]}", torch.cuda.get_arch_list()
        if sup and sm not in sup:
            raise SystemExit(f"UNSUPPORTED GPU {torch.cuda.get_device_name(0)} "
                             f"({sm}); build supports {sup}. Switch accelerator.")

    root = Path(args.data_root)
    df = pd.read_csv(args.index)
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    df["tensor_path"] = ["processed_sleepedf/tensors/%s.pt" % r for r in df["rec"]]

    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    tr = df[df["subject"].isin(sp["splits"]["train"])]
    va = df[df["subject"].isin(sp["splits"]["val"])]
    assert set(tr["subject"]).isdisjoint(va["subject"])
    assert set(tr["subject"]).isdisjoint(sp["splits"]["test"])
    assert set(va["subject"]).isdisjoint(sp["splits"]["test"])
    if args.max_recordings:
        print(f"*** --max-recordings={args.max_recordings}: TEST RUN ***")
        tr = tr.head(args.max_recordings)
        va = va.head(max(1, args.max_recordings // 2))

    distilling = args.alpha < 1.0
    tdir = res / f"teacher_logits_{args.teacher}"
    if distilling:
        if not tdir.exists():
            raise SystemExit(f"No cached teacher logits at {tdir}.\n"
                             f"Run: python distillation/cache_teacher_logits.py "
                             f"--teacher {args.teacher}")
        have = {p.stem for p in tdir.glob("*.npz")}
        miss = sorted(set(tr["rec"]) - have)
        if miss:
            raise SystemExit(f"{len(miss)} train recordings have no cached logits, "
                             f"first: {miss[:3]}")

    print(f"\n{'='*72}\nSTUDENT RUN: {run}\n{'='*72}")
    print(f"  device {device} | T={args.T} alpha={args.alpha} "
          f"({'DISTILLING from ' + args.teacher if distilling else 'HARD LABELS ONLY (baseline)'})")
    print(f"  train {len(tr)} rec / {tr['subject'].nunique()} subj | "
          f"val {len(va)} rec / {va['subject'].nunique()} subj | "
          f"test {len(sp['splits']['test'])} subj HELD OUT (not loaded)")

    ds_tr = KDSleepDataset(tr, root, tdir if distilling else None,
                           args.window_size, args.overlap)
    ds_va = KDSleepDataset(va, root, None, args.window_size, 0)
    kw = dict(collate_fn=kd_collate_fn, num_workers=args.num_workers,
              pin_memory=use_amp)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, **kw)
    dl_va = DataLoader(ds_va, batch_size=args.micro_batch, shuffle=False, **kw)
    opt_steps = len(dl_tr)
    print(f"  train windows {len(ds_tr):,} ({opt_steps} steps/epoch) | "
          f"val windows {len(ds_va):,}")

    cw = None
    if args.class_weight_power > 0:
        c = dict.fromkeys(STAGES, 0)
        for s in tr["stage_sequence"]:
            for x in str(s).split():
                c[x] += 1
        counts = torch.tensor([float(c[s]) for s in STAGES])
        cw = (1.0 / counts) ** args.class_weight_power
        cw = (cw / cw.sum() * NUM_CLASSES).to(device)
        print("  class weights: " + ", ".join(f"{s}={w:.3f}"
                                              for s, w in zip(STAGES, cw.tolist())))
    else:
        print("  class weights: none (Run A - honest per-class picture)")

    model = StudentSleepStagingModel(
        embed_dim=args.embed_dim, heads=args.heads, layers=args.layers,
        dropout=args.dropout, n_tokens=args.n_tokens).to(device)
    n_par = count_parameters(model)
    print(f"  student parameters: {n_par:,}  "
          f"(teacher 649,229 -> {649229/n_par:.2f}x compression)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    warm = max(1, int(args.warmup_frac * args.epochs * opt_steps))
    total = args.epochs * opt_steps

    def lr_lambda(step):                       # linear warmup -> cosine decay
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch, best, since_best = 0, -1.0, 0
    last = out / "student_last.pt"
    if not args.no_resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        shape = {"epochs": args.epochs, "steps_per_epoch": opt_steps}
        if ck.get("schedule_shape") != shape:
            raise SystemExit(f"Cannot resume: schedule changed.\n  was "
                             f"{ck.get('schedule_shape')}\n  now {shape}\n"
                             f"Delete {last} to start fresh.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best_macro_f1"]
        since_best = ck.get("since_best", 0)
        print(f"  RESUMED at epoch {start_epoch} (best macro-F1 {best:.4f})")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tot, tot_hard, tot_soft = 0.0, 0.0, 0.0
        for xt, tl, y, mask in dl_tr:
            xt, tl = xt.to(device, non_blocking=True), tl.to(device, non_blocking=True)
            y, mask = y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            n_valid = (y != IGNORE_INDEX).sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            bl = bh = bs = 0.0
            for c0 in range(0, xt.size(0), args.micro_batch):
                sl = slice(c0, c0 + args.micro_batch)
                nv = (y[sl] != IGNORE_INDEX).sum()
                if nv == 0:
                    continue
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(xt[sl], None, mask[sl])
                    loss, parts = distillation_loss(
                        logits, tl[sl], y[sl], T=args.T, alpha=args.alpha,
                        class_weights=cw, return_parts=True)
                # Weight each chunk by its share of the batch's valid tokens so
                # chunking is equivalent to one full-batch loss.
                w = (nv / n_valid).float()
                scaler.scale(loss * w).backward()
                bl += loss.item() * w.item()
                bh += parts["hard"].item() * w.item()
                bs += parts["soft"].item() * w.item()
            tot += bl; tot_hard += bh; tot_soft += bs
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()

        m = evaluate(model, dl_va, device, use_amp)
        rec = {"run": run, "epoch": epoch, "T": args.T, "alpha": args.alpha,
               "loss": tot / opt_steps, "loss_hard": tot_hard / opt_steps,
               "loss_soft": tot_soft / opt_steps,
               "lr": optimizer.param_groups[0]["lr"],
               "elapsed_min": round((time.time() - t0) / 60, 2), **m}
        with open(out / "training_metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

        is_best = m["macro_f1"] > best
        if is_best:
            best = m["macro_f1"]; since_best = 0
        else:
            since_best += 1

        state = {"run": run, "epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "best_macro_f1": best, "since_best": since_best,
                 "args": vars(args), "n_parameters": n_par,
                 "class_weights": cw.tolist() if cw is not None else None,
                 "schedule_shape": {"epochs": args.epochs, "steps_per_epoch": opt_steps}}
        torch.save(state, last)
        if is_best:
            torch.save(state, out / "student_best.pt")

        pc = " ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())
        print(f"ep {epoch:3d} | loss {rec['loss']:.4f} "
              f"(hard {rec['loss_hard']:.3f} soft {rec['loss_soft']:.3f}) | "
              f"mF1 {m['macro_f1']:.4f} | k {m['kappa']:.4f} | "
              f"acc {m['accuracy']:.4f} | {pc} | "
              f"N1ratio {m['prediction_ratio']['N1']:.2f}x"
              f"{'  <- best' if is_best else ''}", flush=True)

        if args.early_stop_patience and since_best >= args.early_stop_patience:
            print(f"\nEarly stop: no val macro-F1 gain for "
                  f"{args.early_stop_patience} epochs.")
            break

    print(f"\nDONE {run} in {(time.time()-t0)/60:.1f} min | "
          f"best val macro-F1 {best:.4f}")
    print(f"Artefacts in {out}. Test split was NEVER loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
