"""
Retrain the teacher under an honest protocol - FAITHFUL REPLICATION.

Kaggle-ready. Runs on GPU (2x T4 via DataParallel), AMP, resumable.

WHAT IS REPLICATED EXACTLY (inherited design - do not tune)
----------------------------------------------------------
Architecture   FusedSleepStagingModel, fusion='concat', embed_dim=128,
               heads=4, layers=3, dropout=0.2, spectral encoder = Linear(34,128),
               NO auxiliary heads      <- matches the concat checkpoint we evaluated
Loss           FocalLoss(gamma=2.0) with inverse-frequency per-class alpha
Optimiser      AdamW(lr=3e-4, weight_decay=1e-2)
Schedule       OneCycleLR(max_lr=3e-4, pct_start=0.05)
Batch          16, grad-clip max_norm=1.0, AMP + GradScaler
Windowing      window_size=256, overlap=0
Epochs         150

WHAT IS CHANGED (protocol only - approved)
------------------------------------------
1. Subject-level splits from splits.json (was: recording-level train_test_split,
   which leaked 30 of 35 val subjects). THIS IS THE POINT OF THE RERUN.
2. Checkpoint selection on macro-F1 (was: weighted F1, dominated by N2+W which
   are 67% of epochs and barely move on N1/N3/REM).
3. Test split is never loaded during training. Selection uses val only.
4. Focal alpha computed from the TRAIN split only (was: full-dataset class
   counts, which read test-set statistics). Numerically near-identical; the
   change is principled, not tuning. Use --alpha-source hardcoded to reproduce
   the original constants exactly.

KNOWN BUG IN THE INHERITED LOSS - PRESERVED DELIBERATELY
--------------------------------------------------------
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
Because `ce` is already alpha-weighted, exp(-ce) = p_t**alpha_c, not p_t. The
focal modulator is therefore (1 - p_t**alpha_c)**gamma rather than
(1 - p_t)**gamma. For classes with alpha > 1 (N1=1.36, N3=1.76) this INCREASES
the modulator (more focus); for alpha < 1 (W=0.49, N2=0.39) it decreases it.
Directionally it amplifies the intended rebalancing, so it is not the cause of
weak N1 - and correcting it would change results and break faithful
replication. Flagged, not fixed. Pass --fix-focal-pt to correct it in a
separate ablation run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------
# Model - identical to distillation/teacher_model.py. Inlined so this file is
# self-contained and can be pasted straight into a Kaggle notebook.
# --------------------------------------------------------------------------
import math

NUM_CLASSES = 5
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class AdaptiveAtrousPyramid(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7,
                      dilation=d, padding=3 * d) for d in dilations])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_channels * len(dilations), len(dilations), 1),
            nn.Softmax(dim=1))
        self.se = SEBlock(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

    def forward(self, x):
        feats = [F.relu(b(x)) for b in self.branches]
        weights = self.gate(torch.cat(feats, dim=1))
        out = 0
        for i, f in enumerate(feats):
            out = out + f * weights[:, i:i + 1]
        return self.proj(self.se(out))


class EpochEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.pyramid = AdaptiveAtrousPyramid()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        B, T, L = x.shape
        f = self.pyramid(x.reshape(B * T, 1, L))
        f = self.pool(f).squeeze(-1)
        return self.fc(f).view(B, T, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SleepTransformer(nn.Module):
    def __init__(self, embed_dim=128, heads=4, layers=3, dropout=0.2, max_len=512):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.cls = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x, padding_mask=None):
        x = self.positional_encoding(x)
        return self.cls(self.encoder(x, src_key_padding_mask=padding_mask))


class FusedSleepStagingModel(nn.Module):
    def __init__(self, spectral_input_dim=34, embed_dim=128, heads=4,
                 layers=3, dropout=0.2):
        super().__init__()
        self.temporal_encoder = EpochEncoder(embed_dim=embed_dim)
        self.spectral_encoder = nn.Linear(spectral_input_dim, embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.context = SleepTransformer(embed_dim, heads, layers, dropout)

    def forward(self, x_temporal, x_spectral, padding_mask=None):
        h = torch.cat([self.temporal_encoder(x_temporal),
                       self.spectral_encoder(x_spectral)], dim=-1)
        return self.context(self.fusion(h), padding_mask)


class FocalLoss(nn.Module):
    """Inherited focal loss. See module docstring re: the pt bug (preserved)."""

    def __init__(self, alpha=None, gamma=2.0, fix_pt=False):
        super().__init__()
        self.gamma = gamma
        self.fix_pt = fix_pt
        if alpha is None:
            self.register_buffer("alpha", torch.ones(NUM_CLASSES))
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float))

    def forward(self, logits, targets):
        valid = targets != -100
        logits, targets = logits[valid], targets[valid]
        if logits.numel() == 0:
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        if self.fix_pt:
            with torch.no_grad():
                pt = torch.softmax(logits, -1).gather(1, targets[:, None]).squeeze(1)
        else:
            pt = torch.exp(-ce)          # inherited behaviour: p_t ** alpha_c
        return ((1 - pt) ** self.gamma * ce).mean()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class FusionSleepDataset(Dataset):
    """Sliding-window fusion dataset over a given list of recordings."""

    def __init__(self, df, data_root: Path, window_size=256, overlap=0):
        self.df = df.reset_index(drop=True)
        self.root = Path(data_root)
        self.window_size = window_size
        self.stride = window_size - overlap
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
        xs = torch.load(self.root / row["spectral"], map_location="cpu")[start:end]
        n = min(len(y), xt.shape[0], xs.shape[0])
        return xt[:n].float(), xs[:n].float(), y[:n]


def fusion_collate_fn(batch):
    lengths = [b[0].shape[0] for b in batch]
    L, B = max(lengths), len(batch)
    xt = torch.zeros(B, L, batch[0][0].shape[1])
    xs = torch.zeros(B, L, batch[0][1].shape[1])
    y = torch.full((B, L), -100, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    for i, (a, b, c) in enumerate(batch):
        T = a.shape[0]
        xt[i, :T], xs[i, :T], y[i, :T], mask[i, :T] = a, b, c, False
    return xt, xs, y, mask


# --------------------------------------------------------------------------
# Evaluation - macro-F1 selection, per-class always reported
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    yt, yp = [], []
    for xt, xs, y, mask in loader:
        xt, xs, mask = xt.to(device), xs.to(device), mask.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(xt, xs, mask)
        valid = ~mask
        yt.append(y.to(device)[valid].cpu().numpy())
        yp.append(logits.argmax(-1)[valid].cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    labels = list(range(NUM_CLASSES))
    p, r, f1, sup = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa": float(cohen_kappa_score(yt, yp, labels=labels)),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", labels=labels, zero_division=0)),
        "per_class_f1": {STAGES[i]: float(f1[i]) for i in labels},
        "per_class_support": {STAGES[i]: int(sup[i]) for i in labels},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".",
                    help="Dir containing processed_sleepedf/ (Kaggle: dataset mount).")
    ap.add_argument("--index", default="processed_sleepedf/index.csv")
    ap.add_argument("--splits", default="distillation/splits.json")
    ap.add_argument("--out-dir", default="/kaggle/working")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--window-size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha-source", choices=["train", "hardcoded"], default="train")
    ap.add_argument("--fix-focal-pt", action="store_true",
                    help="Ablation only. Corrects the inherited pt bug.")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-recordings", type=int, default=None,
                    help="Testing only: cap recordings per split to exercise the "
                         "full loop cheaply. Never use for a real run.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    root = Path(args.data_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    df = pd.read_csv(root / args.index)
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    splits = json.loads((root / args.splits).read_text(encoding="utf-8"))
    rec_by_subj = splits["recordings_by_subject"]

    def recs_for(split):
        return {r for s in splits["splits"][split] for r in rec_by_subj[s]}

    train_recs, val_recs, test_recs = recs_for("train"), recs_for("val"), recs_for("test")
    assert not (train_recs & val_recs) and not (train_recs & test_recs) \
        and not (val_recs & test_recs), "split overlap"

    df_train = df[df["rec"].isin(train_recs)]
    df_val = df[df["rec"].isin(val_recs)]
    # test is deliberately NOT loaded here.

    if args.max_recordings:
        print(f"*** --max-recordings={args.max_recordings}: TEST RUN, not a real train ***")
        df_train = df_train.head(args.max_recordings)
        df_val = df_val.head(max(1, args.max_recordings // 2))

    print(f"device={device} | train {len(df_train)} rec | val {len(df_val)} rec "
          f"| test {len(test_recs)} rec HELD OUT (not loaded)")

    ds_train = FusionSleepDataset(df_train, root, args.window_size, args.overlap)
    ds_val = FusionSleepDataset(df_val, root, args.window_size, 0)
    kw = dict(batch_size=args.batch_size, collate_fn=fusion_collate_fn,
              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_train = DataLoader(ds_train, shuffle=True, drop_last=False, **kw)
    dl_val = DataLoader(ds_val, shuffle=False, **kw)
    print(f"train windows {len(ds_train):,} | val windows {len(ds_val):,}")

    # ---- focal alpha ------------------------------------------------------
    if args.alpha_source == "hardcoded":
        counts = torch.tensor([70154., 25175., 88983., 19454., 34184.])
    else:
        c = {s: 0 for s in STAGES}
        for seq in df_train["stage_sequence"]:
            for s in str(seq).split():
                c[s] += 1
        counts = torch.tensor([float(c[s]) for s in STAGES])
    alpha = 1.0 / counts
    alpha = alpha / alpha.sum() * NUM_CLASSES
    print("focal alpha (" + args.alpha_source + "): "
          + ", ".join(f"{s}={a:.4f}" for s, a in zip(STAGES, alpha.tolist())))

    model = FusedSleepStagingModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,} (expect 649,229)")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")

    criterion = FocalLoss(alpha=alpha, gamma=args.gamma, fix_pt=args.fix_focal_pt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(dl_train),
        epochs=args.epochs, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch, best_macro_f1 = 0, -1.0
    ckpt_last = out / "teacher_last.pt"
    resume_from = args.resume or (ckpt_last if ckpt_last.exists() else None)
    if resume_from and Path(resume_from).exists():
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        # OneCycleLR restores total_steps from the checkpoint. If --epochs or the
        # number of steps per epoch changed, the schedule overruns and dies
        # mid-run. Fail loudly here rather than 40 minutes into a GPU session.
        prev = ck.get("schedule_shape")
        now = {"epochs": args.epochs, "steps_per_epoch": len(dl_train)}
        if prev and prev != now:
            raise SystemExit(
                "Cannot resume: the LR schedule shape changed.\n"
                f"  checkpoint was built for {prev}\n"
                f"  this run would need    {now}\n"
                "OneCycleLR restores total_steps from the checkpoint, so the run "
                "would crash partway through. Either rerun with the original "
                "values, or start fresh (delete teacher_last.pt / pass --resume '')."
            )
        (model.module if hasattr(model, "module") else model).load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_macro_f1 = ck["best_macro_f1"]
        print(f"resumed from {resume_from} at epoch {start_epoch} "
              f"(best macro-F1 {best_macro_f1:.4f})")

    log_path = out / "training_metrics_teacher.jsonl"
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        for xt, xs, y, mask in dl_train:
            xt, xs, y, mask = (xt.to(device, non_blocking=True), xs.to(device, non_blocking=True),
                               y.to(device, non_blocking=True), mask.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(xt, xs, mask)
                loss = criterion(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()

        m = evaluate(model, dl_val, device, use_amp)
        rec = {"epoch": epoch, "train_loss": total / len(dl_train),
               "elapsed_min": round((time.time() - t0) / 60, 2), **m}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        # Decide "is best" BEFORE writing teacher_last.pt. If last.pt stored the
        # pre-update value, a resume would restore a stale best and the next
        # mediocre epoch would overwrite teacher_best.pt with a worse model.
        is_best = m["macro_f1"] > best_macro_f1     # SELECTION ON MACRO-F1
        if is_best:
            best_macro_f1 = m["macro_f1"]

        raw = model.module if hasattr(model, "module") else model
        state = {"epoch": epoch, "model": raw.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "best_macro_f1": best_macro_f1,
                 "args": vars(args), "alpha": alpha.tolist(),
                 "schedule_shape": {"epochs": args.epochs,
                                    "steps_per_epoch": len(dl_train)}}
        torch.save(state, ckpt_last)

        star = ""
        if is_best:
            torch.save(state, out / "teacher_best.pt")
            star = "  <- best"

        pc = "  ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())
        print(f"ep {epoch:3d} | loss {rec['train_loss']:.4f} | "
              f"macroF1 {m['macro_f1']:.4f} | kappa {m['kappa']:.4f} | "
              f"acc {m['accuracy']:.4f} | {pc}{star}", flush=True)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min | best val macro-F1 {best_macro_f1:.4f}")
    print(f"best checkpoint: {out/'teacher_best.pt'}")
    print("Test split was never loaded. Evaluate it once with eval_teacher.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
