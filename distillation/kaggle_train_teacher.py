"""
================================================================================
KAGGLE NOTEBOOK - RETRAIN THE SLEEP-STAGING TEACHER UNDER AN HONEST PROTOCOL
================================================================================
Paste this whole file into ONE Kaggle notebook cell and run.

SETUP
-----
1. Create a Kaggle Dataset containing the `processed_sleepedf/` folder:
       processed_sleepedf/index.csv
       processed_sleepedf/tensors/*.pt        (197 files)
       processed_sleepedf/spectral/*.pt       (197 files)
2. Attach it to the notebook.
3. Set DATA_ROOT below to the mount point (the folder CONTAINING
   processed_sleepedf). Check with:  !ls /kaggle/input
4. Settings -> Accelerator -> GPU T4 x2  (or P100; see BATCH_SIZE note).
5. Run. Checkpoints + metrics land in /kaggle/working.

Splits are INLINED below, so you do not need to upload splits.json.

WHAT IS REPLICATED EXACTLY (inherited design - not tuned)
--------------------------------------------------------
  FusedSleepStagingModel, fusion='concat', embed=128, heads=4, layers=3,
  dropout=0.2, spectral encoder = Linear(34,128), NO auxiliary heads
  FocalLoss(gamma=2.0) with inverse-frequency per-class alpha
  AdamW(lr=3e-4, wd=1e-2) + OneCycleLR(max_lr=3e-4, pct_start=0.05)
  batch 16, grad-clip 1.0, AMP, window 256, overlap 0, 150 epochs
  -> 649,229 trainable parameters

WHAT IS CHANGED (protocol only)
-------------------------------
  1. SUBJECT-level splits (was recording-level, which leaked 30 of 35 val
     subjects). This is the entire point of the rerun.
  2. Checkpoint selection on MACRO-F1 (was weighted-F1, dominated by N2+W).
  3. Test split never loaded during training.
  4. Focal alpha computed from the TRAIN split only (was full-dataset counts,
     which read test statistics). Set ALPHA_SOURCE='hardcoded' to reproduce
     the original constants exactly.

KNOWN BUG IN THE INHERITED LOSS - PRESERVED DELIBERATELY
--------------------------------------------------------
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
`ce` is already alpha-weighted, so exp(-ce) = p_t**alpha_c, not p_t. The focal
modulator becomes (1 - p_t**alpha_c)**gamma. For alpha>1 (N1=1.36, N3=1.76)
this INCREASES focus; for alpha<1 (W=0.49, N2=0.39) it decreases it -
i.e. it amplifies the intended rebalancing rather than breaking it. Correcting
it would change results and break faithful replication. Set FIX_FOCAL_PT=True
only for a separate ablation.
================================================================================
"""

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_ROOT      = None    # None = auto-discover under /kaggle/input. Set a string to force it.
OUT_DIR        = "/kaggle/working"
EPOCHS         = 150
BATCH_SIZE     = 16      # 16 needs 2x T4 (DataParallel). On a SINGLE 16GB GPU use 8.
LR             = 3e-4
WEIGHT_DECAY   = 1e-2
WINDOW_SIZE    = 256
OVERLAP        = 0
GAMMA          = 2.0
NUM_WORKERS    = 2
SEED           = 42
ALPHA_SOURCE   = "train"   # 'train' (protocol-correct) | 'hardcoded' (original)
FIX_FOCAL_PT   = False     # ablation only - leave False for faithful replication
RESUME         = True      # auto-resume from /kaggle/working/teacher_last.pt

# ==============================================================================
import json, math, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_recall_fscore_support)

NUM_CLASSES = 5
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}


def find_data_root(explicit=None, search_base="/kaggle/input"):
    """
    Locate the directory that CONTAINS processed_sleepedf/index.csv.

    Kaggle's mounted directory slug often differs from the dataset's display
    name, and uploading a zipped folder can add an extra nesting level. Rather
    than hardcode a guess, search for the index and derive the root from it.
    """
    if explicit:
        p = Path(explicit) / "processed_sleepedf" / "index.csv"
        if p.exists():
            return Path(explicit)
        raise SystemExit(f"DATA_ROOT set to {explicit!r} but {p} does not exist.\n"
                         f"Set DATA_ROOT = None to auto-discover.")

    base = Path(search_base)
    if not base.exists():
        raise SystemExit(f"{search_base} does not exist - is a dataset attached?")

    hits = sorted(base.glob("**/processed_sleepedf/index.csv"))
    if not hits:
        # Fall back: any index.csv with the expected columns.
        loose = sorted(base.glob("**/index.csv"))
        print("Could not find */processed_sleepedf/index.csv. Contents of "
              f"{search_base}:")
        for p in sorted(base.rglob("*"))[:40]:
            print("   ", p)
        if loose:
            print("\nFound index.csv at these locations instead:")
            for p in loose:
                print("   ", p)
            print("\nMove/rename so the layout is <root>/processed_sleepedf/index.csv, "
                  "or set DATA_ROOT explicitly to the parent of processed_sleepedf.")
        raise SystemExit("Could not locate processed_sleepedf/index.csv.")

    if len(hits) > 1:
        print(f"WARNING: {len(hits)} candidates found; using the first:")
        for h in hits:
            print("   ", h)

    root = hits[0].parent.parent
    print(f"Auto-discovered DATA_ROOT = {root}")

    # Sanity-check the tensor directories are alongside the index.
    for sub in ("tensors", "spectral"):
        d = root / "processed_sleepedf" / sub
        n = len(list(d.glob("*.pt"))) if d.exists() else 0
        print(f"   processed_sleepedf/{sub}: {n} .pt files"
              + ("" if n else "   <-- MISSING/EMPTY"))
    return root

# ------------------------------------------------------------------ SPLITS ---
# Subject-level, cohort-stratified, seed 42. Subject key = recording_id[:5],
# valid for both SC (SC4<ss><night>E0) and ST (ST7<ss><night>J0).
SPLITS = {
    "train": ['SC400','SC403','SC404','SC405','SC406','SC407','SC409','SC410','SC412',
              'SC415','SC416','SC418','SC419','SC422','SC424','SC425','SC426','SC427',
              'SC428','SC430','SC431','SC433','SC436','SC438','SC440','SC441','SC442',
              'SC443','SC445','SC447','SC448','SC449','SC450','SC451','SC454','SC455',
              'SC457','SC459','SC460','SC462','SC463','SC464','SC465','SC466','SC470',
              'SC471','SC472','SC473','SC475','SC476','SC477','SC480','SC481','SC482',
              'ST701','ST702','ST704','ST705','ST706','ST707','ST709','ST711','ST712',
              'ST717','ST718','ST719','ST721','ST722','ST724'],
    "val":   ['SC408','SC411','SC413','SC417','SC432','SC434','SC435','SC444','SC446',
              'SC456','SC458','SC467','ST708','ST713','ST715','ST720'],
    "test":  ['SC401','SC402','SC414','SC420','SC421','SC423','SC429','SC437','SC452',
              'SC453','SC461','SC474','ST710','ST714','ST716'],
}

# ------------------------------------------------------------------- MODEL ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s)); s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)

class AdaptiveAtrousPyramid(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels, hidden_channels, 7, dilation=d, padding=3*d)
            for d in dilations])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_channels*len(dilations), len(dilations), 1),
            nn.Softmax(dim=1))
        self.se = SEBlock(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)
    def forward(self, x):
        feats = [F.relu(b(x)) for b in self.branches]
        w = self.gate(torch.cat(feats, dim=1))
        out = 0
        for i, f in enumerate(feats):
            out = out + f * w[:, i:i+1]
        return self.proj(self.se(out))

class EpochEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.pyramid = AdaptiveAtrousPyramid()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embed_dim)
    def forward(self, x):
        B, T, L = x.shape
        f = self.pyramid(x.reshape(B*T, 1, L))
        f = self.pool(f).squeeze(-1)
        return self.fc(f).view(B, T, -1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SleepTransformer(nn.Module):
    def __init__(self, embed_dim=128, heads=4, layers=3, dropout=0.2, max_len=512):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4,
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
        self.temporal_encoder = EpochEncoder(embed_dim)
        self.spectral_encoder = nn.Linear(spectral_input_dim, embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.context = SleepTransformer(embed_dim, heads, layers, dropout)
    def forward(self, x_temporal, x_spectral, padding_mask=None):
        h = torch.cat([self.temporal_encoder(x_temporal),
                       self.spectral_encoder(x_spectral)], dim=-1)
        return self.context(self.fusion(h), padding_mask)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, fix_pt=False):
        super().__init__()
        self.gamma = gamma; self.fix_pt = fix_pt
        self.register_buffer("alpha", torch.ones(NUM_CLASSES) if alpha is None
                             else torch.as_tensor(alpha, dtype=torch.float))
    def forward(self, logits, targets):
        valid = targets != -100
        logits, targets = logits[valid], targets[valid]
        if logits.numel() == 0:
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        if self.fix_pt:
            pt = torch.softmax(logits.float(), -1).gather(1, targets[:, None]).squeeze(1)
        else:
            pt = torch.exp(-ce)      # inherited: p_t ** alpha_c
        return ((1 - pt) ** self.gamma * ce).mean()

# -------------------------------------------------------------------- DATA ---
class FusionSleepDataset(Dataset):
    def __init__(self, df, data_root, window_size=256, overlap=0):
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
    L, B = max(b[0].shape[0] for b in batch), len(batch)
    xt = torch.zeros(B, L, batch[0][0].shape[1])
    xs = torch.zeros(B, L, batch[0][1].shape[1])
    y = torch.full((B, L), -100, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    for i, (a, b, c) in enumerate(batch):
        T = a.shape[0]
        xt[i, :T], xs[i, :T], y[i, :T], mask[i, :T] = a, b, c, False
    return xt, xs, y, mask

# -------------------------------------------------------------------- EVAL ---
@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval(); yt, yp = [], []
    for xt, xs, y, mask in loader:
        xt, xs, mask = xt.to(device), xs.to(device), mask.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(xt, xs, mask)
        valid = ~mask
        yt.append(y.to(device)[valid].cpu().numpy())
        yp.append(logits.argmax(-1)[valid].cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    lab = list(range(NUM_CLASSES))
    p, r, f1, sup = precision_recall_fscore_support(yt, yp, labels=lab, zero_division=0)
    return {"accuracy": float(accuracy_score(yt, yp)),
            "kappa": float(cohen_kappa_score(yt, yp, labels=lab)),
            "macro_f1": float(f1_score(yt, yp, average="macro", labels=lab, zero_division=0)),
            "weighted_f1": float(f1_score(yt, yp, average="weighted", labels=lab, zero_division=0)),
            "per_class_f1": {STAGES[i]: float(f1[i]) for i in lab},
            "per_class_support": {STAGES[i]: int(sup[i]) for i in lab}}

# -------------------------------------------------------------------- MAIN ---
def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    root = find_data_root(DATA_ROOT)
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    df = pd.read_csv(root / "processed_sleepedf" / "index.csv")
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]

    # The index stores repo-relative paths. If it was uploaded before being
    # repaired it will still hold absolute paths from another machine, so
    # rebuild them from the recording id and verify before training starts.
    df["tensor_path"] = ["processed_sleepedf/tensors/%s.pt" % r for r in df["rec"]]
    df["spectral"] = ["processed_sleepedf/spectral/%s_spectral.pt" % r for r in df["rec"]]
    missing = [p for p in list(df["tensor_path"]) + list(df["spectral"])
               if not (root / p).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} referenced tensor(s) not found under {root}.\n"
            f"  first missing: {missing[:3]}\n"
            "Check that tensors/ and spectral/ were both included in the dataset.")
    print(f"index OK: {len(df)} recordings, all {len(df)*2} tensor files present")

    tr = df[df["subject"].isin(SPLITS["train"])]
    va = df[df["subject"].isin(SPLITS["val"])]
    # test deliberately NOT loaded
    assert set(tr["subject"]).isdisjoint(va["subject"])
    assert set(tr["subject"]).isdisjoint(SPLITS["test"])
    assert set(va["subject"]).isdisjoint(SPLITS["test"])
    print(f"device={device} | GPUs={torch.cuda.device_count()}")
    print(f"train {len(tr)} rec / {tr['subject'].nunique()} subj | "
          f"val {len(va)} rec / {va['subject'].nunique()} subj | "
          f"test {len(SPLITS['test'])} subj HELD OUT (not loaded)")

    ds_tr = FusionSleepDataset(tr, root, WINDOW_SIZE, OVERLAP)
    ds_va = FusionSleepDataset(va, root, WINDOW_SIZE, 0)
    kw = dict(batch_size=BATCH_SIZE, collate_fn=fusion_collate_fn,
              num_workers=NUM_WORKERS, pin_memory=use_amp)
    dl_tr = DataLoader(ds_tr, shuffle=True, **kw)
    dl_va = DataLoader(ds_va, shuffle=False, **kw)
    print(f"train windows {len(ds_tr):,} | val windows {len(ds_va):,}")

    if ALPHA_SOURCE == "hardcoded":
        counts = torch.tensor([70154., 25175., 88983., 19454., 34184.])
    else:
        c = dict.fromkeys(STAGES, 0)
        for seq in tr["stage_sequence"]:
            for s in str(seq).split():
                c[s] += 1
        counts = torch.tensor([float(c[s]) for s in STAGES])
    alpha = 1.0 / counts
    alpha = alpha / alpha.sum() * NUM_CLASSES
    print("focal alpha (%s): %s" % (ALPHA_SOURCE,
          ", ".join(f"{s}={a:.4f}" for s, a in zip(STAGES, alpha.tolist()))))

    model = FusedSleepStagingModel().to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_par:,}  (expect 649,229)")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = FocalLoss(alpha, GAMMA, FIX_FOCAL_PT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(dl_tr), epochs=EPOCHS, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch, best = 0, -1.0
    last_ckpt = out / "teacher_last.pt"
    if RESUME and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        # OneCycleLR restores total_steps from the checkpoint. If EPOCHS or
        # BATCH_SIZE changed between sessions the schedule overruns and the run
        # dies partway through. Fail loudly now instead.
        prev = ck.get("schedule_shape")
        now = {"epochs": EPOCHS, "steps_per_epoch": len(dl_tr)}
        if prev and prev != now:
            raise SystemExit(
                "Cannot resume: LR schedule shape changed.\n"
                f"  checkpoint built for {prev}\n"
                f"  this run needs       {now}\n"
                "Keep EPOCHS and BATCH_SIZE identical across sessions, or delete "
                "/kaggle/working/teacher_last.pt to start fresh.")
        (model.module if hasattr(model, "module") else model).load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best_macro_f1"]
        print(f"RESUMED at epoch {start_epoch} (best macro-F1 {best:.4f})")

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        model.train(); total = 0.0
        for xt, xs, y, mask in dl_tr:
            xt, xs = xt.to(device, non_blocking=True), xs.to(device, non_blocking=True)
            y, mask = y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(xt, xs, mask)
                loss = criterion(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total += loss.item()

        m = evaluate(model, dl_va, device, use_amp)
        rec = {"epoch": epoch, "train_loss": total/len(dl_tr),
               "elapsed_min": round((time.time()-t0)/60, 2), **m}
        with open(out / "training_metrics_teacher.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

        # Decide "is best" BEFORE writing teacher_last.pt, otherwise a resume
        # restores a stale best and the next mediocre epoch overwrites
        # teacher_best.pt with a worse model.
        is_best = m["macro_f1"] > best                 # SELECTION ON MACRO-F1
        if is_best:
            best = m["macro_f1"]

        raw = model.module if hasattr(model, "module") else model
        state = {"epoch": epoch, "model": raw.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "best_macro_f1": best,
                 "alpha": alpha.tolist(), "splits": SPLITS,
                 "schedule_shape": {"epochs": EPOCHS, "steps_per_epoch": len(dl_tr)}}
        torch.save(state, last_ckpt)

        star = ""
        if is_best:
            torch.save(state, out / "teacher_best.pt"); star = "  <- best"
        pc = " ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())
        print(f"ep {epoch:3d} | loss {rec['train_loss']:.4f} | macroF1 {m['macro_f1']:.4f} "
              f"| kappa {m['kappa']:.4f} | acc {m['accuracy']:.4f} | {pc}{star}", flush=True)

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min | best val macro-F1 {best:.4f}")
    print("Artefacts: teacher_best.pt, teacher_last.pt, training_metrics_teacher.jsonl")
    print("Test split was NEVER loaded. Evaluate it once, afterwards.")

main()
