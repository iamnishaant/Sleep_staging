"""
================================================================================
KAGGLE NOTEBOOK - STUDENT WITH EEG NORMALISATION  (experiment "N1norm")
================================================================================
A/B TEST OF ONE CHANGE. Everything is identical to the run that produced
student_baseline_E0 - same architecture, data, splits, schedule, class
weighting, seed 42 - except that the raw EEG is scaled by a constant before it
reaches the model.

WHY
---
The preprocessed temporal tensors are in VOLTS (MNE's default, never converted).
Stored values sit around 2e-5 while the 34 spectral features sit around 7 - a
scale mismatch of roughly 450,000x.

Verified by ablation, not inferred: zeroing the ENTIRE raw EEG input changes
ZERO predictions in every trained model - the 121K student and both 649K
teachers, prediction agreement 1.00000. The temporal branch is numerically
inert. Every result in this project so far comes from 34 spectral features per
epoch and nothing else. The temporal branch is 10,182 params (8.4% of 121,099);
the bulk is the cross-epoch transformer at 82.6%. So this is a dead INPUT
PATHWAY, not a large block of wasted capacity.

THE FIX
-------
    xt = xt * 15849.46          # 1 / std, std = 6.309e-05 V over the TRAIN split

Fitted on the training split only (distillation/eeg_normalization.py, written
to distillation/results/eeg_normalization.json). Unit variance overall while
PRESERVING relative amplitude between epochs - which matters, because N3 is
defined by high-amplitude delta activity and a per-epoch z-score would normalise
away the single most diagnostic property of the signal.

NOTHING ELSE IS TOUCHED. This is a new file. It does not modify the existing
trainers, the preprocessed tensors, or any checkpoint, so every published result
stays reproducible.

WHAT SUCCESS LOOKS LIKE
-----------------------
    baseline to beat : validation macro-F1 0.6881, kappa 0.6389
                       (student_baseline_E0; its held-out TEST kappa is 0.6449,
                        which must NOT be used to decide anything)

At the end this script re-runs the zeroed-EEG ablation on its own checkpoint.
If prediction agreement is still 1.00000, the branch is STILL dead and the
normalisation did not work - report that, do not tune around it.

SETUP
-----
1. Attach the sleepedf dataset (processed_sleepedf/...).
2. Accelerator -> GPU T4 x2 (single GPU is used; P100 has no kernels here).
3. Run. ~2.5 h. Resumes from /kaggle/working if the session dies.
================================================================================
"""

import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# ==============================================================================
# CONFIG - every value below is copied from the run that produced
# student_baseline_E0. Change nothing here except EEG_SCALE if you want the
# comparison to remain an A/B test of one variable.
# ==============================================================================
EXPERIMENT     = "N1norm"

# THE ONE CHANGE. 1/std over the training split. Set to 1.0 to reproduce the
# original (broken) behaviour exactly.
EEG_SCALE      = 15849.46

CLASS_WEIGHT_POWER = 0.5
EPOCHS         = 75
BATCH_SIZE     = 16
MICRO_BATCH    = 8
LR             = 3e-4
WEIGHT_DECAY   = 1e-2
WARMUP_FRAC    = 0.05
WINDOW_SIZE    = 256
OVERLAP        = 192
SEED           = 42
EARLY_STOP_PATIENCE = 25
NUM_WORKERS    = 2

EMBED_DIM      = 64
HEADS          = 2
LAYERS         = 2
N_TOKENS       = 1
DROPOUT        = 0.2
USE_SPECTRAL   = True

# The bar, from the stored baseline. VALIDATION figures - the test split is not
# loaded by this script and must not be used to decide whether this worked.
BASELINE_VAL_MACRO_F1 = 0.6881
BASELINE_VAL_KAPPA    = 0.6389

OUT_ROOT       = "/kaggle/working"

# ==============================================================================
import json, math, time
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
IGNORE_INDEX = -100
LAB = list(range(NUM_CLASSES))

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


def find_file(pattern, base="/kaggle/input"):
    hits = sorted(Path(base).glob(f"**/{pattern}"))
    if not hits:
        raise SystemExit(f"Missing required file: {pattern}")
    return hits[0]


# ------------------------------------------------------------------- MODEL ---
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Linear(ch, max(1, ch // r)); self.fc2 = nn.Linear(max(1, ch // r), ch)
    def forward(self, x):
        s = x.mean(-1); s = F.relu(self.fc1(s)); s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)

class AtrousPyramid(nn.Module):
    def __init__(self, inc=1, hid=64, dil=(1, 4)):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(inc, hid, 7, dilation=d, padding=3*d) for d in dil])
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  nn.Conv1d(hid*len(dil), len(dil), 1), nn.Softmax(dim=1))
        self.se = SEBlock(hid); self.proj = nn.Conv1d(hid, hid, 1)
    def forward(self, x):
        f = [F.relu(b(x)) for b in self.branches]
        w = self.gate(torch.cat(f, 1))
        out = 0
        for i, fi in enumerate(f):
            out = out + fi * w[:, i:i+1]
        return self.proj(self.se(out))

class EpochEncoder(nn.Module):
    def __init__(self, embed=64, hid=64, dil=(1, 4), n_tokens=1):
        super().__init__()
        self.n_tokens = n_tokens
        self.pyramid = AtrousPyramid(1, hid, dil)
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.fc = nn.Linear(hid, embed)
        if n_tokens > 1:
            self.token_attn = nn.Linear(embed, 1)
    def forward(self, x):
        B, Tt, L = x.shape
        f = self.pyramid(x.reshape(B*Tt, 1, L))
        f = self.fc(self.pool(f).transpose(1, 2))
        f = (f * torch.softmax(self.token_attn(f), 1)).sum(1) if self.n_tokens > 1 else f.squeeze(1)
        return f.view(B, Tt, -1)

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d); pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0)/d))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class StudentSleepStagingModel(nn.Module):
    """Identical to the class in kaggle_train_student.py - 121,099 params."""
    def __init__(self, embed=64, heads=2, layers=2, dropout=0.2, hid=64, dil=(1, 4),
                 n_tokens=1, max_len=512, use_spectral=True, spec_dim=34):
        super().__init__()
        self.use_spectral = use_spectral
        self.temporal_encoder = EpochEncoder(embed, hid, dil, n_tokens)
        if use_spectral:
            self.spectral_encoder = nn.Linear(spec_dim, embed)
            self.fusion = nn.Sequential(nn.Linear(embed*2, embed),
                                        nn.LayerNorm(embed), nn.GELU())
        self.positional_encoding = PositionalEncoding(embed, max_len)
        l = nn.TransformerEncoderLayer(d_model=embed, nhead=heads, dim_feedforward=embed*4,
                                       dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(l, layers)
        self.cls = nn.Linear(embed, NUM_CLASSES)
    def forward(self, xt, xs=None, mask=None):
        h = self.temporal_encoder(xt)
        if self.use_spectral:
            h = self.fusion(torch.cat([h, self.spectral_encoder(xs)], -1))
        return self.cls(self.encoder(self.positional_encoding(h), src_key_padding_mask=mask))


# -------------------------------------------------------------------- DATA ---
class SleepDataset(Dataset):
    """
    Identical to KDSleepDataset with the teacher cache disabled, EXCEPT for the
    single line applying EEG_SCALE. That line is the whole experiment.
    """
    def __init__(self, df, root, window=256, overlap=0):
        self.df = df.reset_index(drop=True); self.root = Path(root)
        self.window = window; self.stride = max(1, window - overlap)
        self.index = []
        for i, row in self.df.iterrows():
            n = len(str(row["stage_sequence"]).split())
            for s in range(0, n, self.stride):
                self.index.append((i, s))
    def __len__(self):
        return len(self.index)
    def __getitem__(self, k):
        i, st = self.index[k]; row = self.df.iloc[i]; en = st + self.window
        stages = str(row["stage_sequence"]).split()[st:en]
        y = torch.tensor([STAGE_TO_IDX[s] for s in stages], dtype=torch.long)
        xt = torch.load(self.root / row["tensor_path"], map_location="cpu")[st:en]
        xs = torch.load(self.root / row["spectral"], map_location="cpu")[st:en]
        n = min(len(y), xt.shape[0], xs.shape[0])
        xt = xt[:n].float() * EEG_SCALE          # <<< THE ONE CHANGE
        return xt, xs[:n].float(), y[:n]

def collate(batch):
    L, B = max(b[0].shape[0] for b in batch), len(batch)
    xt = torch.zeros(B, L, batch[0][0].shape[1])
    xs = torch.zeros(B, L, batch[0][1].shape[1])
    y = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    for i, (a, sp, c) in enumerate(batch):
        n = a.shape[0]
        xt[i, :n], xs[i, :n], y[i, :n], mask[i, :n] = a, sp, c, False
    return xt, xs, y, mask


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval(); yt, yp = [], []
    for xt, xs, y, mask in loader:
        xt, xs, mask = xt.to(device), xs.to(device), mask.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=amp):
            lg = model(xt, xs, mask)
        v = ~mask
        yt.append(y.to(device)[v].cpu().numpy()); yp.append(lg.argmax(-1)[v].cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    _, _, f1, _ = precision_recall_fscore_support(yt, yp, labels=LAB, zero_division=0)
    return {"accuracy": float(accuracy_score(yt, yp)),
            "kappa": float(cohen_kappa_score(yt, yp, labels=LAB)),
            "macro_f1": float(f1_score(yt, yp, average="macro", labels=LAB, zero_division=0)),
            "per_class_f1": {STAGES[i]: float(f1[i]) for i in LAB},
            "prediction_ratio": {STAGES[i]: float((yp == i).sum()/max((yt == i).sum(), 1)) for i in LAB}}


@torch.no_grad()
def ablation_check(model, loader, device, amp):
    """
    THE VERIFICATION THAT MATTERS. Re-run with the EEG zeroed. If prediction
    agreement is still 1.00000 the temporal branch is STILL inert and the
    normalisation did not work - which is a reportable negative, not something
    to tune around.
    """
    model.eval(); same = tot = 0
    for xt, xs, y, mask in loader:
        xt, xs, mask = xt.to(device), xs.to(device), mask.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=amp):
            a = model(xt, xs, mask).argmax(-1)
            b = model(torch.zeros_like(xt), xs, mask).argmax(-1)
        v = ~mask
        same += int((a[v] == b[v]).sum()); tot += int(v.sum())
    return same / max(tot, 1)


# ======================================================================= MAIN =
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0); sm = f"sm_{cap[0]}{cap[1]}"
        sup = torch.cuda.get_arch_list()
        if sup and sm not in sup:
            raise SystemExit(f"UNSUPPORTED GPU ({sm}); build supports {sup}. Use 'GPU T4 x2'.")
        print(f"GPU: {torch.cuda.get_device_name(0)} ({sm})")

    index = find_file("index.csv"); root = index.parent.parent
    df = pd.read_csv(index)
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    df["tensor_path"] = ["processed_sleepedf/tensors/%s.pt" % r for r in df["rec"]]
    df["spectral"] = ["processed_sleepedf/spectral/%s_spectral.pt" % r for r in df["rec"]]

    tr = df[df["subject"].isin(SPLITS["train"])]
    va = df[df["subject"].isin(SPLITS["val"])]
    assert set(tr["subject"]).isdisjoint(va["subject"])
    assert set(tr["subject"]).isdisjoint(SPLITS["test"])
    assert set(va["subject"]).isdisjoint(SPLITS["test"])

    print(f"\n{'='*72}\nEXPERIMENT {EXPERIMENT}  -  EEG normalisation A/B\n{'='*72}")
    print(f"  EEG_SCALE = {EEG_SCALE}   ({'ACTIVE' if EEG_SCALE != 1.0 else 'DISABLED - reproduces the broken run'})")
    print(f"  train {len(tr)} rec / {tr['subject'].nunique()} subj | "
          f"val {len(va)} rec / {va['subject'].nunique()} subj | "
          f"test {len(SPLITS['test'])} subj HELD OUT (never loaded)")
    print(f"  bar to beat: val macro-F1 {BASELINE_VAL_MACRO_F1:.4f}, kappa {BASELINE_VAL_KAPPA:.4f}")

    out = Path(OUT_ROOT) / f"student_{EXPERIMENT}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)

    ds_tr = SleepDataset(tr, root, WINDOW_SIZE, OVERLAP)
    ds_va = SleepDataset(va, root, WINDOW_SIZE, 0)
    kw = dict(collate_fn=collate, num_workers=NUM_WORKERS, pin_memory=amp)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, **kw)
    dl_va = DataLoader(ds_va, batch_size=MICRO_BATCH, shuffle=False, **kw)
    steps = len(dl_tr)
    print(f"  train windows {len(ds_tr):,} ({steps} steps/epoch) | val {len(ds_va):,}")

    # sanity: what the model actually receives
    _xt, _xs, _y = ds_tr[0]
    print(f"  input scale check -> EEG |x| mean {_xt.abs().mean():.4f} (std {_xt.std():.4f})"
          f" | spectral |x| mean {_xs.abs().mean():.4f}")
    if _xt.abs().mean() < 1e-3:
        print("  *** WARNING: EEG still ~0 after scaling. The branch will stay inert. ***")

    cw = None
    if CLASS_WEIGHT_POWER > 0:
        c = dict.fromkeys(STAGES, 0)
        for s in tr["stage_sequence"]:
            for x in str(s).split():
                c[x] += 1
        counts = torch.tensor([float(c[s]) for s in STAGES])
        cw = (1.0/counts) ** CLASS_WEIGHT_POWER
        cw = (cw/cw.sum()*NUM_CLASSES).to(device)
        print("  class weights: " + ", ".join(f"{s}={w:.3f}" for s, w in zip(STAGES, cw.tolist())))

    model = StudentSleepStagingModel(EMBED_DIM, HEADS, LAYERS, DROPOUT,
                                     n_tokens=N_TOKENS, use_spectral=USE_SPECTRAL).to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {npar:,}  (expect 121,099)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warm = max(1, int(WARMUP_FRAC*EPOCHS*steps)); total = EPOCHS*steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s/warm if s < warm else
        0.5*(1+math.cos(math.pi*(s-warm)/max(1, total-warm))))
    scaler = torch.amp.GradScaler(enabled=amp)

    start, best, since = 0, -1.0, 0
    last = out / "student_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck.get("eeg_scale") != EEG_SCALE or ck.get("schedule_shape") != {"epochs": EPOCHS, "steps": steps}:
            raise SystemExit(f"Cannot resume: config changed "
                             f"(scale {ck.get('eeg_scale')} -> {EEG_SCALE}). Delete {out}.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best, since = ck["epoch"]+1, ck["best_macro_f1"], ck.get("since_best", 0)
        print(f"  RESUMED at epoch {start} (best {best:.4f})")

    print()
    t0 = time.time()
    for ep in range(start, EPOCHS):
        model.train(); tot_loss = 0.0
        for xt, xs, y, mask in dl_tr:
            xt, xs = xt.to(device, non_blocking=True), xs.to(device, non_blocking=True)
            y, mask = y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            nv = (y != IGNORE_INDEX).sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            for c0 in range(0, xt.size(0), MICRO_BATCH):
                sl = slice(c0, c0+MICRO_BATCH)
                nvc = (y[sl] != IGNORE_INDEX).sum()
                if nvc == 0:
                    continue
                with torch.amp.autocast(device_type="cuda", enabled=amp):
                    lg = model(xt[sl], xs[sl], mask[sl])
                    loss = F.cross_entropy(lg.reshape(-1, NUM_CLASSES), y[sl].reshape(-1),
                                           weight=cw, ignore_index=IGNORE_INDEX,
                                           reduction="sum") / nv
                w = (nvc/nv).float()
                scaler.scale(loss*w).backward()
                tot_loss += loss.item()*w.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()

        m = evaluate(model, dl_va, device, amp)
        rec = {"experiment": EXPERIMENT, "epoch": ep, "eeg_scale": EEG_SCALE,
               "loss": tot_loss/steps, "lr": opt.param_groups[0]["lr"],
               "elapsed_min": round((time.time()-t0)/60, 2), **m}
        with open(out / "training_metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

        is_best = m["macro_f1"] > best
        if is_best:
            best, since = m["macro_f1"], 0
        else:
            since += 1
        state = {"experiment": EXPERIMENT, "epoch": ep, "model": model.state_dict(),
                 "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                 "scaler": scaler.state_dict(), "best_macro_f1": best, "since_best": since,
                 "n_parameters": npar, "eeg_scale": EEG_SCALE, "alpha": 1.0, "T": 3.0,
                 "class_weight_power": CLASS_WEIGHT_POWER, "use_spectral": USE_SPECTRAL,
                 "splits": SPLITS, "schedule_shape": {"epochs": EPOCHS, "steps": steps},
                 "args": {"embed_dim": EMBED_DIM, "heads": HEADS, "layers": LAYERS,
                          "dropout": DROPOUT, "n_tokens": N_TOKENS}}
        torch.save(state, last)
        if is_best:
            torch.save(state, out / "student_best.pt")

        pc = " ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())
        print(f"ep {ep:3d} | loss {rec['loss']:.4f} | mF1 {m['macro_f1']:.4f} | "
              f"k {m['kappa']:.4f} | acc {m['accuracy']:.4f} | {pc}"
              f"{'  <- best' if is_best else ''}", flush=True)
        if EARLY_STOP_PATIENCE and since >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stop: no gain for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\nDONE {EXPERIMENT} in {(time.time()-t0)/60:.1f} min | best val macro-F1 {best:.4f}")

    # ---------------------------------------------------------------- verdict
    ck = torch.load(out / "student_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    m = evaluate(model, dl_va, device, amp)
    agree = ablation_check(model, dl_va, device, amp)

    print(f"\n{'='*72}\nRESULT\n{'='*72}")
    print(f"  val macro-F1  {m['macro_f1']:.4f}   vs baseline {BASELINE_VAL_MACRO_F1:.4f}"
          f"   ({m['macro_f1']-BASELINE_VAL_MACRO_F1:+.4f})")
    print(f"  val kappa     {m['kappa']:.4f}   vs baseline {BASELINE_VAL_KAPPA:.4f}"
          f"   ({m['kappa']-BASELINE_VAL_KAPPA:+.4f})")
    print(f"\n  IS THE TEMPORAL BRANCH LIVE?")
    print(f"    prediction agreement with EEG zeroed: {agree:.5f}")
    if agree > 0.9995:
        print("    -> STILL INERT. Normalisation did not revive the branch.")
        print("       Report as a negative finding; do not tune around it.")
    else:
        print(f"    -> LIVE. The EEG now changes {(1-agree)*100:.2f}% of predictions.")
    better = m["macro_f1"] > BASELINE_VAL_MACRO_F1
    print(f"\n  VERDICT: {'BEATS' if better else 'DOES NOT BEAT'} the baseline on validation.")
    print("  Test split was NEVER loaded. Evaluate it once, afterwards, only if this passed.")
    print("\nDownload from /kaggle/working:")
    print(f"  student_{EXPERIMENT}/student_best.pt + training_metrics.jsonl")


main()
