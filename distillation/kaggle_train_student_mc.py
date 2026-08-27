"""
================================================================================
KAGGLE NOTEBOOK - MULTI-CHANNEL STUDENT  (experiment "N3mc")
================================================================================
Adds EOG and a second EEG derivation to the multiscale student. Same encoder,
same schedule, same splits, seed 42.

WHY EOG
-------
AASM scoring defines two of the five stages partly by the eyes: rapid eye
movements mark REM, slow rolling eye movements mark N1. Those are the two
classes this model is weakest at (test F1 0.78 and 0.44).

separability_verdict.json established that N1 is REPRESENTATION-bound - no
decision rule on the existing probabilities improves N1 F1 by more than +0.0086.
Representation-bound means only new information helps, and the 34 spectral
features cannot be that information because they are computed FROM the same
EEG. EOG is the first genuinely independent measurement this project has used.

WHY Pz-Oz AS WELL
-----------------
A second cortical derivation, not a summary of the first. Cheap to include once
the tensors are multi-channel. If it does not earn its ~6,000 parameters, drop
it - CHANNELS is a config line.

PER-CHANNEL SCALING IS NOT OPTIONAL
-----------------------------------
The raw EDF is in volts. A single scalar tuned for Fpz-Cz would leave EOG and
Pz-Oz numerically small, and check_branch_live.py already measured what happens
to a numerically small input: it goes INERT, silently, with no error and no
warning. Scales are 1/std per channel, fitted on the TRAINING split only, in
results/multichannel_norm.json.

WHAT SUCCESS LOOKS LIKE
-----------------------
    beat  : student_N2multiscale, validation macro-F1 0.7236, kappa 0.6735
            (same encoder, EEG only - so this isolates the added channels)
    watch : N1 and REM specifically. If the added channels help anywhere, it is
            there. A gain concentrated in W or N2 would suggest the model is
            just using more capacity rather than more information.

Held-out TEST must NOT be used to decide anything here.

ONE CONFOUND, STATED UP FRONT
-----------------------------
student_N2multiscale was trained with a gradient-accumulation bug that this
trainer fixes. Its loss was already `sum / nv` - this chunk's share of the
full-batch mean - and it was then multiplied by `nvc / nv` again. Measured
against a true full-batch backward: gradients came out exactly 0.5x too small
with balanced chunks, and MIS-DIRECTED (cosine 0.9916) whenever padding made
the chunks uneven, because each chunk was effectively weighted by nvc^2 rather
than nvc. AdamW normalises away a uniform rescale; it cannot fix a direction.

So "N3mc beats N2multiscale" would confound the added channels with the
gradient fix. To get a clean read, re-run N2multiscale from the regenerated
kaggle_train_student_v2.py first - it is ~20 minutes - and use ITS number as
the bar. The bar constants below are the OLD (buggy-run) figures until that
happens; update them when it does.

The zeroed-input ablation at the end zeroes ALL channels together. To attribute
per channel, re-run it zeroing one at a time - that is a separate script, not a
knob here.

RUNNING K SEEDS FOR THE ENSEMBLE
--------------------------------
    python kaggle_train_student_mc.py        # seed 42
    python kaggle_train_student_mc.py 1      # seed 1, separate OUT_DIR

Each seed writes to student_N3mc_s<SEED>/, so K runs coexist. The ensemble that
supplies soft targets is built from those checkpoints; see the distillation
plan in the project notes.
================================================================================
"""

import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# ==============================================================================
# CONFIG - every value below is copied from the run that produced
# student_baseline_E0. Change nothing here except EEG_SCALE if you want the
# comparison to remain an A/B test of one variable.
# ==============================================================================
EXPERIMENT     = "N3mc"

# Which of the preprocessed channels the model sees. All three are written by
# preprocess_multichannel.py; this line selects, it does not reprocess.
CHANNELS       = ["EEG Fpz-Cz", "EOG horizontal", "EEG Pz-Oz"]

# 1/std per channel over the TRAIN split, from results/multichannel_norm.json.
# A single scalar would leave EOG and Pz-Oz numerically small, and a numerically
# small input goes INERT - measured, not assumed (check_branch_live.py).
CHANNEL_SCALE  = [22886.88, 14203.08, 15015.06]

DATA_DIR       = "processed_sleepedf_mc"

# THE ONE CHANGE. 1/std over the training split. Set to 1.0 to reproduce the
# original (broken) behaviour exactly.
# Superseded by CHANNEL_SCALE above. Set to None so inherited code that
# still references it fails loudly rather than silently scaling by 1.
EEG_SCALE      = None

CLASS_WEIGHT_POWER = 0.5
EPOCHS         = 75
BATCH_SIZE     = 16
MICRO_BATCH    = 8
LR             = 3e-4
WEIGHT_DECAY   = 1e-2
WARMUP_FRAC    = 0.05
WINDOW_SIZE    = 256
OVERLAP        = 192
import sys as _sys
# The ensemble needs K runs differing only in seed:
#     python kaggle_train_student_mc.py 1
SEED           = int(_sys.argv[1]) if len(_sys.argv) > 1 else 42
EARLY_STOP_PATIENCE = 25
NUM_WORKERS    = 2

EMBED_DIM      = 64
HEADS          = 2
LAYERS         = 2
N_TOKENS       = 4
DROPOUT        = 0.2
USE_SPECTRAL   = True

# The bar, from the stored baseline. VALIDATION figures - the test split is not
# loaded by this script and must not be used to decide whether this worked.
# student_N2multiscale: SAME encoder, EEG only. The bar that isolates
# the added channels, and the one this experiment is judged against.
BASELINE_VAL_MACRO_F1 = 0.7236
BASELINE_VAL_KAPPA    = 0.6735

# student_N1norm: SAME EEG_SCALE, old encoder. This is the bar that
# isolates the encoder; the one above also carries the scaling change.
# student_baseline_E0: the original shipped model, EEG branch inert.
# Kept as the long-run reference, not as the bar.
N1NORM_VAL_MACRO_F1   = 0.6881
N1NORM_VAL_KAPPA      = 0.6389

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

_CH_SCALE = torch.tensor(CHANNEL_SCALE, dtype=torch.float32).view(1, -1, 1)

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
# Lifted verbatim from student_encoder.py by make_v2_trainer.py so the
# two cannot drift apart. Receptive field measured there, not derived.
class ConvBlock(nn.Module):
    """Conv-BN-GELU. BN over (B*T) epochs, which is a large effective batch."""

    def __init__(self, cin, cout, kernel, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv1d(cin, cout, kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))


class MultiScaleEpochEncoder(nn.Module):
    """
    (B, T, L) single-channel or (B, T, C, L) multi-channel -> (B, T, embed).

    Two branches at different time scales, concatenated on the channel axis,
    then pooled to `n_tokens` positions per epoch. With n_tokens > 1 the tokens
    are attention-pooled to a single embedding, so the transformer's sequence
    length is unchanged and the rest of the model is untouched.

    MULTI-CHANNEL
    -------------
    `in_channels > 1` widens the first convolution of each branch; everything
    after is shared. That is the minimal change, and minimal is the point: how
    best to fuse EOG with EEG is its own experiment, and baking an answer in
    here would confound it with the encoder question this class exists to test.

    EOG matters because AASM scoring uses it definitionally - rapid eye
    movements mark REM, slow rolling eye movements mark N1 - and REM and N1 are
    where this model is weakest. Unlike the 34 spectral features, which are
    computed FROM the EEG and so cannot add information, EOG is a genuinely
    independent measurement. It needs `preprocess_multichannel.py` to be run
    first; the tensors currently on disk are single-channel.
    """

    def __init__(self, embed=64, n_tokens=4, fine_ch=(24, 48), coarse_ch=(24, 48),
                 dropout=0.1, in_channels=1):
        super().__init__()
        self.n_tokens = n_tokens
        self.in_channels = in_channels

        f1, f2 = fine_ch
        # 0.5 s kernel at 0.06 s stride: resolves spindles and K-complexes
        self.fine = nn.Sequential(
            ConvBlock(in_channels, f1, kernel=50, stride=6, padding=25),
            nn.MaxPool1d(8, 8),
            nn.Dropout(dropout),
            ConvBlock(f1, f2, kernel=8),
            nn.MaxPool1d(4, 4),
        )

        c1, c2 = coarse_ch
        # 2 s kernel at 0.25 s stride: resolves slow waves, delta, and the slow
        # rolling eye movements that mark N1 when an EOG channel is present
        self.coarse = nn.Sequential(
            ConvBlock(in_channels, c1, kernel=200, stride=25, padding=100),
            nn.MaxPool1d(4, 4),
            nn.Dropout(dropout),
            ConvBlock(c1, c2, kernel=6),
            nn.MaxPool1d(2, 2),
        )

        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.proj = nn.Linear(f2 + c2, embed)
        if n_tokens > 1:
            self.token_attn = nn.Linear(embed, 1)

    def forward(self, x):
        if x.dim() == 3:                                     # (B, T, L)
            B, T, L = x.shape
            z = x.reshape(B * T, 1, L)
        elif x.dim() == 4:                                   # (B, T, C, L)
            B, T, C, L = x.shape
            if C != self.in_channels:
                raise ValueError(f"input has {C} channels, encoder built for "
                                 f"{self.in_channels}")
            z = x.reshape(B * T, C, L)
        else:
            raise ValueError(f"expected (B,T,L) or (B,T,C,L), got {tuple(x.shape)}")
        if z.shape[1] != self.in_channels:
            raise ValueError(f"input has {z.shape[1]} channels, encoder built for "
                             f"{self.in_channels}")

        h = torch.cat([self.pool(self.fine(z)), self.pool(self.coarse(z))], dim=1)
        h = self.proj(h.transpose(1, 2))                     # (B*T, n_tokens, embed)
        if self.n_tokens > 1:
            h = (h * torch.softmax(self.token_attn(h), dim=1)).sum(1)
        else:
            h = h.squeeze(1)
        return h.view(B, T, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class StudentSleepStagingModel(nn.Module):
    """
    Identical to the N1norm student except for `temporal_encoder`.

    The name is kept so that every downstream consumer - evaluate_student.py,
    check_branch_live.py, cache_teacher_logits.py - loads this checkpoint
    without a special case. The config block recorded in the checkpoint carries
    `encoder: "multiscale"` so a checkpoint can still be identified.
    """

    def __init__(self, embed=64, heads=2, layers=2, dropout=0.2,
                 n_tokens=4, max_len=512, use_spectral=True, spec_dim=34):
        super().__init__()
        self.use_spectral = use_spectral
        self.temporal_encoder = MultiScaleEpochEncoder(
            embed=embed, n_tokens=n_tokens, in_channels=len(CHANNELS))
        if use_spectral:
            self.spectral_encoder = nn.Linear(spec_dim, embed)
            self.fusion = nn.Sequential(nn.Linear(embed * 2, embed),
                                        nn.LayerNorm(embed), nn.GELU())
        self.positional_encoding = PositionalEncoding(embed, max_len)
        l = nn.TransformerEncoderLayer(d_model=embed, nhead=heads,
                                       dim_feedforward=embed * 4, dropout=dropout,
                                       batch_first=True, norm_first=True)
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
        # (n, C, 3000) * (1, C, 1) - per channel, not one scalar
        xt = xt[:n].float() * _CH_SCALE
        return xt, xs[:n].float(), y[:n]

def collate(batch):
    L, B = max(b[0].shape[0] for b in batch), len(batch)
    xt = torch.zeros(B, L, *batch[0][0].shape[1:])      # (B, L, C, 3000)
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
    df["tensor_path"] = [DATA_DIR + "/tensors/%s.pt" % r for r in df["rec"]]
    df["spectral"] = ["processed_sleepedf/spectral/%s_spectral.pt" % r for r in df["rec"]]

    tr = df[df["subject"].isin(SPLITS["train"])]
    va = df[df["subject"].isin(SPLITS["val"])]
    assert set(tr["subject"]).isdisjoint(va["subject"])
    assert set(tr["subject"]).isdisjoint(SPLITS["test"])
    assert set(va["subject"]).isdisjoint(SPLITS["test"])

    print(f"\n{'='*72}\nEXPERIMENT {EXPERIMENT}  -  added-channel A/B\n{'='*72}")
    print(f"  channels  = {CHANNELS}")
    print(f"  scales    = {CHANNEL_SCALE}")
    print(f"  seed      = {SEED}")
    print(f"  train {len(tr)} rec / {tr['subject'].nunique()} subj | "
          f"val {len(va)} rec / {va['subject'].nunique()} subj | "
          f"test {len(SPLITS['test'])} subj HELD OUT (never loaded)")
    print(f"  bar to beat: val macro-F1 {BASELINE_VAL_MACRO_F1:.4f}, kappa {BASELINE_VAL_KAPPA:.4f}")

    out = Path(OUT_ROOT) / f"student_{EXPERIMENT}_s{SEED}"
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
    print(f"  parameters: {npar:,}  (N2multiscale was 139,606; {len(CHANNELS)} channels cost +{npar-139606:,})")

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
        _want = {"channels": CHANNELS, "channel_scale": CHANNEL_SCALE,
                 "schedule_shape": {"epochs": EPOCHS, "steps": steps}}
        _have = {k: ck.get(k) for k in _want}
        if _have != _want:
            raise SystemExit("Cannot resume: config changed.\n"
                             f"  checkpoint {_have}\n"
                             f"  this run   {_want}\n"
                             f"Delete {out} to start fresh.")
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
                # KNOWN BUG - LEFT IN PLACE DELIBERATELY.
                # `loss` is already sum/nv, i.e. this chunk's share of the
                # full-batch mean. Multiplying by nvc/nv divides by nv a second
                # time: gradients come out 0.5x too small with balanced chunks,
                # and MIS-DIRECTED (cosine 0.9916) when padding makes chunks
                # uneven, since each chunk is then weighted by nvc^2 not nvc.
                #
                # Not fixed here because this file must keep reproducing the
                # stored student_N1norm checkpoint. The fix is applied by
                # make_v2_trainer.py to every generated trainer; run those.
                # `loss` is ALREADY this chunk's share of the full-batch
                # mean (its sum / nv). Summing over chunks therefore gives
                # the full-batch mean exactly. Do NOT reweight it again.
                scaler.scale(loss).backward()
                tot_loss += loss.item()
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
                 "class_weight_power": CLASS_WEIGHT_POWER, "use_spectral": USE_SPECTRAL, "encoder": "multiscale", "channels": CHANNELS,
                 "channel_scale": CHANNEL_SCALE,
                   "n_tokens": N_TOKENS,
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
    _mf1, _kap = m["macro_f1"], m["kappa"]
    print("\n  vs student_baseline_E0 (the original shipped model, "
          "EEG branch inert):")
    print(f"    macro-F1 {_mf1:.4f} vs {N1NORM_VAL_MACRO_F1:.4f}"
          f"   ({_mf1 - N1NORM_VAL_MACRO_F1:+.4f})")
    print(f"    kappa    {_kap:.4f} vs {N1NORM_VAL_KAPPA:.4f}"
          f"   ({_kap - N1NORM_VAL_KAPPA:+.4f})")
    better = m["macro_f1"] > BASELINE_VAL_MACRO_F1
    print(f"\n  VERDICT: {'BEATS' if better else 'DOES NOT BEAT'} the baseline on validation.")
    print("  Test split was NEVER loaded. Evaluate it once, afterwards, only if this passed.")
    print("\nDownload from /kaggle/working:")
    print(f"  student_{EXPERIMENT}_s{SEED}/student_best.pt + training_metrics.jsonl")


main()
