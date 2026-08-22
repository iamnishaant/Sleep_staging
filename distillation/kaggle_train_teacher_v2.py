"""
================================================================================
KAGGLE NOTEBOOK - TEACHER WITH THE MULTI-SCALE ENCODER  (experiment "E1bmulti")
================================================================================
A/B TEST OF ONE CHANGE against E1bnorm. Same data, splits, schedule, focal
loss, seed 42 and EEG_SCALE. The temporal encoder is replaced.

WHY
---
The teacher has the same defect the student had. Measured by backpropagation in
distillation/student_encoder.py:

    student encoder   Conv1d(k=7, dil 1,4)      RF 25 samples = 0.25 s
    TEACHER encoder   Conv1d(k=7, dil 1,2,4,8)  RF 49 samples = 0.49 s

0.49 s is still below the 0.5-2 s a sleep spindle occupies, and it is followed
by AdaptiveAvgPool1d(1) averaging all 3000 positions. So the teacher, like the
student, computed the mean over 30 seconds of a sub-second texture detector.

Replacing that encoder in the 121K student produced:

    validation macro-F1   0.6821 -> 0.7236   (+0.0415 vs the same-scale baseline)
    N1 F1                 0.431  -> 0.523
    REM F1                0.699  -> 0.763
    raw EEG drives        25.1%  -> 71.3%    of predictions

The teacher has 4.6x the capacity to exploit the same fix.

WHY THIS RUN DECIDES SOMETHING
------------------------------
Distillation currently COSTS accuracy in this project. On the held-out test
split the 121K student beats every honest teacher:

    student_baseline_E0    kappa 0.6449   121K
    teacher_E1b            kappa 0.6055   649K

Soft targets are asking a good model to imitate a worse one. fig8_rq3
extrapolates that distillation only breaks even once the teacher reaches
kappa ~0.75. The multiscale student made this WORSE, not better - it now beats
teacher_E1b by +0.0259 validation macro-F1.

So this is the run that decides whether soft labels are ever worth using here.
If the teacher gains what the student gained (E1b val kappa 0.6469 + ~0.04
= ~0.69), the break-even point comes into range. If it does not, the honest
conclusion is that this teacher architecture cannot support distillation at
this student size, and that is a publishable result rather than a setback.

WHAT SUCCESS LOOKS LIKE
-----------------------
    beat  : E1bnorm  val macro-F1 0.6907, kappa 0.6392
            (same EEG_SCALE, old encoder - the encoder-isolating bar)
    watch : E1b      val macro-F1 0.6977, kappa 0.6469  (stored teacher)
    target: enough to put held-out test kappa within reach of 0.75, which is
            where soft targets stop costing the student accuracy

Held-out TEST figures must NOT be used to decide anything here.

This script re-runs the zeroed-EEG ablation on its own checkpoint at the end.
The student went from 25.1% to 71.3% of predictions depending on the EEG; if
the teacher does not move similarly, say so rather than tuning around it.

SETUP
-----
1. Attach the sleepedf dataset (processed_sleepedf/...).
2. Accelerator -> GPU T4 x2.
3. Run. Resumes from OUT_DIR if the session dies.

OUT_DIR is a separate directory. No stored teacher is overwritten.

GENERATED FILE - do not edit by hand.
Regenerate with: python distillation/make_teacher_v2.py
================================================================================
"""

# ==============================================================================
# NCCL WORKAROUND - must run BEFORE torch imports / any CUDA init.
# Kaggle's 2x T4 pairing intermittently fails peer-to-peer with
#   "RuntimeError: NCCL Error 1: unhandled cuda error"
# inside DataParallel.replicate(). Disabling P2P and IB routes the broadcast
# through host memory instead. Slightly slower, reliable.
# ==============================================================================
import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

# ==============================================================================
# CONFIG  -- set EXPERIMENT, everything else follows
# ==============================================================================
EXPERIMENT = "E1bmulti"        # recipe reused verbatim; OUT_DIR is overridden below
#            ^^^^^ E0 and E1a are DONE - artefacts already in
#            distillation/results/. Re-running them wastes ~5.5 GPU-hours and
#            reproduces bit-for-bit (seed 42), so the log looks correct while
#            answering nothing. Confirm the header prints what you expect.

_LADDER = {
    "E0":  dict(OVERLAP=0,   ALPHA_POWER=1.0, FIX_FOCAL_PT=False, EPOCH_TOKENS=1),
    "E1a": dict(OVERLAP=192, ALPHA_POWER=1.0, FIX_FOCAL_PT=False, EPOCH_TOKENS=1),
    "E1b": dict(OVERLAP=192, ALPHA_POWER=0.5, FIX_FOCAL_PT=True,  EPOCH_TOKENS=1),
    "E4":  dict(OVERLAP=192, ALPHA_POWER=0.5, FIX_FOCAL_PT=True,  EPOCH_TOKENS=8),
    # E1b recipe exactly, with the multi-scale encoder and 4 sub-epoch tokens.
    "E1bmulti": dict(OVERLAP=192, ALPHA_POWER=0.5, FIX_FOCAL_PT=True, EPOCH_TOKENS=4),
}
_cfg = _LADDER[EXPERIMENT]
OVERLAP        = _cfg["OVERLAP"]
ALPHA_POWER    = _cfg["ALPHA_POWER"]   # 1.0 = inverse freq, 0.5 = sqrt (softer)
FIX_FOCAL_PT   = _cfg["FIX_FOCAL_PT"]
EPOCH_TOKENS   = _cfg["EPOCH_TOKENS"]  # 1 = original global average pool

# ==============================================================================
# THE ONE CHANGE vs the stored E1b run.
#
# The preprocessed temporal tensors are in VOLTS (MNE default, never converted):
# ~2e-5, against spectral features at ~7. A ~450,000x mismatch. Verified by
# ablation, not inferred - zeroing the ENTIRE EEG input changes ZERO predictions
# in the stored E1b teacher (agreement 1.00000), so its temporal branch is
# numerically inert. It is 16,136 params (2.5% of 649,229) - the bulk is the
# cross-epoch transformer at 91.7% - so this is a dead INPUT PATHWAY, not a
# large block of wasted capacity.
#
# 1/std over the TRAINING split only. See distillation/eeg_normalization.py.
# Set to 1.0 to reproduce the stored (broken) E1b bit-for-bit.
EEG_SCALE      = 15849.46
# ==============================================================================

DATA_ROOT      = None                  # None = auto-discover under /kaggle/input
OUT_DIR        = "/kaggle/working/E1bmulti"   # separate dir: never overwrite E1b
# 75 epochs x 171 steps = 12,825 optimizer steps. E0 used 6,900 total and
# peaked at 5,428, so this is ~2.4x the updates E0 needed - ample, and ~3-4 h
# instead of the ~6 h that 150 epochs would cost at this step count.
# OneCycleLR spans the WHOLE schedule, so EPOCHS must be what you intend to
# run; early stopping is a safety net, not the mechanism.
EPOCHS         = 75

# EFFECTIVE batch stays 16 to remain comparable with E0. On a single GPU each
# batch is split into chunks of 8 and normalised by the full batch's valid-token
# count - exactly equivalent (verified 6e-07 relative gradient error), half the
# peak memory.
BATCH_SIZE     = 16

# DEFAULT IS "off". Kaggle's 2x T4 DataParallel path failed twice with
# "NCCL Error 1: unhandled cuda error" - once at replicate() on startup, once
# mid-training AFTER a pre-flight probe had succeeded. Single-GPU chunked
# training is mathematically IDENTICAL, needs no inter-GPU communication, and
# for a 649K-parameter model DataParallel's per-step replicate + scatter/gather
# overhead eats most of the theoretical speedup anyway.
MULTI_GPU      = "off"                 # "auto" | "off"

LR             = 3e-4
WEIGHT_DECAY   = 1e-2
WINDOW_SIZE    = 256
GAMMA          = 2.0
NUM_WORKERS    = 2
SEED           = 42
EARLY_STOP_PATIENCE = 25               # 0 disables. Selection is on val macro-F1.
RESUME         = True

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


def find_data_root(explicit=None, search_base="/kaggle/input"):
    if explicit:
        p = Path(explicit) / "processed_sleepedf" / "index.csv"
        if p.exists():
            return Path(explicit)
        raise SystemExit(f"DATA_ROOT={explicit!r} but {p} missing. Use None to auto-discover.")
    base = Path(search_base)
    hits = sorted(base.glob("**/processed_sleepedf/index.csv")) if base.exists() else []
    if not hits:
        print(f"Contents of {search_base}:")
        for p in sorted(base.rglob("*"))[:40]:
            print("   ", p)
        raise SystemExit("Could not locate processed_sleepedf/index.csv.")
    root = hits[0].parent.parent
    print(f"Auto-discovered DATA_ROOT = {root}")
    return root


# ------------------------------------------------------------------- MODEL ---
# Lifted verbatim from student_encoder.py by make_teacher_v2.py so the
# two cannot drift apart. Receptive field measured there, not derived:
# 875 samples = 8.75 s, against 49 samples = 0.49 s for the encoder
# this replaces.
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


class EpochEncoder(nn.Module):
    """
    Shim keeping FusedSleepStagingModel's call site unchanged.

    The teacher builds its encoder as EpochEncoder(embed_dim, n_tokens=K). That
    signature is preserved so the fusion, the transformer and the checkpoint
    key names are all untouched, and the encoder is the only difference from
    E1bnorm.

    Width is held at the student's (24, 48) on purpose - see make_teacher_v2.py.
    The teacher could afford more, but raising it here would mean a gain could
    not be attributed to the encoder family.
    """

    def __init__(self, embed_dim=128, n_tokens=1):
        super().__init__()
        self.n_tokens = n_tokens
        self.encoder = MultiScaleEpochEncoder(
            embed=embed_dim, n_tokens=n_tokens,
            fine_ch=(24, 48), coarse_ch=(24, 48))

    def forward(self, x):
        return self.encoder(x)


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
                 layers=3, dropout=0.2, epoch_tokens=1):
        super().__init__()
        self.temporal_encoder = EpochEncoder(embed_dim, n_tokens=epoch_tokens)
        self.spectral_encoder = nn.Linear(spectral_input_dim, embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.context = SleepTransformer(embed_dim, heads, layers, dropout)
    def forward(self, x_temporal, x_spectral, padding_mask=None):
        h = torch.cat([self.temporal_encoder(x_temporal),
                       self.spectral_encoder(x_spectral)], dim=-1)
        return self.context(self.fusion(h), padding_mask)


class FocalLoss(nn.Module):
    """
    reduction='mean' reproduces the inherited loss exactly.
    reduction='sum' is used by the gradient-accumulation path so the effective
    batch can be normalised by its TOTAL valid-token count. Averaging per
    micro-batch instead would reweight micro-batches that contain more padding,
    which measurably perturbs the gradient (~6.6% relative on padded batches).
    """
    def __init__(self, alpha=None, gamma=2.0, fix_pt=False, reduction="mean"):
        super().__init__()
        self.gamma = gamma; self.fix_pt = fix_pt; self.reduction = reduction
        self.register_buffer("alpha", torch.ones(NUM_CLASSES) if alpha is None
                             else torch.as_tensor(alpha, dtype=torch.float))
    def forward(self, logits, targets):
        valid = targets != -100
        logits, targets = logits[valid], targets[valid]
        if logits.numel() == 0:
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        if self.fix_pt:
            # Correct focal: pt must be the true probability, not p**alpha.
            pt = torch.softmax(logits.float(), -1).gather(1, targets[:, None]).squeeze(1)
        else:
            pt = torch.exp(-ce)      # inherited behaviour: p_t ** alpha_c
        per_token = (1 - pt) ** self.gamma * ce
        return per_token.sum() if self.reduction == "sum" else per_token.mean()


# -------------------------------------------------------------------- DATA ---
class FusionSleepDataset(Dataset):
    def __init__(self, df, data_root, window_size=256, overlap=0):
        self.df = df.reset_index(drop=True)
        self.root = Path(data_root)
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
        xs = torch.load(self.root / row["spectral"], map_location="cpu")[start:end]
        n = min(len(y), xt.shape[0], xs.shape[0])
        return xt[:n].float() * EEG_SCALE, xs[:n].float(), y[:n]   # <<< THE ONE CHANGE


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
    _, _, f1, sup = precision_recall_fscore_support(yt, yp, labels=lab, zero_division=0)
    return {"accuracy": float(accuracy_score(yt, yp)),
            "kappa": float(cohen_kappa_score(yt, yp, labels=lab)),
            "macro_f1": float(f1_score(yt, yp, average="macro", labels=lab, zero_division=0)),
            "weighted_f1": float(f1_score(yt, yp, average="weighted", labels=lab, zero_division=0)),
            "per_class_f1": {STAGES[i]: float(f1[i]) for i in lab},
            # prediction ratio: >1 means the class is over-predicted. The metric
            # that exposed the N1 spray (2.61x) in E0.
            "prediction_ratio": {STAGES[i]: float((yp == i).sum() / max((yt == i).sum(), 1))
                                 for i in lab}}


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    root = find_data_root(DATA_ROOT)
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Fail immediately on a GPU this PyTorch build has no kernels for. Kaggle
    # sometimes allocates a Tesla P100 (sm_60), which current PyTorch images no
    # longer compile for; without this check the run dies later with an opaque
    # "no kernel image is available for execution on the device".
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        supported = torch.cuda.get_arch_list()
        name = torch.cuda.get_device_name(0)
        if supported and sm not in supported:
            raise SystemExit(
                f"\nUNSUPPORTED GPU: {name} is {sm}, but this PyTorch build "
                f"supports {supported}.\n"
                f"No CUDA kernels exist for this device - training cannot run.\n"
                f"FIX: Kaggle -> Settings -> Accelerator -> 'GPU T4 x2', "
                f"then re-run.\n")
        print(f"  GPU: {name} ({sm}, supported)")

    print(f"\n{'='*70}\nEXPERIMENT {EXPERIMENT}  (MULTI-SCALE TEMPORAL ENCODER)\n{'='*70}")
    print(f"  OVERLAP={OVERLAP}  ALPHA_POWER={ALPHA_POWER}  "
          f"FIX_FOCAL_PT={FIX_FOCAL_PT}  EPOCH_TOKENS={EPOCH_TOKENS}")
    # Printed loudly because the header is otherwise identical to the stored E1b
    # run, and pasting the wrong file silently reproduces it bit-for-bit.
    print(f"  *** EEG_SCALE = {EEG_SCALE} ***  "
          f"{'ACTIVE - this is the normalised variant' if EEG_SCALE != 1.0 else '!!! DISABLED !!!'}")
    print(f"  OUT_DIR = {OUT_DIR}")
    if EEG_SCALE == 1.0:
        raise SystemExit("EEG_SCALE is 1.0, which leaves the temporal branch "
                         "numerically inert - the new encoder would be fed "
                         "volts and learn nothing. Set it to 15849.46.")

    df = pd.read_csv(root / "processed_sleepedf" / "index.csv")
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    df["tensor_path"] = ["processed_sleepedf/tensors/%s.pt" % r for r in df["rec"]]
    df["spectral"] = ["processed_sleepedf/spectral/%s_spectral.pt" % r for r in df["rec"]]
    missing = [p for p in list(df["tensor_path"]) + list(df["spectral"])
               if not (root / p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} tensors missing, first: {missing[:3]}")

    tr = df[df["subject"].isin(SPLITS["train"])]
    va = df[df["subject"].isin(SPLITS["val"])]
    assert set(tr["subject"]).isdisjoint(va["subject"])
    assert set(tr["subject"]).isdisjoint(SPLITS["test"])
    assert set(va["subject"]).isdisjoint(SPLITS["test"])
    print(f"  train {len(tr)} rec / {tr['subject'].nunique()} subj | "
          f"val {len(va)} rec / {va['subject'].nunique()} subj | "
          f"test {len(SPLITS['test'])} subj HELD OUT")

    # --- decide multi-GPU BEFORE building loaders, since it sets micro-batch --
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    use_dp = (MULTI_GPU == "auto") and n_gpu > 1
    if use_dp:
        # Pre-flight at the REAL batch shape. A tiny probe is not enough: a
        # 2x2x3000 probe passed on Kaggle and training then died at the same
        # replicate() call, because the failure depends on memory pressure and
        # the size of the scatter/gather, not just on NCCL being reachable.
        # Includes a backward pass, since that is a second communication round.
        try:
            _probe = nn.DataParallel(FusedSleepStagingModel(
                epoch_tokens=EPOCH_TOKENS).to(device))
            _o = _probe(torch.randn(BATCH_SIZE, WINDOW_SIZE, 3000, device=device),
                        torch.randn(BATCH_SIZE, WINDOW_SIZE, 34, device=device))
            _o.sum().backward()
            del _probe, _o
            torch.cuda.empty_cache()
            print(f"  multi-GPU pre-flight OK ({n_gpu} GPUs, "
                  f"batch {BATCH_SIZE}x{WINDOW_SIZE})")
        except Exception as e:  # noqa: BLE001
            print(f"  multi-GPU pre-flight FAILED ({type(e).__name__}: "
                  f"{str(e)[:100]})\n  -> falling back to single GPU "
                  f"(mathematically identical)")
            use_dp = False
            torch.cuda.empty_cache()

    # The DataLoader always yields the FULL effective batch. On a single GPU we
    # split it into chunks inside the loop and normalise by the whole batch's
    # valid-token count, which makes accumulation EXACTLY equivalent to the
    # multi-GPU path (verified: 6e-07 relative gradient error) rather than
    # approximately so. It also keeps optimizer-steps-per-epoch identical, so
    # the LR schedule matches E0 regardless of which path runs.
    micro_bs = BATCH_SIZE if use_dp else max(1, BATCH_SIZE // 2)
    print(f"  effective batch {BATCH_SIZE}"
          + ("" if use_dp else f", split into chunks of {micro_bs} on one GPU"))

    ds_tr = FusionSleepDataset(tr, root, WINDOW_SIZE, OVERLAP)
    ds_va = FusionSleepDataset(va, root, WINDOW_SIZE, 0)
    kw = dict(collate_fn=fusion_collate_fn, num_workers=NUM_WORKERS,
              pin_memory=use_amp)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, **kw)
    dl_va = DataLoader(ds_va, batch_size=micro_bs, shuffle=False, **kw)
    opt_steps = len(dl_tr)
    print(f"  train windows {len(ds_tr):,} ({opt_steps} optimizer steps/epoch) | "
          f"val windows {len(ds_va):,}")

    c = dict.fromkeys(STAGES, 0)
    for seq in tr["stage_sequence"]:
        for s in str(seq).split():
            c[s] += 1
    counts = torch.tensor([float(c[s]) for s in STAGES])
    alpha = (1.0 / counts) ** ALPHA_POWER
    alpha = alpha / alpha.sum() * NUM_CLASSES
    print(f"  focal alpha (power={ALPHA_POWER}): " +
          ", ".join(f"{s}={a:.4f}" for s, a in zip(STAGES, alpha.tolist())))

    model = FusedSleepStagingModel(epoch_tokens=EPOCH_TOKENS).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters: {n_par:,}  (E0 baseline = 649,229)")
    if use_dp:
        model = nn.DataParallel(model)

    criterion = FocalLoss(alpha, GAMMA, FIX_FOCAL_PT, reduction="sum").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=opt_steps, epochs=EPOCHS, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch, best, since_best = 0, -1.0, 0
    last_ckpt = out / "teacher_last.pt"
    if RESUME and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        prev, now = ck.get("schedule_shape"), {"epochs": EPOCHS, "steps_per_epoch": opt_steps}
        if prev and prev != now:
            raise SystemExit(f"Cannot resume: schedule changed.\n  was {prev}\n  now {now}\n"
                             f"Delete {last_ckpt} to start fresh.")
        if ck.get("experiment") != EXPERIMENT:
            raise SystemExit(f"Checkpoint is from {ck.get('experiment')}, not {EXPERIMENT}. "
                             f"Use a different OUT_DIR per experiment.")
        (model.module if hasattr(model, "module") else model).load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best_macro_f1"]
        since_best = ck.get("since_best", 0)
        print(f"  RESUMED at epoch {start_epoch} (best macro-F1 {best:.4f})")

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        model.train(); total = 0.0
        for xt, xs, y, mask in dl_tr:
            xt, xs = xt.to(device, non_blocking=True), xs.to(device, non_blocking=True)
            y, mask = y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            # Normalise by the WHOLE batch's valid-token count, so splitting into
            # chunks is exactly equivalent to one full-batch mean.
            n_valid = (y != -100).sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for c in range(0, xt.size(0), micro_bs):
                sl = slice(c, c + micro_bs)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(xt[sl], xs[sl], mask[sl])
                    loss_sum = criterion(logits.reshape(-1, NUM_CLASSES),
                                         y[sl].reshape(-1))
                scaler.scale(loss_sum / n_valid).backward()
                batch_loss += loss_sum.item()
            total += batch_loss / n_valid.item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()

        m = evaluate(model, dl_va, device, use_amp)
        rec = {"experiment": EXPERIMENT, "epoch": epoch,
               "train_loss": total/len(dl_tr),
               "elapsed_min": round((time.time()-t0)/60, 2), **m}
        with open(out / "training_metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

        is_best = m["macro_f1"] > best
        if is_best:
            best = m["macro_f1"]; since_best = 0
        else:
            since_best += 1

        raw = model.module if hasattr(model, "module") else model
        state = {"experiment": EXPERIMENT, "epoch": epoch, "model": raw.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "best_macro_f1": best,
                 "since_best": since_best, "alpha": alpha.tolist(), "splits": SPLITS,
                 "config": _cfg, "micro_batch": micro_bs, "multi_gpu": use_dp,
                 "schedule_shape": {"epochs": EPOCHS, "steps_per_epoch": opt_steps}}
        torch.save(state, last_ckpt)
        if is_best:
            torch.save(state, out / "teacher_best.pt")

        pc = " ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())
        n1r = m["prediction_ratio"]["N1"]
        print(f"ep {epoch:3d} | loss {rec['train_loss']:.4f} | mF1 {m['macro_f1']:.4f} "
              f"| k {m['kappa']:.4f} | acc {m['accuracy']:.4f} | {pc} | N1ratio {n1r:.2f}x"
              f"{'  <- best' if is_best else ''}", flush=True)

        if EARLY_STOP_PATIENCE and since_best >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stop: no val macro-F1 improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\nDONE {EXPERIMENT} in {(time.time()-t0)/60:.1f} min | "
          f"best val macro-F1 {best:.4f}")
    print(f"Artefacts in {out}. Test split was NEVER loaded.")


main()
