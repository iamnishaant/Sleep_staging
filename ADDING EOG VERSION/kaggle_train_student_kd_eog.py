"""
================================================================================
KAGGLE NOTEBOOK - EEG + EOG STUDENT  (experiment "N5eog")
================================================================================
Adds an EOG channel to the distilled multiscale student (student_N4kd) in the
way most likely to help REM and N1 without costing the stages EEG already does
well.

THIS FILE LIVES IN "ADDING EOG VERSION/" AND CHANGES NOTHING ELSE IN THE REPO.
Every component that also exists in distillation/kaggle_train_student_kd.py is
copied here verbatim (same class and attribute names), so the N4kd checkpoint
loads into this model without conversion. The original trainer is untouched.

WHY ANOTHER EOG ATTEMPT
-----------------------
student_N3mc_s42 (EEG + EOG + Pz-Oz) measured: test REM F1 +3.7, N1 F1 +2.6,
but validation -0.7 / -0.9, paired p = 0.865 -> reported as noise. That run:
  * stacked EOG into the FIRST convolution of the EEG branches (kernels sized
    for spindles and delta, not eye movements),
  * added EOG and Pz-Oz together (two variables at once),
  * was one seed, hard labels only, judged on 31 + 29 nights.
None of that shows EOG is useless; it shows that test could not tell.

WHAT IS DIFFERENT HERE
----------------------
1. A SEPARATE EOG ENCODER with kernels for eye movements:
     saccade branch  ~0.5 s kernel  -> rapid eye movements      (REM)
     slow branch     ~4 s kernel    -> slow rolling movements   (N1)
   EOG is low-passed at 15 Hz first: above that it is mostly EEG/EMG bleed.
2. EIGHT EOG FEATURES per epoch (band powers 0.3-1/1-2/2-4/4-8 Hz, variance,
   saccade fraction, mean slope, EEG-EOG correlation), standardised on the
   TRAIN split only. Kept OUT of the 34-feature spectral path so the N4kd
   weights still load.
3. GATED FUSION, CLOSED AT START:
     h = h_eeg + m * g * proj(e)
   proj is zero-initialised and the gate bias starts at -4, so at step 0 the
   model is EXACTLY the EEG-only model. The gate learns, per epoch, how much
   to trust EOG. This targets the validation drop seen last time: EOG cannot
   damage stages where EEG is already right unless it earns its way in.
4. WARM START FROM N4kd (original split only), then a short FREEZE: the EEG
   backbone is frozen (lr 0, BatchNorm in eval) for FREEZE_EPOCHS while the
   EOG side learns, then unfrozen at EEG_LR_MULT x the base learning rate.
5. MODALITY DROPOUT: during training the EOG contribution is switched off for
   MODALITY_DROPOUT of sequences (m = 0). The model keeps a working EEG-only
   path - useful when an EOG electrode loses contact - and cannot lean on EOG
   entirely with ~69 training subjects.
6. SAME DISTILLATION as N4kd: ENSmc teacher (which already contains an EOG
   member), alpha 0.5, T 3.0, sqrt-inverse class weights.

ARMS (run both, same seeds, compare paired)
-------------------------------------------
  ARM = "eog"      the design above
  ARM = "eegonly"  identical code path, no EOG modules - architecturally N4kd.
                   The control. Without it, any gain could be warm start or
                   extra epochs rather than EOG.

TWO PROTOCOLS - the guards below enforce which settings are legal
-----------------------------------------------------------------
A) THE EOG QUESTION, under 5-fold CV:   CV_FOLD = 0..4, ALPHA = 1.0,
   WARM_START = False, 3 seeds per arm.  Teacher and N4kd were both trained on
   the original split, so under CV each would leak validation subjects; this
   protocol uses neither. This is the experiment that decides.
B) THE PRODUCTION CANDIDATE, original split:  CV_FOLD = None, ALPHA = 0.5,
   WARM_START = True. Run only if (A) shows EOG helps.

SETUP (Kaggle)
--------------
Attach as datasets:
  1. processed_sleepedf_mc   (index.csv + tensors/, 197 recordings, 3 channels)
  2. processed_sleepedf      (spectral/ - the 34 features N4kd uses)
  3. teacher_logits_ENSmc    (protocol B only)
  4. student_N4kd/student_best.pt   (protocol B only)
Accelerator: GPU T4 x2. Then run this file, or paste it into one cell.
Command-line overrides (optional): --arm eog|eegonly --seed N --cv-fold K

Held-out TEST is never loaded by this script. Do not use it to decide.
================================================================================
"""

import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# ==============================================================================
# CONFIG
# ==============================================================================
EXPERIMENT     = "N5eog"
ARM            = "eog"          # "eog" | "eegonly"
SEED           = 42

# Protocol B defaults (see docstring). For protocol A set CV_FOLD to 0..4,
# ALPHA = 1.0 and WARM_START = False - the guards refuse anything else.
CV_FOLD        = None
ALPHA          = 0.5
T              = 3.0
TEACHER_TAG    = "ENSmc"
WARM_START     = True
WARM_START_FILE = "student_N4kd/student_best.pt"

# Scaling. Channel 0 of processed_sleepedf_mc is bit-identical to
# processed_sleepedf (verified on SC4001E0: torch.equal -> True), so the EEG
# takes N4kd's scale exactly; anything else would break the warm start.
EEG_SCALE      = 15849.46       # = N4kd's EEG_SCALE
EOG_SCALE      = 14203.08       # 1/std of EOG over the TRAIN split (results/multichannel_norm.json)
EEG_CH, EOG_CH = 0, 1           # channel order in processed_sleepedf_mc: Fpz-Cz | EOG horizontal | Pz-Oz

# EOG design
EOG_LOWPASS_HZ = 15.0
EOG_EMBED      = 32
EOG_FEATURES   = True
N_EOG_FEATURES = 8
SACCADE_THR    = 0.10           # |d EOG| per 10 ms sample, in train-unit-variance units (~1000 uV/s)
MODALITY_DROPOUT = 0.25
GATE_BIAS_INIT = -4.0
FREEZE_EPOCHS  = 10             # warm start only
EEG_LR_MULT    = 0.1            # warm start only: backbone lr after the freeze

# Copied from N4kd - change nothing here, or the comparison is no longer one variable
CLASS_WEIGHT_POWER = 0.5
EPOCHS         = 75
BATCH_SIZE     = 16
MICRO_BATCH    = 8
LR             = 3e-4
WEIGHT_DECAY   = 1e-2
WARMUP_FRAC    = 0.05
WINDOW_SIZE    = 256
OVERLAP        = 192
EARLY_STOP_PATIENCE = 25
NUM_WORKERS    = 2
EMBED_DIM      = 64
HEADS          = 2
LAYERS         = 2
N_TOKENS       = 4
DROPOUT        = 0.2
USE_SPECTRAL   = True
FS             = 100

# Reference points, VALIDATION only (distillation/results/eval_students.json,
# student_N4kd.validation_recomputed). Protocol B compares against these; the
# eegonly arm is still the proper control.
N4KD_VAL = {"accuracy": 0.7611, "kappa": 0.6810, "macro_f1": 0.7253,
            "per_class_f1": {"W": 0.888, "N1": 0.504, "N2": 0.782, "N3": 0.690, "REM": 0.762}}
# Hard-label multiscale EEG-only, 5-fold CV (kaggle_train_student_kd.py).
CV_BASELINE_MACRO_F1 = 0.7419
CV_BASELINE_KAPPA    = 0.7143
CV_BASELINE_SD_KAPPA = 0.0193

OUT_ROOT       = "/kaggle/working"

# ==============================================================================
import json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES = 5
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}
IGNORE_INDEX = -100
LAB = list(range(NUM_CLASSES))

# Copied verbatim from distillation/kaggle_train_student_kd.py
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

CV_VAL_SUBJECTS = {
    0: ['SC400', 'SC403', 'SC406', 'SC410', 'SC411', 'SC417', 'SC425', 'SC443', 'SC447',
        'SC457', 'SC463', 'SC465', 'SC467', 'SC477', 'ST701', 'ST713', 'ST720'],
    1: ['SC404', 'SC422', 'SC428', 'SC430', 'SC441', 'SC442', 'SC445', 'SC450', 'SC459',
        'SC471', 'SC473', 'SC481', 'SC482', 'ST708', 'ST719', 'ST721', 'ST722'],
    2: ['SC413', 'SC415', 'SC418', 'SC427', 'SC434', 'SC435', 'SC436', 'SC438', 'SC440',
        'SC449', 'SC454', 'SC456', 'SC476', 'ST704', 'ST709', 'ST711', 'ST712'],
    3: ['SC408', 'SC416', 'SC419', 'SC426', 'SC431', 'SC432', 'SC444', 'SC446', 'SC448',
        'SC455', 'SC464', 'SC470', 'SC480', 'ST706', 'ST707', 'ST715', 'ST717'],
    4: ['SC405', 'SC407', 'SC409', 'SC412', 'SC424', 'SC433', 'SC451', 'SC458', 'SC460',
        'SC462', 'SC466', 'SC472', 'SC475', 'ST702', 'ST705', 'ST718', 'ST724'],
}


# ------------------------------------------------------------------ GUARDS ---
def check_config(cv_fold, alpha, warm_start, arm):
    """Refuse settings that would leak validation subjects. Raises SystemExit."""
    if arm not in ("eog", "eegonly"):
        raise SystemExit(f"ARM must be 'eog' or 'eegonly', got {arm!r}")
    if cv_fold is not None and cv_fold not in CV_VAL_SUBJECTS:
        raise SystemExit(f"CV_FOLD must be None or 0..4, got {cv_fold!r}")
    if cv_fold is not None and alpha < 1.0:
        raise SystemExit(
            "CV_FOLD is set and the soft term is active (ALPHA < 1.0). Refused.\n"
            f"  teacher_logits_{TEACHER_TAG} come from a teacher trained on the original\n"
            "  69-subject split, which contains 12-16 of every fold's validation\n"
            "  subjects. Protocol A uses ALPHA = 1.0.")
    if cv_fold is not None and warm_start:
        raise SystemExit(
            "CV_FOLD is set and WARM_START is on. Refused.\n"
            "  student_N4kd was trained on the original split, which contains most of\n"
            "  every fold's validation subjects; warm-starting from it leaks them.\n"
            "  Protocol A uses WARM_START = False.")


def build_splits(cv_fold):
    if cv_fold is None:
        return {k: list(v) for k, v in SPLITS.items()}
    pool = list(SPLITS["train"]) + list(SPLITS["val"])
    assert len(pool) == 85, f"CV pool should be 85 subjects, got {len(pool)}"
    va = CV_VAL_SUBJECTS[cv_fold]
    assert set(va) <= set(pool), "fold validation subjects must come from the pool"
    assert not (set(va) & set(SPLITS["test"])), "a CV fold may never contain a test subject"
    return {"train": [s for s in pool if s not in set(va)], "val": list(va),
            "test": list(SPLITS["test"])}


def find_file(pattern, base="/kaggle/input"):
    hits = sorted(Path(base).glob(f"**/{pattern}"))
    if not hits:
        raise SystemExit(f"Missing required file: {pattern}")
    return hits[0]


# ------------------------------------------------------------- EOG SIGNAL ---
def lowpass(x, fs=FS, cutoff=EOG_LOWPASS_HZ, order=4):
    """Zero-phase Butterworth low-pass along the last axis of a 1-D array."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


EOG_BANDS = ((0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))


def eog_epoch_features(eog, eeg, fs=FS, saccade_thr=SACCADE_THR):
    """
    eog, eeg: (n_epochs, samples) - scaled; eog already low-passed.
    Returns (n_epochs, 8) float32, UNSTANDARDISED:
      0-3  log10 Welch power in 0.3-1, 1-2, 2-4, 4-8 Hz   (slow rolling -> low bands)
      4    log10 variance
      5    saccade fraction: share of samples with |d eog| > saccade_thr   (REM)
      6    log10 mean |d eog|
      7    Pearson r between EOG and low-passed EEG (frontal EEG picks up eye
           movement; low r means the EOG carries information the EEG does not)
    """
    from scipy.signal import welch
    eog = np.asarray(eog, dtype=np.float64)
    eeg = np.asarray(eeg, dtype=np.float64)
    n = eog.shape[0]
    f, p = welch(eog, fs=fs, nperseg=min(1000, eog.shape[1]), axis=-1)
    out = np.zeros((n, N_EOG_FEATURES), dtype=np.float64)
    for i, (lo, hi) in enumerate(EOG_BANDS):
        sel = (f >= lo) & (f < hi)
        out[:, i] = np.log10(p[:, sel].mean(axis=1) + 1e-8)
    out[:, 4] = np.log10(eog.var(axis=1) + 1e-8)
    d = np.abs(np.diff(eog, axis=1))
    out[:, 5] = (d > saccade_thr).mean(axis=1)
    out[:, 6] = np.log10(d.mean(axis=1) + 1e-8)
    eeg_lp = lowpass(eeg, fs=fs)
    a = eog - eog.mean(axis=1, keepdims=True)
    b = eeg_lp - eeg_lp.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)) + 1e-12
    out[:, 7] = (a * b).sum(axis=1) / den
    return out.astype(np.float32)


def recording_eog_features(mc_tensor):
    """(n, C, 3000) raw volts -> (n, 8) features for one whole night. The EOG is
    filtered as ONE continuous signal, so epoch edges carry no filter artefact."""
    x = mc_tensor.float().numpy() if torch.is_tensor(mc_tensor) else np.asarray(mc_tensor)
    n, _, L = x.shape
    eog = lowpass((x[:, EOG_CH, :] * EOG_SCALE).reshape(-1)).reshape(n, L)
    eeg = x[:, EEG_CH, :] * EEG_SCALE
    return eog_epoch_features(eog, eeg)


# ------------------------------------------------------------------- MODEL ---
# ConvBlock / MultiScaleEpochEncoder / PositionalEncoding: copied verbatim from
# distillation/kaggle_train_student_kd.py (single-channel path), so the
# student_N4kd state_dict keys match exactly.
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
    """(B, T, L) single-channel EEG -> (B, T, embed). Identical to N4kd's."""

    def __init__(self, embed=64, n_tokens=4, fine_ch=(24, 48), coarse_ch=(24, 48),
                 dropout=0.1, in_channels=1):
        super().__init__()
        self.n_tokens = n_tokens
        self.in_channels = in_channels
        f1, f2 = fine_ch
        self.fine = nn.Sequential(
            ConvBlock(in_channels, f1, kernel=50, stride=6, padding=25),
            nn.MaxPool1d(8, 8),
            nn.Dropout(dropout),
            ConvBlock(f1, f2, kernel=8),
            nn.MaxPool1d(4, 4),
        )
        c1, c2 = coarse_ch
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
        B, T_, L = x.shape
        z = x.reshape(B * T_, 1, L)
        h = torch.cat([self.pool(self.fine(z)), self.pool(self.coarse(z))], dim=1)
        h = self.proj(h.transpose(1, 2))
        if self.n_tokens > 1:
            h = (h * torch.softmax(self.token_attn(h), dim=1)).sum(1)
        else:
            h = h.squeeze(1)
        return h.view(B, T_, -1)


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


class EOGEncoder(nn.Module):
    """
    (B, T, L) low-passed EOG [+ (B, T, 8) features] -> (B, T, eog_embed).

    saccade branch: 0.5 s kernel at 0.06 s stride  - sharp rapid eye movements
    slow branch:    4 s kernel at 0.5 s stride     - slow rolling movements
    Each branch is pooled to one vector per epoch. Small on purpose (~6.8K params;
    15.1K for the whole EOG side with gate and projection):
    with ~69 training subjects a large EOG side would memorise them.
    """

    def __init__(self, eog_embed=EOG_EMBED, n_features=N_EOG_FEATURES,
                 use_features=True, dropout=0.1):
        super().__init__()
        self.use_features = use_features
        self.saccade = nn.Sequential(
            ConvBlock(1, 8, kernel=50, stride=6, padding=25),
            nn.MaxPool1d(8, 8),
            nn.Dropout(dropout),
            ConvBlock(8, 16, kernel=8),
            nn.AdaptiveMaxPool1d(1),      # max: a saccade is an event, not an average
        )
        self.slow = nn.Sequential(
            ConvBlock(1, 8, kernel=400, stride=50, padding=200),
            nn.Dropout(dropout),
            ConvBlock(8, 16, kernel=5),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cnn_proj = nn.Linear(32, eog_embed)
        if use_features:
            self.feat_proj = nn.Sequential(nn.Linear(n_features, eog_embed), nn.GELU())
        self.norm = nn.LayerNorm(eog_embed)

    def forward(self, xo, xf=None):
        B, T_, L = xo.shape
        z = xo.reshape(B * T_, 1, L)
        h = torch.cat([self.saccade(z).squeeze(-1), self.slow(z).squeeze(-1)], dim=1)
        e = self.cnn_proj(h).view(B, T_, -1)
        if self.use_features:
            if xf is None:
                raise ValueError("EOGEncoder built with use_features=True needs xf")
            e = e + self.feat_proj(xf)
        return self.norm(e)


class StudentSleepStagingModel(nn.Module):
    """
    N4kd's model, plus an optional gated EOG side. With use_eog=False the
    module tree is identical to N4kd's (same keys, same parameter count).

    forward(xe, xs, mask, xo=None, xf=None, eog_on=None)
      xe (B,T,3000) EEG | xs (B,T,34) spectral | mask (B,T) True = padding
      xo (B,T,3000) low-passed EOG | xf (B,T,8) EOG features
      eog_on (B,) in {0,1}: 0 switches EOG off for that sequence (modality
      dropout in training; a missing electrode at inference). xo=None is the
      EEG-only path.
    """

    def __init__(self, embed=64, heads=2, layers=2, dropout=0.2, n_tokens=4,
                 max_len=512, use_spectral=True, spec_dim=34, use_eog=True,
                 eog_embed=EOG_EMBED, eog_features=True, gate_bias=GATE_BIAS_INIT):
        super().__init__()
        self.use_spectral = use_spectral
        self.use_eog = use_eog
        self.temporal_encoder = MultiScaleEpochEncoder(embed=embed, n_tokens=n_tokens)
        if use_spectral:
            self.spectral_encoder = nn.Linear(spec_dim, embed)
            self.fusion = nn.Sequential(nn.Linear(embed * 2, embed),
                                        nn.LayerNorm(embed), nn.GELU())
        if use_eog:
            self.eog_encoder = EOGEncoder(eog_embed, use_features=eog_features)
            self.eog_proj = nn.Linear(eog_embed, embed)
            nn.init.zeros_(self.eog_proj.weight)       # closed at start:
            nn.init.zeros_(self.eog_proj.bias)         # output == EEG-only model
            self.eog_gate = nn.Linear(embed + eog_embed, embed)
            nn.init.zeros_(self.eog_gate.weight)
            nn.init.constant_(self.eog_gate.bias, gate_bias)
        self.positional_encoding = PositionalEncoding(embed, max_len)
        l = nn.TransformerEncoderLayer(d_model=embed, nhead=heads,
                                       dim_feedforward=embed * 4, dropout=dropout,
                                       batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(l, layers)
        self.cls = nn.Linear(embed, NUM_CLASSES)
        self._last_gate = None

    def forward(self, xe, xs=None, mask=None, xo=None, xf=None, eog_on=None):
        h = self.temporal_encoder(xe)
        if self.use_eog and xo is not None:
            e = self.eog_encoder(xo, xf)
            g = torch.sigmoid(self.eog_gate(torch.cat([h, e], -1)))
            add = g * self.eog_proj(e)
            if eog_on is not None:
                add = add * eog_on.view(-1, 1, 1).to(add.dtype)
            self._last_gate = g.detach()
            h = h + add
        else:
            self._last_gate = None
        if self.use_spectral:
            h = self.fusion(torch.cat([h, self.spectral_encoder(xs)], -1))
        return self.cls(self.encoder(self.positional_encoding(h), src_key_padding_mask=mask))


def is_eog_param(name):
    return name.startswith("eog_")


def warm_start_from(model, state_dict):
    """Load an N4kd state_dict. Every backbone key must match; only eog_* may be
    missing. Returns the list of keys left at their fresh initialisation."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise SystemExit(f"Warm start: checkpoint has keys this model lacks: {unexpected[:5]}")
    bad = [k for k in missing if not is_eog_param(k)]
    if bad:
        raise SystemExit(f"Warm start: backbone keys not loaded: {bad[:5]}")
    return list(missing)


def set_backbone_bn_eval(model, frozen):
    """While the backbone is frozen its BatchNorm running stats must not move
    either; lr 0 alone would still let them drift."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm1d) and not is_eog_param(name):
            mod.train(not frozen and model.training)


# ---------------------------------------------------------------- KD LOSS ---
# Copied verbatim from distillation/kaggle_train_student_kd.py (itself lifted
# from kd_loss.py). The T**2 term is mandatory.
def distillation_loss(student_logits, teacher_logits, targets, T=3.0, alpha=0.5,
                      class_weights=None, return_parts=False):
    C = student_logits.size(-1)
    s = student_logits.reshape(-1, C)
    t = teacher_logits.reshape(-1, C)
    y = targets.reshape(-1)
    mask = y != IGNORE_INDEX
    s, t, y = s[mask], t[mask], y[mask]
    if s.numel() == 0:
        z = student_logits.sum() * 0.0
        return (z, {"hard": z, "soft": z}) if return_parts else z
    hard = F.cross_entropy(s, y, weight=class_weights)
    s_log_p = F.log_softmax(s.float() / T, dim=-1)
    t_p = F.softmax(t.float() / T, dim=-1)
    soft = F.kl_div(s_log_p, t_p, reduction="batchmean") * (T ** 2)
    loss = alpha * hard + (1.0 - alpha) * soft
    if return_parts:
        return loss, {"hard": hard.detach(), "soft": soft.detach()}
    return loss


# -------------------------------------------------------------------- DATA ---
def compute_feature_cache(df, root):
    """rec -> (n, 8) unstandardised EOG features, one pass per recording."""
    cache = {}
    for _, row in df.iterrows():
        x = torch.load(Path(root) / row["tensor_path"], map_location="cpu")
        cache[row["rec"]] = torch.from_numpy(recording_eog_features(x))
    return cache


def fit_standardiser(cache, train_recs):
    allf = torch.cat([cache[r] for r in train_recs], 0)
    mu, sd = allf.mean(0), allf.std(0).clamp_min(1e-6)
    return mu, sd


class SleepDatasetEOG(Dataset):
    """Windows of (EEG, low-passed EOG, spectral, EOG features, teacher logits, y)."""

    def __init__(self, df, root, window=256, overlap=0, teacher_dir=None,
                 feat_cache=None, feat_mu=None, feat_sd=None, with_eog=True):
        self.df = df.reset_index(drop=True); self.root = Path(root)
        self.teacher_dir = Path(teacher_dir) if teacher_dir else None
        self.window = window; self.stride = max(1, window - overlap)
        self.with_eog = with_eog
        self.feat_cache, self.mu, self.sd = feat_cache, feat_mu, feat_sd
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
        x = torch.load(self.root / row["tensor_path"], map_location="cpu")[st:en]
        xs = torch.load(self.root / row["spectral"], map_location="cpu")[st:en]
        n = min(len(y), x.shape[0], xs.shape[0])
        if self.teacher_dir is not None:
            d = np.load(self.teacher_dir / f"{row['rec']}.npz")
            tl = torch.from_numpy(d["logits"][st:en].astype(np.float32))
            n = min(n, tl.shape[0]); tl = tl[:n]
        else:
            tl = torch.zeros(n, NUM_CLASSES)
        xe = x[:n, EEG_CH].float() * EEG_SCALE
        if self.with_eog:
            raw = (x[:n, EOG_CH].float() * EOG_SCALE).numpy()
            xo = torch.from_numpy(lowpass(raw.reshape(-1)).reshape(raw.shape).astype(np.float32))
            if self.feat_cache is not None:
                xf = ((self.feat_cache[row["rec"]][st:st + n] - self.mu) / self.sd).clamp(-6, 6)
            else:
                xf = torch.zeros(n, N_EOG_FEATURES)
        else:
            xo = torch.zeros(n, 1); xf = torch.zeros(n, N_EOG_FEATURES)
        return xe, xo, xs[:n].float(), xf, tl, y[:n]


def collate(batch):
    L, B = max(b[0].shape[0] for b in batch), len(batch)
    xe = torch.zeros(B, L, batch[0][0].shape[1])
    xo = torch.zeros(B, L, batch[0][1].shape[1])
    xs = torch.zeros(B, L, batch[0][2].shape[1])
    xf = torch.zeros(B, L, batch[0][3].shape[1])
    tl = torch.zeros(B, L, NUM_CLASSES)
    y = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    for i, (a, o, sp, f, t, c) in enumerate(batch):
        n = a.shape[0]
        xe[i, :n], xo[i, :n], xs[i, :n], xf[i, :n] = a, o, sp, f
        tl[i, :n], y[i, :n], mask[i, :n] = t, c, False
    return xe, xo, xs, xf, tl, y, mask


# -------------------------------------------------------------- EVALUATION ---
@torch.no_grad()
def evaluate(model, loader, device, amp, eog=True):
    """eog=False runs the same weights with EOG switched off (eog_on = 0)."""
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                                 precision_recall_fscore_support)
    model.eval(); yt, yp, gates = [], [], {s: [] for s in STAGES}
    for xe, xo, xs, xf, _tl, y, mask in loader:
        xe, xs, mask = xe.to(device), xs.to(device), mask.to(device)
        use = model.use_eog
        kw = {}
        if use:
            on = torch.full((xe.size(0),), 1.0 if eog else 0.0, device=device)
            kw = dict(xo=xo.to(device), xf=xf.to(device), eog_on=on)
        with torch.amp.autocast(device_type="cuda", enabled=amp):
            lg = model(xe, xs, mask, **kw)
        v = ~mask
        yv = y.to(device)[v]
        yt.append(yv.cpu().numpy()); yp.append(lg.argmax(-1)[v].cpu().numpy())
        if use and eog and model._last_gate is not None:
            gm = model._last_gate.float().mean(-1)[v]
            for i, s in enumerate(STAGES):
                sel = yv == i
                if sel.any():
                    gates[s].append(gm[sel].cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=LAB, zero_division=0)
    out = {"accuracy": float(accuracy_score(yt, yp)),
           "kappa": float(cohen_kappa_score(yt, yp, labels=LAB)),
           "macro_f1": float(f1_score(yt, yp, average="macro", labels=LAB, zero_division=0)),
           "per_class_f1": {STAGES[i]: float(f1[i]) for i in LAB},
           "per_class_precision": {STAGES[i]: float(p[i]) for i in LAB},
           "per_class_recall": {STAGES[i]: float(r[i]) for i in LAB}}
    if any(gates[s] for s in STAGES):
        out["mean_gate_by_true_stage"] = {
            s: float(np.concatenate(gates[s]).mean()) if gates[s] else None for s in STAGES}
    return out


# ================================================================== SCHEDULE =
def make_lr_lambdas(steps, epochs, warm_start, freeze_epochs, eeg_lr_mult):
    """Per-group multipliers on LR: [backbone, eog]. Warmup + cosine for both;
    during a warm start the backbone is 0 for freeze_epochs, then eeg_lr_mult."""
    total = epochs * steps
    warm = max(1, int(WARMUP_FRAC * total))

    def base(s):
        return s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    def backbone(s):
        if not warm_start:
            return base(s)
        return 0.0 if s < freeze_epochs * steps else eeg_lr_mult * base(s)

    return [backbone, base]


# ======================================================================= MAIN =
def parse_overrides(argv):
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--arm"); ap.add_argument("--seed", type=int)
    ap.add_argument("--cv-fold", type=int)
    a, _ = ap.parse_known_args(argv)          # tolerant of Jupyter's own argv
    return a


def main():
    global ARM, SEED, CV_FOLD, ALPHA, WARM_START
    a = parse_overrides(sys.argv[1:])
    if a.arm: ARM = a.arm
    if a.seed is not None: SEED = a.seed
    if a.cv_fold is not None:
        CV_FOLD = a.cv_fold
        # Protocol A is the only legal CV setting; set it rather than refuse a
        # command line that plainly asked for CV.
        ALPHA, WARM_START = 1.0, False
    check_config(CV_FOLD, ALPHA, WARM_START, ARM)
    use_eog = ARM == "eog"
    splits = build_splits(CV_FOLD)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type == "cuda":                      # same check as the original trainers
        cap = torch.cuda.get_device_capability(0); sm = f"sm_{cap[0]}{cap[1]}"
        sup = torch.cuda.get_arch_list()
        if sup and sm not in sup:
            raise SystemExit(f"UNSUPPORTED GPU ({sm}); this PyTorch build supports {sup}. "
                             "In Kaggle choose Accelerator -> 'GPU T4 x2'.")
        print(f"GPU: {torch.cuda.get_device_name(0)} ({sm})")
    else:
        print("WARNING: no GPU - this will be very slow. Kaggle: Accelerator -> GPU T4 x2.")
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)

    index = find_file("processed_sleepedf_mc/index.csv")
    root = index.parent.parent
    df = pd.read_csv(index)
    if "channels" not in df.columns:
        raise SystemExit(f"{index} has no 'channels' column - attach processed_sleepedf_mc.")
    ch = str(df["channels"].iloc[0]).split("|")
    if ch[EEG_CH] != "EEG Fpz-Cz" or ch[EOG_CH] != "EOG horizontal":
        raise SystemExit(f"Unexpected channel order {ch}; expected Fpz-Cz | EOG horizontal | ...")
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    df["subject"] = df["rec"].str[:5]
    df["tensor_path"] = ["processed_sleepedf_mc/tensors/%s.pt" % r for r in df["rec"]]
    spec_root = find_file("processed_sleepedf/spectral").parent.parent
    df["spectral"] = [str(spec_root / "processed_sleepedf/spectral" / f"{r}_spectral.pt") for r in df["rec"]]

    tr = df[df["subject"].isin(splits["train"])]
    va = df[df["subject"].isin(splits["val"])]
    assert set(tr["subject"]).isdisjoint(va["subject"])
    assert set(tr["subject"]).isdisjoint(splits["test"])
    assert set(va["subject"]).isdisjoint(splits["test"])

    name = f"{EXPERIMENT}_{ARM}_s{SEED}" + (f"_cv{CV_FOLD}" if CV_FOLD is not None else "")
    out = Path(OUT_ROOT) / f"student_{name}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\nEXPERIMENT {name}\n{'='*72}")
    print(f"  arm {ARM} | seed {SEED} | cv_fold {CV_FOLD} | alpha {ALPHA} | warm start {WARM_START}")
    print(f"  train {len(tr)} rec / {tr['subject'].nunique()} subj | val {len(va)} rec / "
          f"{va['subject'].nunique()} subj | test {len(splits['test'])} subj NEVER LOADED")

    feat_cache = mu = sd = None
    if use_eog and EOG_FEATURES:
        t0 = time.time()
        feat_cache = compute_feature_cache(pd.concat([tr, va]), root)
        mu, sd = fit_standardiser(feat_cache, list(tr["rec"]))      # TRAIN only
        print(f"  EOG features cached for {len(feat_cache)} recordings in {time.time()-t0:.0f}s")

    tdir = None
    if ALPHA < 1.0:
        tdir = find_file(f"teacher_logits_{TEACHER_TAG}").resolve()
        miss = sorted(set(tr["rec"]) - {p.stem for p in tdir.glob("*.npz")})
        if miss:
            raise SystemExit(f"{len(miss)} training recordings have no teacher logits, first {miss[:3]}")
        print(f"  teacher   {tdir}")

    dkw = dict(feat_cache=feat_cache, feat_mu=mu, feat_sd=sd, with_eog=use_eog)
    ds_tr = SleepDatasetEOG(tr, root, WINDOW_SIZE, OVERLAP, teacher_dir=tdir, **dkw)
    ds_va = SleepDatasetEOG(va, root, WINDOW_SIZE, 0, **dkw)
    lkw = dict(collate_fn=collate, num_workers=NUM_WORKERS, pin_memory=amp)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, **lkw)
    dl_va = DataLoader(ds_va, batch_size=MICRO_BATCH, shuffle=False, **lkw)
    steps = len(dl_tr)

    _xe, _xo, _xs, _xf, _tl, _y = ds_tr[0]
    print(f"  input check -> EEG |x| {_xe.abs().mean():.3f}"
          + (f" | EOG |x| {_xo.abs().mean():.3f} | feat |x| {_xf.abs().mean():.3f}" if use_eog else ""))
    if ALPHA < 1.0 and float(_tl.abs().sum()) == 0.0:
        raise SystemExit("Teacher logits for the first sample are all zero - cache present but empty.")

    cw = None
    if CLASS_WEIGHT_POWER > 0:
        c = dict.fromkeys(STAGES, 0)
        for s in tr["stage_sequence"]:
            for x in str(s).split():
                c[x] += 1
        counts = torch.tensor([float(c[s]) for s in STAGES])
        cw = (1.0 / counts) ** CLASS_WEIGHT_POWER
        cw = (cw / cw.sum() * NUM_CLASSES).to(device)

    model = StudentSleepStagingModel(EMBED_DIM, HEADS, LAYERS, DROPOUT, n_tokens=N_TOKENS,
                                     use_spectral=USE_SPECTRAL, use_eog=use_eog,
                                     eog_features=EOG_FEATURES).to(device)
    if WARM_START:
        ck = torch.load(find_file(WARM_START_FILE), map_location=device, weights_only=False)
        if ck.get("eeg_scale") != EEG_SCALE:
            raise SystemExit(f"Warm start: checkpoint eeg_scale {ck.get('eeg_scale')} != {EEG_SCALE}")
        fresh = warm_start_from(model, ck["model"])
        print(f"  warm start from N4kd: backbone loaded, {len(fresh)} EOG tensors fresh")
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    neog = sum(p.numel() for n, p in model.named_parameters() if is_eog_param(n))
    print(f"  parameters {npar:,} (EOG side {neog:,}; N4kd is 139,606)")

    groups = [{"params": [p for n, p in model.named_parameters() if not is_eog_param(n)]}]
    eog_params = [p for n, p in model.named_parameters() if is_eog_param(n)]
    lambdas = make_lr_lambdas(steps, EPOCHS, WARM_START, FREEZE_EPOCHS, EEG_LR_MULT)
    if eog_params:
        groups.append({"params": eog_params})
    else:
        lambdas = lambdas[:1]          # eegonly arm: one group, no EOG parameters
    opt = torch.optim.AdamW(groups, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)
    scaler = torch.amp.GradScaler(enabled=amp)
    gen = torch.Generator(device="cpu").manual_seed(SEED)

    config = {"experiment": name, "arm": ARM, "seed": SEED, "cv_fold": CV_FOLD, "alpha": ALPHA,
              "T": T, "teacher_tag": TEACHER_TAG, "warm_start": WARM_START,
              "eeg_scale": EEG_SCALE, "eog_scale": EOG_SCALE, "eog_lowpass_hz": EOG_LOWPASS_HZ,
              "eog_features": EOG_FEATURES, "saccade_thr": SACCADE_THR,
              "modality_dropout": MODALITY_DROPOUT, "freeze_epochs": FREEZE_EPOCHS,
              "eeg_lr_mult": EEG_LR_MULT, "splits": splits,
              "schedule_shape": {"epochs": EPOCHS, "steps": steps},
              "args": {"embed_dim": EMBED_DIM, "heads": HEADS, "layers": LAYERS,
                       "dropout": DROPOUT, "n_tokens": N_TOKENS, "use_spectral": USE_SPECTRAL,
                       "use_eog": use_eog, "eog_embed": EOG_EMBED}}

    start, best, since = 0, -1.0, 0
    last = out / "student_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if {k: ck["config"].get(k) for k in ("arm", "seed", "cv_fold", "alpha", "warm_start", "schedule_shape")} != \
           {k: config[k] for k in ("arm", "seed", "cv_fold", "alpha", "warm_start", "schedule_shape")}:
            raise SystemExit(f"Cannot resume: config changed. Delete {out}.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best, since = ck["epoch"] + 1, ck["best_macro_f1"], ck.get("since_best", 0)
        print(f"  RESUMED at epoch {start} (best {best:.4f})")

    t0 = time.time()
    for ep in range(start, EPOCHS):
        model.train()
        frozen = WARM_START and ep < FREEZE_EPOCHS
        set_backbone_bn_eval(model, frozen)
        tot = 0.0
        lr_sum = [0.0] * len(opt.param_groups)      # mean lr actually USED this epoch
        for xe, xo, xs, xf, tl, y, mask in dl_tr:
            for gi, g in enumerate(opt.param_groups):
                lr_sum[gi] += g["lr"]
            xe, xs, tl = xe.to(device, non_blocking=True), xs.to(device, non_blocking=True), tl.to(device, non_blocking=True)
            y, mask = y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            if use_eog:
                xo, xf = xo.to(device, non_blocking=True), xf.to(device, non_blocking=True)
                on = (torch.rand(xe.size(0), generator=gen) >= MODALITY_DROPOUT).float().to(device)
            nv = (y != IGNORE_INDEX).sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            for c0 in range(0, xe.size(0), MICRO_BATCH):
                sl = slice(c0, c0 + MICRO_BATCH)
                nvc = (y[sl] != IGNORE_INDEX).sum()
                if nvc == 0:
                    continue
                kw = dict(xo=xo[sl], xf=xf[sl], eog_on=on[sl]) if use_eog else {}
                with torch.amp.autocast(device_type="cuda", enabled=amp):
                    lg = model(xe[sl], xs[sl], mask[sl], **kw)
                    loss = distillation_loss(lg, tl[sl], y[sl], T=T, alpha=ALPHA,
                                             class_weights=cw) * (nvc / nv).float()
                scaler.scale(loss).backward()
                tot += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()

        m = evaluate(model, dl_va, device, amp, eog=True)
        rec = {"experiment": name, "epoch": ep, "frozen_backbone": frozen, "loss": tot / steps,
               "lr_backbone": lr_sum[0] / steps,
               "lr_eog": lr_sum[1] / steps if len(lr_sum) > 1 else None,
               "elapsed_min": round((time.time() - t0) / 60, 2), **m}
        with open(out / "training_metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        is_best = m["macro_f1"] > best
        best, since = (m["macro_f1"], 0) if is_best else (best, since + 1)
        state = {"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                 "best_macro_f1": best, "since_best": since, "n_parameters": npar,
                 "config": config,
                 "eog_feature_standardiser": None if mu is None else {"mu": mu.tolist(), "sd": sd.tolist()}}
        torch.save(state, last)
        if is_best:
            torch.save(state, out / "student_best.pt")
        pc = " ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())
        gate = m.get("mean_gate_by_true_stage")
        gs = (" | gate " + " ".join(f"{k}={v:.2f}" for k, v in gate.items() if v is not None)) if gate else ""
        print(f"ep {ep:3d}{' F' if frozen else '  '}| loss {rec['loss']:.4f} | mF1 {m['macro_f1']:.4f} | "
              f"k {m['kappa']:.4f} | acc {m['accuracy']:.4f} | {pc}{gs}{'  <- best' if is_best else ''}",
              flush=True)
        if EARLY_STOP_PATIENCE and since >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stop: no gain for {EARLY_STOP_PATIENCE} epochs.")
            break

    # ---------------------------------------------------------------- verdict
    ck = torch.load(out / "student_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    m = evaluate(model, dl_va, device, amp, eog=True)
    result = {"experiment": name, "arm": ARM, "seed": SEED, "cv_fold": CV_FOLD,
              "alpha": ALPHA, "warm_start": WARM_START, "best_epoch": ck["epoch"], "val": m}
    print(f"\n{'='*72}\nRESULT  {name}  (validation)\n{'='*72}")
    print(f"  acc {m['accuracy']:.4f} | kappa {m['kappa']:.4f} | macro-F1 {m['macro_f1']:.4f}")
    print("  per-class F1  " + " ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items()))
    if use_eog:
        m0 = evaluate(model, dl_va, device, amp, eog=False)
        result["val_eog_off"] = m0
        print("\n  WHAT DOES EOG DO IN THIS MODEL? (same weights, EOG switched off)")
        print(f"    kappa  {m0['kappa']:.4f} -> {m['kappa']:.4f}  ({m['kappa']-m0['kappa']:+.4f})")
        for s in STAGES:
            print(f"    {s:<4} F1 {m0['per_class_f1'][s]:.3f} -> {m['per_class_f1'][s]:.3f}"
                  f"  ({m['per_class_f1'][s]-m0['per_class_f1'][s]:+.3f})")
        if m.get("mean_gate_by_true_stage"):
            print("    mean gate by true stage: " + " ".join(
                f"{k}={v:.2f}" for k, v in m["mean_gate_by_true_stage"].items() if v is not None))
            print("    (expected if EOG is doing its job: highest on REM and N1)")
    if CV_FOLD is None:
        print("\n  vs student_N4kd on the SAME validation split (reference; the eegonly arm is the control):")
        print(f"    kappa {m['kappa']:.4f} vs {N4KD_VAL['kappa']:.4f} ({m['kappa']-N4KD_VAL['kappa']:+.4f})")
        for s in ("N1", "REM"):
            print(f"    {s} F1 {m['per_class_f1'][s]:.3f} vs {N4KD_VAL['per_class_f1'][s]:.3f}"
                  f" ({m['per_class_f1'][s]-N4KD_VAL['per_class_f1'][s]:+.3f})")
        print("    One split, one seed decides nothing - see the README's run plan.")
    else:
        print("\n  NO VERDICT FROM ONE FOLD. Run all 5 folds x 3 seeds x both arms, then")
        print("  python summarise_eog_cv.py <folder with the result_*.json files>")
        print(f"  CV reference (hard-label EEG-only): macro-F1 {CV_BASELINE_MACRO_F1:.4f}, "
              f"kappa {CV_BASELINE_KAPPA:.4f} (sd {CV_BASELINE_SD_KAPPA:.4f})")
    (out / f"result_{name}.json").write_text(json.dumps(result, indent=1))
    print("  Test split was NEVER loaded.")
    print(f"\nDownload /kaggle/working/student_{name}/ (student_best.pt, training_metrics.jsonl, result_{name}.json)")


if __name__ == "__main__":
    main()
