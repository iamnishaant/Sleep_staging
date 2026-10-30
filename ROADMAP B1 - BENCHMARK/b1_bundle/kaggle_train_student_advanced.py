"""
================================================================================
KAGGLE NOTEBOOK - ADVANCED STAGING STUDENT  (experiment "N6adv", <= 200K params)
================================================================================
Aims at the two weak stages, REM and N1, using what the error analysis
(ADDING EOG VERSION - ERROR ANALYSIS/) and the ST-night investigation
(ADDING EOG VERSION - ST NIGHTS/) measured - not generic tricks.

LIVES IN ITS OWN FOLDER AND CHANGES NOTHING ELSE IN THE REPOSITORY. Every
component shared with student_N4kd is copied verbatim, so the N4kd checkpoint
loads unchanged, and with every new component switched on but untrained the
model's output EQUALS N4kd's (tested). Each component must earn its way in.

WHAT IS MEASURED, AND THE COMPONENT AIMED AT IT
-----------------------------------------------
REM: confident errors in the MIDDLE of REM periods, 59% to N2  ->  missing information
  [EOG]  separate eye-movement encoder, gated (rapid eye movements define REM).
         The old EOG model gained REM on 21/31 nights (p=0.011) but lost N2/N3
         through early fusion - here EOG cannot enter where EEG is already right.
  [TF]   learned time-frequency branch: STFT at 0.5 Hz resolution, 0.5-30 Hz.
         The 34 hand-made features have NO sigma band (12-15 Hz, sleep
         spindles) - the textbook N2-vs-REM marker is split across "alpha 8-13"
         and "beta 13-30". This branch sees it directly.
N1: 65% of errors at the EDGE of a run, true class runner-up 90% of the time
  [LOCAL] depthwise temporal convolution across neighbouring epochs (k=7) before
         the transformer: sharp local context for transitions.
  [AUX]  auxiliary head predicting "stage changes here" (weight 0.1): forces a
         representation of boundaries. 65 parameters, unused at inference.
  [SMOOTH] boundary-aware label smoothing: at an epoch where the expert label
         changes, the hard target gives EPS to the neighbouring stage - the
         uncertainty human scorers have exactly there (N1 agreement ~63%).
Training recipe
  [QMASK] epochs with a flat or stuck (>= 1 s identical samples) signal are
         excluded from the LOSS (never from evaluation). An ST recorder fault:
         29/30 ST training recordings contain some, SC none; N4kd is right on
         only 21.6% of such epochs, so training on them teaches noise.
  [GAIN] amplitude augmentation (x0.75-1.33, log-symmetric) with the DWT
         spectral features moved consistently. ST recordings are ~1.8x SC
         amplitude and N4kd scores them best when scaled to x0.7-0.85; a full
         per-recording normalisation was measured to HURT (PROJECT_REPORT 6i,
         and re-measured in ST NIGHTS), because amplitude defines N3. So:
         robustness, not normalisation. Plus 2% Gaussian noise.
  [EMA]  exponential moving average of weights (decay 0.999) for evaluation.

ARMS (ablation - each runs the same code with a flag set)
---------------------------------------------------------
  base      none                          = N4kd architecture and recipe (control)
  recipe    SMOOTH AUX QMASK GAIN EMA     training only, +65 params
  tf        TF                            spindle/theta view
  eog       EOG                           (needs an EOG electrode at inference)
  local     LOCAL
  full_eeg  TF LOCAL + recipe             best EEG-only candidate (deployable on the current device)
  full      EOG TF LOCAL + recipe         everything
"full" is asserted <= 200,000 parameters at construction.

PROTOCOLS (the guards refuse the leaking combinations)
------------------------------------------------------
A) Which components help?  --cv-fold 0..4 -> ALPHA 1.0, no warm start.
   The ENSmc teacher and N4kd were trained on the original split, which holds
   most of every fold's validation subjects.
B) Production candidate: CV_FOLD None, ALPHA 0.5 (ENSmc), WARM_START from N4kd.

SETUP (Kaggle): attach processed_sleepedf_mc (index.csv + tensors/),
processed_sleepedf (spectral/), and for protocol B teacher_logits_ENSmc/ and
student_N4kd/student_best.pt. GPU T4 x2.
    python kaggle_train_student_advanced.py --arm full_eeg --seed 0 --cv-fold 0

The TEST split is never loaded by this script.
================================================================================
"""

import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# ==============================================================================
# CONFIG
# ==============================================================================
EXPERIMENT     = "N6adv"
ARM            = "full_eeg"
SEED           = 0
CV_FOLD        = None
ALPHA          = 0.5
T              = 3.0
TEACHER_TAG    = "ENSmc"
WARM_START     = True
WARM_START_FILE = "student_N4kd/student_best.pt"
PARAM_BUDGET   = 200_000

# Recordings whose expert labels appear SHIFTED in time against the signal
# (ADDING EOG VERSION - ST NIGHTS, section A7: agreement with N4kd jumps by
# 0.08-0.66 kappa when the labels are slid by 6.5-36 min; correctly aligned
# nights peak at offset 0). Likely cause: the preprocessing builds labels from
# hypnogram-relative onsets. Unverified until verify_hypnogram_alignment.py is
# run on the raw files. OFF by default, so results stay comparable with N4kd's
# history; when ON it applies to every arm of the run (train AND validation),
# keeping paired comparisons fair. Test recordings are never listed or loaded.
EXCLUDE_MISALIGNED = False

# THE PROPER FIX. The preprocessing (extract_epochs_and_trim / epoch_bounds)
# drops unscored epochs from the labels, then cuts the FULL signal with trim
# points computed on the SHORTENED labels, so labels drift ahead of the signal
# (audited on all 197 recordings: 54/137 train, 13/31 val recordings affected;
# 5.65% of training labels wrong). ADDING EOG VERSION - ST NIGHTS/
# verify_hypnogram_alignment.py rebuilds the CORRECT label of every stored
# epoch from the raw hypnograms -> corrected_labels_train_val.json (train + val
# only; "?" = never scored, ignored in loss and scoring). The signal tensors are
# unchanged. Re-scored on validation, the same N4kd goes 76.1% -> 79.2%.
# Attach that file as a dataset and pass --corrected-labels. Applies to train
# AND validation; results are not comparable with stored-label runs.
USE_CORRECTED_LABELS = False
CORRECTED_LABELS_FILE = "corrected_labels_train_val.json"
MISALIGNED = {   # recording: best offset in epochs (negative = labels ahead of the signal)
    "ST7191J0-PSG": -72, "ST7182J0-PSG": -66, "ST7201J0-PSG": -48, "ST7202J0-PSG": -44,
    "ST7242J0-PSG": -32, "ST7241J0-PSG": -33, "ST7051J0-PSG": -22, "ST7041J0-PSG": -21,
    "ST7042J0-PSG": -13, "SC4761E0-PSG": -20,
}

ARMS = {
    "base":     dict(eog=False, tf=False, local=False, recipe=False),
    "recipe":   dict(eog=False, tf=False, local=False, recipe=True),
    "tf":       dict(eog=False, tf=True,  local=False, recipe=False),
    "eog":      dict(eog=True,  tf=False, local=False, recipe=False),
    "local":    dict(eog=False, tf=False, local=True,  recipe=False),
    "full_eeg": dict(eog=False, tf=True,  local=True,  recipe=True),
    "full":     dict(eog=True,  tf=True,  local=True,  recipe=True),
}

EEG_SCALE      = 15849.46       # N4kd's; channel 0 of processed_sleepedf_mc is bit-identical to its input
EOG_SCALE      = 14203.08
EEG_CH, EOG_CH = 0, 1
FS             = 100

# components
EOG_LOWPASS_HZ = 15.0
EOG_EMBED      = 32
N_EOG_FEATURES = 8
SACCADE_THR    = 0.10
TF_EMBED       = 32
TF_FMIN, TF_FMAX = 0.5, 30.0
LOCAL_KERNEL   = 7
GATE_BIAS_INIT = -4.0
MODALITY_DROPOUT = 0.25          # EOG and TF each switched off for this share of sequences
# recipe
SMOOTH_EPS     = 0.2
AUX_WEIGHT     = 0.1
GAIN_RANGE     = (0.75, 1.33)    # log-symmetric; ST nights are ~1.8x SC amplitude and score best at x0.7-0.85
NOISE_STD      = 0.02            # in scaled (unit-variance) units
QMASK_MIN_STD_V = 1e-6           # flat: epoch std below 1 uV
QMASK_STUCK_RUN = 100            # stuck: >= 1 s of identical consecutive samples (ST recorder fault)
EMA_DECAY      = 0.999
# warm start
FREEZE_EPOCHS  = 10
NEW_LR_MULT    = 1.0
EEG_LR_MULT    = 0.1

# copied from N4kd
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

N4KD_VAL = {"accuracy": 0.7611, "kappa": 0.6810, "macro_f1": 0.7253,
            "per_class_f1": {"W": 0.888, "N1": 0.504, "N2": 0.782, "N3": 0.690, "REM": 0.762}}
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
LABEL_TO_IDX = {**STAGE_TO_IDX, "?": IGNORE_INDEX}     # "?" = never scored by the expert
LAB = list(range(NUM_CLASSES))
NEW_PREFIXES = ("eog_", "tf_", "loc_", "aux_")

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
    if arm not in ARMS:
        raise SystemExit(f"ARM must be one of {list(ARMS)}, got {arm!r}")
    if cv_fold is not None and cv_fold not in CV_VAL_SUBJECTS:
        raise SystemExit(f"CV_FOLD must be None or 0..4, got {cv_fold!r}")
    if cv_fold is not None and alpha < 1.0:
        raise SystemExit("CV_FOLD with ALPHA < 1.0 refused: the ENSmc teacher was trained on the "
                         "original split, which holds 12-16 of every fold's validation subjects.")
    if cv_fold is not None and warm_start:
        raise SystemExit("CV_FOLD with WARM_START refused: student_N4kd was trained on the original "
                         "split, which holds most of every fold's validation subjects.")


def build_splits(cv_fold):
    if cv_fold is None:
        return {k: list(v) for k, v in SPLITS.items()}
    pool = list(SPLITS["train"]) + list(SPLITS["val"])
    assert len(pool) == 85
    va = CV_VAL_SUBJECTS[cv_fold]
    assert not (set(va) & set(SPLITS["test"]))
    return {"train": [s for s in pool if s not in set(va)], "val": list(va), "test": list(SPLITS["test"])}


def find_file(pattern, base="/kaggle/input"):
    hits = sorted(Path(base).glob(f"**/{pattern}"))
    if not hits:
        raise SystemExit(f"Missing required file: {pattern}")
    return hits[0]


# ------------------------------------------------------------- EOG SIGNAL ---
# copied from ADDING EOG VERSION/kaggle_train_student_kd_eog.py
def lowpass(x, fs=FS, cutoff=EOG_LOWPASS_HZ, order=4):
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(order, cutoff, btype="low", fs=fs, output="sos"), x, axis=-1)


EOG_BANDS = ((0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))


def eog_epoch_features(eog, eeg, fs=FS, saccade_thr=SACCADE_THR):
    from scipy.signal import welch
    eog = np.asarray(eog, dtype=np.float64); eeg = np.asarray(eeg, dtype=np.float64)
    f, p = welch(eog, fs=fs, nperseg=min(1000, eog.shape[1]), axis=-1)
    out = np.zeros((eog.shape[0], N_EOG_FEATURES))
    for i, (lo, hi) in enumerate(EOG_BANDS):
        out[:, i] = np.log10(p[:, (f >= lo) & (f < hi)].mean(axis=1) + 1e-8)
    out[:, 4] = np.log10(eog.var(axis=1) + 1e-8)
    d = np.abs(np.diff(eog, axis=1))
    out[:, 5] = (d > saccade_thr).mean(axis=1)
    out[:, 6] = np.log10(d.mean(axis=1) + 1e-8)
    eeg_lp = lowpass(eeg, fs=fs)
    a = eog - eog.mean(1, keepdims=True); b = eeg_lp - eeg_lp.mean(1, keepdims=True)
    out[:, 7] = (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-12)
    return out.astype(np.float32)


def recording_eog_features(mc_tensor):
    x = mc_tensor.float().numpy() if torch.is_tensor(mc_tensor) else np.asarray(mc_tensor)
    n, _, L = x.shape
    eog = lowpass((x[:, EOG_CH, :] * EOG_SCALE).reshape(-1)).reshape(n, L)
    return eog_epoch_features(eog, x[:, EEG_CH, :] * EEG_SCALE)


# --------------------------------------------------------- RECIPE HELPERS ---
def quality_mask(eeg_volts):
    """(n, 3000) raw volts -> (n,) bool, True = usable for TRAINING. Unusable: flat
    (std < 1 uV) or stuck (>= 1 s of identical consecutive samples). Same rule as
    ADDING EOG VERSION - ST NIGHTS: 29 of 30 ST training recordings contain such
    epochs, no SC recording does, and N4kd is right on only 21.6% of them."""
    x = torch.as_tensor(eeg_volts, dtype=torch.float32)
    flat = x.std(-1) < QMASK_MIN_STD_V
    z = (x.diff(dim=-1) == 0).float().unsqueeze(1)
    stuck = (F.avg_pool1d(z, kernel_size=QMASK_STUCK_RUN - 1, stride=1).squeeze(1) >= 1.0 - 1e-6).any(-1)
    return ~(flat | stuck)


def gain_spectral(xs, g):
    """Move the 34 cached spectral features with an EEG gain g (per sequence).
    DWT block (first 24 = 6 levels x [energy, log_energy, entropy, variance]):
    energy and variance scale by g^2, log_energy by +2 log g (exact - the DWT is
    linear). The STFT/ratio block is scale-free up to a 1e-12 guard term."""
    xs = xs.clone()
    k = torch.arange(24)
    g2 = (g * g).view(-1, *([1] * (xs.dim() - 1)))
    xs[..., k[k % 4 == 0]] = xs[..., k[k % 4 == 0]] * g2
    xs[..., k[k % 4 == 3]] = xs[..., k[k % 4 == 3]] * g2
    xs[..., k[k % 4 == 1]] = xs[..., k[k % 4 == 1]] + torch.log(g2)
    return xs


def boundary_targets(y, n_classes=NUM_CLASSES, eps=SMOOTH_EPS):
    """(B, L) labels -> (B, L, C) target distribution. At an epoch whose expert
    label differs from a valid neighbour, eps moves to that neighbour's stage
    (split if both neighbours differ). Elsewhere one-hot. Padding rows are zero."""
    B, L = y.shape
    valid = y != IGNORE_INDEX
    yc = y.clamp_min(0)
    q = F.one_hot(yc, n_classes).float() * valid.unsqueeze(-1)
    if eps <= 0:
        return q
    prev = torch.full_like(y, IGNORE_INDEX); prev[:, 1:] = y[:, :-1]
    nxt = torch.full_like(y, IGNORE_INDEX); nxt[:, :-1] = y[:, 1:]
    dp = valid & (prev != IGNORE_INDEX) & (prev != y)
    dn = valid & (nxt != IGNORE_INDEX) & (nxt != y)
    k = dp.float() + dn.float()
    share = torch.where(k > 0, eps / k.clamp_min(1), torch.zeros_like(k))
    q = q * (1 - eps * (k > 0).float()).unsqueeze(-1)
    q = q + F.one_hot(prev.clamp_min(0), n_classes).float() * (share * dp.float()).unsqueeze(-1)
    q = q + F.one_hot(nxt.clamp_min(0), n_classes).float() * (share * dn.float()).unsqueeze(-1)
    return q


def transition_targets(y):
    """(B, L) -> (B, L) float target (1 = label differs from the previous epoch)
    and a validity mask (both this epoch and the previous one labelled)."""
    prev = torch.full_like(y, IGNORE_INDEX); prev[:, 1:] = y[:, :-1]
    m = (y != IGNORE_INDEX) & (prev != IGNORE_INDEX)
    return (prev != y).float() * m, m


# ------------------------------------------------------------------- MODEL ---
# ConvBlock / MultiScaleEpochEncoder / PositionalEncoding: verbatim from
# distillation/kaggle_train_student_kd.py (single-channel path).
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv1d(cin, cout, kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))


class MultiScaleEpochEncoder(nn.Module):
    def __init__(self, embed=64, n_tokens=4, fine_ch=(24, 48), coarse_ch=(24, 48), dropout=0.1, in_channels=1):
        super().__init__()
        self.n_tokens = n_tokens
        self.in_channels = in_channels
        f1, f2 = fine_ch
        self.fine = nn.Sequential(ConvBlock(in_channels, f1, kernel=50, stride=6, padding=25),
                                  nn.MaxPool1d(8, 8), nn.Dropout(dropout),
                                  ConvBlock(f1, f2, kernel=8), nn.MaxPool1d(4, 4))
        c1, c2 = coarse_ch
        self.coarse = nn.Sequential(ConvBlock(in_channels, c1, kernel=200, stride=25, padding=100),
                                    nn.MaxPool1d(4, 4), nn.Dropout(dropout),
                                    ConvBlock(c1, c2, kernel=6), nn.MaxPool1d(2, 2))
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.proj = nn.Linear(f2 + c2, embed)
        if n_tokens > 1:
            self.token_attn = nn.Linear(embed, 1)

    def forward(self, x):
        B, T_, L = x.shape
        z = x.reshape(B * T_, 1, L)
        h = torch.cat([self.pool(self.fine(z)), self.pool(self.coarse(z))], dim=1)
        h = self.proj(h.transpose(1, 2))
        h = (h * torch.softmax(self.token_attn(h), dim=1)).sum(1) if self.n_tokens > 1 else h.squeeze(1)
        return h.view(B, T_, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EOGEncoder(nn.Module):
    """Copied from the EOG trainer: saccade (0.5 s) + slow-rolling (4 s) branches + 8 features."""

    def __init__(self, eog_embed=EOG_EMBED, n_features=N_EOG_FEATURES, dropout=0.1):
        super().__init__()
        self.saccade = nn.Sequential(ConvBlock(1, 8, kernel=50, stride=6, padding=25), nn.MaxPool1d(8, 8),
                                     nn.Dropout(dropout), ConvBlock(8, 16, kernel=8), nn.AdaptiveMaxPool1d(1))
        self.slow = nn.Sequential(ConvBlock(1, 8, kernel=400, stride=50, padding=200), nn.Dropout(dropout),
                                  ConvBlock(8, 16, kernel=5), nn.AdaptiveAvgPool1d(1))
        self.cnn_proj = nn.Linear(32, eog_embed)
        self.feat_proj = nn.Sequential(nn.Linear(n_features, eog_embed), nn.GELU())
        self.norm = nn.LayerNorm(eog_embed)

    def forward(self, xo, xf):
        B, T_, L = xo.shape
        z = xo.reshape(B * T_, 1, L)
        h = torch.cat([self.saccade(z).squeeze(-1), self.slow(z).squeeze(-1)], dim=1)
        return self.norm(self.cnn_proj(h).view(B, T_, -1) + self.feat_proj(xf))


class TFEncoder(nn.Module):
    """
    Learned time-frequency view of each EEG epoch.
    STFT: 2 s Hann window, 1 s hop -> 29 frames; 0.5 Hz bins from 0.5 to 30 Hz
    (60 bins, so sigma 12-15 Hz is 7 bins of its own). log power, normalised per
    frequency (BatchNorm), then convolutions over time with frequency as
    channels, mean+max pooled over the epoch. Computed in float32 on the device
    from the EEG already in the batch - no new data files.
    """

    def __init__(self, tf_embed=TF_EMBED, hidden=24, dropout=0.1):
        super().__init__()
        self.n_fft, self.hop = 2 * FS, FS
        f = torch.arange(self.n_fft // 2 + 1) * FS / self.n_fft
        sel = torch.nonzero((f >= TF_FMIN) & (f <= TF_FMAX - 1e-9)).squeeze(-1)
        self.register_buffer("bins", sel, persistent=False)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        nf = len(sel)
        self.norm_in = nn.BatchNorm1d(nf)
        self.conv = nn.Sequential(nn.Conv1d(nf, hidden, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
                                  nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.out = nn.Linear(2 * hidden, tf_embed)
        self.norm = nn.LayerNorm(tf_embed)

    def spectrogram(self, x):
        B, T_, L = x.shape
        z = x.reshape(B * T_, L).float()
        with torch.autocast(device_type=z.device.type, enabled=False):
            S = torch.stft(z, self.n_fft, self.hop, window=self.window, center=False, return_complex=True)
            P = torch.log(S.abs().pow(2) + 1e-6)
        return P[:, self.bins, :]                                   # (B*T, nf, frames)

    def forward(self, xe):
        B, T_, _ = xe.shape
        h = self.conv(self.norm_in(self.spectrogram(xe)))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.norm(self.out(h)).view(B, T_, -1)


class AdvancedStudent(nn.Module):
    """
    N4kd + optional components. Every new path is closed at construction:
    eog_proj / tf_proj / loc_pw are zero-initialised, so with all components on
    and untrained the logits EQUAL N4kd's. Names of new modules start with
    eog_/tf_/loc_/aux_ (used for warm start and learning-rate groups).

    forward(xe, xs, mask, xo=None, xf=None, eog_on=None, tf_on=None, return_aux=False)
    """

    def __init__(self, eog=False, tf=False, local=False, aux=False, embed=EMBED_DIM, heads=HEADS,
                 layers=LAYERS, dropout=DROPOUT, n_tokens=N_TOKENS, spec_dim=34, max_len=512,
                 gate_bias=GATE_BIAS_INIT):
        super().__init__()
        self.use_eog, self.use_tf, self.use_local, self.use_aux = eog, tf, local, aux
        self.temporal_encoder = MultiScaleEpochEncoder(embed=embed, n_tokens=n_tokens)
        self.spectral_encoder = nn.Linear(spec_dim, embed)
        self.fusion = nn.Sequential(nn.Linear(embed * 2, embed), nn.LayerNorm(embed), nn.GELU())
        if eog:
            self.eog_encoder = EOGEncoder()
            self.eog_proj = nn.Linear(EOG_EMBED, embed)
            self.eog_gate = nn.Linear(embed + EOG_EMBED, embed)
            self._close(self.eog_proj, self.eog_gate, gate_bias)
        if tf:
            self.tf_encoder = TFEncoder()
            self.tf_proj = nn.Linear(TF_EMBED, embed)
            self.tf_gate = nn.Linear(embed + TF_EMBED, embed)
            self._close(self.tf_proj, self.tf_gate, gate_bias)
        if local:
            self.loc_dw = nn.Conv1d(embed, embed, LOCAL_KERNEL, padding=LOCAL_KERNEL // 2, groups=embed)
            self.loc_pw = nn.Linear(embed, embed)
            nn.init.zeros_(self.loc_pw.weight); nn.init.zeros_(self.loc_pw.bias)
        self.positional_encoding = PositionalEncoding(embed, max_len)
        l = nn.TransformerEncoderLayer(d_model=embed, nhead=heads, dim_feedforward=embed * 4,
                                       dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(l, layers)
        self.cls = nn.Linear(embed, NUM_CLASSES)
        if aux:
            self.aux_transition = nn.Linear(embed, 1)
        self.gates = {}

    @staticmethod
    def _close(proj, gate, bias):
        nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)
        nn.init.zeros_(gate.weight); nn.init.constant_(gate.bias, bias)

    @staticmethod
    def _switch(add, on):
        return add if on is None else add * on.view(-1, 1, 1).to(add.dtype)

    def forward(self, xe, xs, mask=None, xo=None, xf=None, eog_on=None, tf_on=None, return_aux=False):
        h = self.temporal_encoder(xe)
        self.gates = {}
        if self.use_eog and xo is not None:
            e = self.eog_encoder(xo, xf)
            g = torch.sigmoid(self.eog_gate(torch.cat([h, e], -1)))
            self.gates["eog"] = g.detach()
            h = h + self._switch(g * self.eog_proj(e), eog_on)
        if self.use_tf:
            e = self.tf_encoder(xe)
            g = torch.sigmoid(self.tf_gate(torch.cat([h, e], -1)))
            self.gates["tf"] = g.detach()
            h = h + self._switch(g * self.tf_proj(e), tf_on)
        h = self.fusion(torch.cat([h, self.spectral_encoder(xs)], -1))
        if self.use_local:
            hm = h if mask is None else h.masked_fill(mask.unsqueeze(-1), 0.0)
            h = h + self.loc_pw(F.gelu(self.loc_dw(hm.transpose(1, 2)).transpose(1, 2)))
        z = self.encoder(self.positional_encoding(h), src_key_padding_mask=mask)
        logits = self.cls(z)
        if return_aux:
            aux = self.aux_transition(z).squeeze(-1) if self.use_aux else None
            return logits, aux
        return logits


def build_model(arm):
    a = ARMS[arm]
    m = AdvancedStudent(eog=a["eog"], tf=a["tf"], local=a["local"], aux=a["recipe"])
    n = sum(p.numel() for p in m.parameters())
    if n > PARAM_BUDGET:
        raise SystemExit(f"arm {arm}: {n:,} parameters exceeds the {PARAM_BUDGET:,} budget")
    return m


def is_new_param(name):
    return name.startswith(NEW_PREFIXES)


def warm_start_from(model, state_dict):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise SystemExit(f"Warm start: checkpoint has keys this model lacks: {unexpected[:5]}")
    bad = [k for k in missing if not is_new_param(k)]
    if bad:
        raise SystemExit(f"Warm start: backbone keys not loaded: {bad[:5]}")
    return list(missing)


def set_backbone_bn_eval(model, frozen):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm1d) and not is_new_param(name):
            mod.train(not frozen and model.training)


class EMA:
    """Exponential moving average of all floating-point parameters AND buffers
    (BatchNorm statistics included); integer buffers are copied."""

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.n = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def current_decay(self):
        """Warm-up: min(decay, (1 + n) / (10 + n)). Early on the average follows the
        live weights closely; otherwise, trained from scratch, the averaged model
        lags the live one for thousands of steps and early epochs score badly."""
        return min(self.decay, (1 + self.n) / (10 + self.n))

    @torch.no_grad()
    def update(self, model):
        d = self.current_decay()
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)
        self.n += 1

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


# ------------------------------------------------------------------ LOSSES ---
def distillation_loss(student_logits, teacher_logits, targets, T=3.0, alpha=0.5,
                      class_weights=None, smooth_eps=0.0, return_parts=False):
    """N4kd's loss (verbatim when smooth_eps == 0). With smooth_eps > 0 the HARD
    term uses boundary_targets() instead of one-hot; the class weight is the
    weight of the epoch's own expert label, exactly as F.cross_entropy weights."""
    C = student_logits.size(-1)
    y = targets.reshape(-1)
    valid = y != IGNORE_INDEX
    if valid.sum() == 0:
        z = student_logits.sum() * 0.0
        return (z, {"hard": z, "soft": z}) if return_parts else z
    s = student_logits.reshape(-1, C)[valid]
    t = teacher_logits.reshape(-1, C)[valid]
    yv = y[valid]
    if smooth_eps > 0:
        q = boundary_targets(targets, C, smooth_eps).reshape(-1, C)[valid]
        nll = -(q * F.log_softmax(s.float(), -1)).sum(-1)
        w = class_weights[yv] if class_weights is not None else torch.ones_like(nll)
        hard = (w * nll).sum() / w.sum()
    else:
        hard = F.cross_entropy(s, yv, weight=class_weights)
    s_log_p = F.log_softmax(s.float() / T, dim=-1)
    t_p = F.softmax(t.float() / T, dim=-1)
    soft = F.kl_div(s_log_p, t_p, reduction="batchmean") * (T ** 2)
    loss = alpha * hard + (1.0 - alpha) * soft
    if return_parts:
        return loss, {"hard": hard.detach(), "soft": soft.detach()}
    return loss


def aux_loss(aux_logits, y):
    tgt, m = transition_targets(y)
    if m.sum() == 0:
        return aux_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(aux_logits[m].float(), tgt[m])


# -------------------------------------------------------------------- DATA ---
def compute_feature_cache(df, root):
    return {row["rec"]: torch.from_numpy(recording_eog_features(
        torch.load(Path(root) / row["tensor_path"], map_location="cpu"))) for _, row in df.iterrows()}


def build_recording_caches(df, root, want_eog, want_quality):
    """
    ONE pass per recording at start-up, instead of repeating the same work for
    every training window (~2,700 windows per epoch, each recording sampled ~16
    times). Measured per 256-epoch window: quality mask 20 ms, EOG low-pass
    29 ms - together they would make data loading, not the GPU, the bottleneck.

    eog:     rec -> (n, 3000) float16 low-passed, scaled EOG (whole night filtered
             as one continuous signal, so window edges carry no filter artefact)
    feat:    rec -> (n, 8) unstandardised EOG features
    quality: rec -> (n,) bool, True = usable for training
    Memory: ~1.1 GB for the EOG cache over 168 recordings.
    """
    eog, feat, qual = ({} if want_eog else None), ({} if want_eog else None), ({} if want_quality else None)
    t0 = time.time()
    for k, (_, row) in enumerate(df.iterrows()):
        x = torch.load(Path(root) / row["tensor_path"], map_location="cpu")
        if want_quality:
            qual[row["rec"]] = quality_mask(x[:, EEG_CH].float())
        if want_eog:
            xn = x.float().numpy(); n, _, L = xn.shape
            lp = lowpass((xn[:, EOG_CH, :] * EOG_SCALE).reshape(-1)).reshape(n, L)
            eog[row["rec"]] = torch.from_numpy(lp.astype(np.float16))
            feat[row["rec"]] = torch.from_numpy(eog_epoch_features(lp, xn[:, EEG_CH, :] * EEG_SCALE))
    print(f"  per-recording caches ({'EOG ' if want_eog else ''}{'quality' if want_quality else ''}) "
          f"for {len(df)} recordings in {time.time() - t0:.0f}s")
    return eog, feat, qual


def fit_standardiser(cache, train_recs):
    allf = torch.cat([cache[r] for r in train_recs], 0)
    return allf.mean(0), allf.std(0).clamp_min(1e-6)


def preload_signals(df, root):
    """
    Read every recording ONCE into memory: the raw EEG channel and the 34
    spectral features. Measured on Kaggle (Stage A runs): with per-window
    torch.load of the whole 3-channel recording (~40 MB each, ~2,700 windows per
    epoch) an epoch took 53-55 s against N4kd's 21 s - data loading, not the
    GPU, was the bottleneck, and two parallel runs would fight over the same
    disk. ~13 MB of EEG per recording -> ~2.2 GB for 168 recordings. DataLoader
    workers are forked on Kaggle (Linux), so they share it without copying.
    """
    t0 = time.time()
    eeg, spec = {}, {}
    for _, row in df.iterrows():
        x = torch.load(Path(root) / row["tensor_path"], map_location="cpu")
        eeg[row["rec"]] = x[:, EEG_CH].float().contiguous()
        spec[row["rec"]] = torch.load(Path(root) / row["spectral"], map_location="cpu").float()
    gb = sum(v.numel() for v in eeg.values()) * 4 / 1e9
    print(f"  preloaded EEG + spectral for {len(df)} recordings ({gb:.1f} GB) in {time.time() - t0:.0f}s")
    return eeg, spec


class SleepDatasetAdv(Dataset):
    """Returns xe, xo, xs, xf, tl, y, q   (q = epoch usable for training)."""

    def __init__(self, df, root, window=256, overlap=0, teacher_dir=None, with_eog=False,
                 feat_cache=None, feat_mu=None, feat_sd=None, eog_cache=None, quality_cache=None,
                 eeg_cache=None, spec_cache=None):
        self.df = df.reset_index(drop=True); self.root = Path(root)
        self.teacher_dir = Path(teacher_dir) if teacher_dir else None
        self.window = window; self.stride = max(1, window - overlap)
        self.with_eog = with_eog
        self.feat_cache, self.mu, self.sd = feat_cache, feat_mu, feat_sd
        self.eog_cache, self.quality_cache = eog_cache, quality_cache
        self.eeg_cache, self.spec_cache = eeg_cache, spec_cache
        self.index = [(i, s) for i, row in self.df.iterrows()
                      for s in range(0, len(str(row["stage_sequence"]).split()), self.stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, st = self.index[k]; row = self.df.iloc[i]; en = st + self.window
        y = torch.tensor([LABEL_TO_IDX[s] for s in str(row["stage_sequence"]).split()[st:en]], dtype=torch.long)
        rec = row["rec"]
        if self.eeg_cache is not None:                     # preloaded: a slice, no disk read
            eeg_w = self.eeg_cache[rec][st:en]
            xs = self.spec_cache[rec][st:en]
            x = None
        else:
            x = torch.load(self.root / row["tensor_path"], map_location="cpu")[st:en]
            xs = torch.load(self.root / row["spectral"], map_location="cpu")[st:en]
            eeg_w = x[:, EEG_CH]
        n = min(len(y), eeg_w.shape[0], xs.shape[0])
        if self.teacher_dir is not None:
            d = np.load(self.teacher_dir / f"{row['rec']}.npz")
            tl = torch.from_numpy(d["logits"][st:en].astype(np.float32)); n = min(n, tl.shape[0]); tl = tl[:n]
        else:
            tl = torch.zeros(n, NUM_CLASSES)
        raw_eeg = eeg_w[:n].float()
        if self.quality_cache is not None:
            q = self.quality_cache[rec][st:st + n]
        else:
            q = torch.ones(n, dtype=torch.bool)       # QMASK off: every epoch counts (as in N4kd)
        xe = raw_eeg * EEG_SCALE
        if self.with_eog:
            if self.eog_cache is not None:
                xo = self.eog_cache[rec][st:st + n].float()
            else:                                      # fallback: filter this window only
                if x is None:
                    x = torch.load(self.root / row["tensor_path"], map_location="cpu")[st:en]
                raw = (x[:n, EOG_CH].float() * EOG_SCALE).numpy()
                xo = torch.from_numpy(lowpass(raw.reshape(-1)).reshape(raw.shape).astype(np.float32))
            xf = ((self.feat_cache[rec][st:st + n] - self.mu) / self.sd).clamp(-6, 6)
        else:
            xo = torch.zeros(n, 1); xf = torch.zeros(n, N_EOG_FEATURES)
        return xe, xo, xs[:n].float(), xf, tl, y[:n], q


def collate(batch):
    L, B = max(b[0].shape[0] for b in batch), len(batch)
    xe = torch.zeros(B, L, batch[0][0].shape[1]); xo = torch.zeros(B, L, batch[0][1].shape[1])
    xs = torch.zeros(B, L, batch[0][2].shape[1]); xf = torch.zeros(B, L, batch[0][3].shape[1])
    tl = torch.zeros(B, L, NUM_CLASSES)
    y = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    q = torch.zeros(B, L, dtype=torch.bool)
    mask = torch.ones(B, L, dtype=torch.bool)
    for i, (a, o, sp, f, t, c, qq) in enumerate(batch):
        n = a.shape[0]
        xe[i, :n], xo[i, :n], xs[i, :n], xf[i, :n], tl[i, :n] = a, o, sp, f, t
        y[i, :n], q[i, :n], mask[i, :n] = c, qq, False
    return xe, xo, xs, xf, tl, y, q, mask


def augment(xe, xs, xo, gen):
    """Per-sequence gain on EEG (spectral features moved with it) and on EOG
    (independently), plus small Gaussian noise. Returns new tensors."""
    B = xe.size(0)
    lo, hi = GAIN_RANGE
    g = (lo + (hi - lo) * torch.rand(B, generator=gen)).to(xe.device)
    ge = g.view(B, 1, 1)
    xe = xe * ge + NOISE_STD * torch.randn(xe.shape, generator=gen).to(xe.device)
    xs = gain_spectral(xs, g)
    if xo.size(-1) > 1:
        go = (lo + (hi - lo) * torch.rand(B, generator=gen)).to(xo.device).view(B, 1, 1)
        xo = xo * go + NOISE_STD * torch.randn(xo.shape, generator=gen).to(xo.device)
    return xe, xs, xo


# -------------------------------------------------------------- EVALUATION ---
@torch.no_grad()
def evaluate(model, loader, device, amp, eog=True, tf=True):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_fscore_support
    model.eval(); yt, yp = [], []
    for xe, xo, xs, xf, _tl, y, _q, mask in loader:
        xe, xs, mask = xe.to(device), xs.to(device), mask.to(device)
        B = xe.size(0)
        kw = {}
        if model.use_eog:
            kw.update(xo=xo.to(device), xf=xf.to(device), eog_on=torch.full((B,), float(eog), device=device))
        if model.use_tf:
            kw["tf_on"] = torch.full((B,), float(tf), device=device)
        with torch.amp.autocast(device_type="cuda", enabled=amp):
            lg = model(xe, xs, mask, **kw)
        v = ~mask
        yt.append(y.to(device)[v].cpu().numpy()); yp.append(lg.argmax(-1)[v].cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=LAB, zero_division=0)
    return {"accuracy": float(accuracy_score(yt, yp)), "kappa": float(cohen_kappa_score(yt, yp, labels=LAB)),
            "macro_f1": float(f1_score(yt, yp, average="macro", labels=LAB, zero_division=0)),
            "per_class_f1": {STAGES[i]: float(f1[i]) for i in LAB},
            "per_class_precision": {STAGES[i]: float(p[i]) for i in LAB},
            "per_class_recall": {STAGES[i]: float(r[i]) for i in LAB}}


def make_lr_lambdas(steps, epochs, warm_start, freeze_epochs, eeg_lr_mult):
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
    ap.add_argument("--arm"); ap.add_argument("--seed", type=int); ap.add_argument("--cv-fold", type=int)
    ap.add_argument("--exclude-misaligned", action="store_true")
    ap.add_argument("--corrected-labels", action="store_true")
    a, _ = ap.parse_known_args(argv)
    return a


def main():
    global ARM, SEED, CV_FOLD, ALPHA, WARM_START, EXCLUDE_MISALIGNED, USE_CORRECTED_LABELS
    a = parse_overrides(sys.argv[1:])
    if a.exclude_misaligned: EXCLUDE_MISALIGNED = True
    if a.corrected_labels: USE_CORRECTED_LABELS = True
    if a.arm: ARM = a.arm
    if a.seed is not None: SEED = a.seed
    if a.cv_fold is not None:
        CV_FOLD, ALPHA, WARM_START = a.cv_fold, 1.0, False     # protocol A, the only legal CV setting
    check_config(CV_FOLD, ALPHA, WARM_START, ARM)
    comp = ARMS[ARM]
    recipe = comp["recipe"]
    splits = build_splits(CV_FOLD)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0); sm = f"sm_{cap[0]}{cap[1]}"
        sup = torch.cuda.get_arch_list()
        if sup and sm not in sup:
            raise SystemExit(f"UNSUPPORTED GPU ({sm}); supports {sup}. Kaggle: Accelerator -> 'GPU T4 x2'.")
        print(f"GPU: {torch.cuda.get_device_name(0)} ({sm})")
    else:
        print("WARNING: no GPU - this will be very slow.")
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    ############
    # index = find_file("processed_sleepedf_mc/**/index.csv")
    # root = index.parent
    # df = pd.read_csv(index)
    # ch = str(df["channels"].iloc[0]).split("|")
    # if ch[EEG_CH] != "EEG Fpz-Cz" or ch[EOG_CH] != "EOG horizontal":
    #     raise SystemExit(f"Unexpected channel order {ch}")
    # df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    # df["subject"] = df["rec"].str[:5]
    # df["tensor_path"] = [
    #     f"tensors/{r}.pt" for r in df["rec"]
    #     ]
    # # df["tensor_path"] = ["processed_sleepedf_mc/**/tensors/%s.pt" % r for r in df["rec"]]
    # # spec_root = find_file("processed_sleepedf/**/spectral").parent.parent
    # # df["spectral"] = [str(spec_root / "processed_sleepedf/**/spectral" / f"{r}_spectral.pt") for r in df["rec"]]
    # spectral_dir = find_file("processed_sleepedf/spectral/spectral")
    # # Locate index.csv
    index = find_file("processed_sleepedf_mc/**/index.csv")

    # index.csv and tensors/ are in the same directory
    root = index.parent

    df = pd.read_csv(index)

    ch = str(df["channels"].iloc[0]).split("|")

    if ch[EEG_CH] != "EEG Fpz-Cz" or ch[EOG_CH] != "EOG horizontal":
        raise SystemExit(f"Unexpected channel order {ch}")

    df["rec"] = [
        str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3]
        for p in df["tensor_path"]
    ]

    df["subject"] = df["rec"].str[:5]

    # Tensor paths relative to root
    df["tensor_path"] = [
        f"tensors/{r}.pt" for r in df["rec"]
    ]

    # Locate the actual inner spectral directory
    spectral_dir = find_file(
        "processed_sleepedf/spectral/spectral"
    )
    ####################
    df["spectral"] = [
        str(spectral_dir / f"{r}_spectral.pt")
        for r in df["rec"]
    ]
    tr = df[df["subject"].isin(splits["train"])]
    va = df[df["subject"].isin(splits["val"])]
    assert set(tr["subject"]).isdisjoint(va["subject"]) and set(tr["subject"]).isdisjoint(splits["test"])
    assert set(va["subject"]).isdisjoint(splits["test"])
    if USE_CORRECTED_LABELS:
        fixed = json.loads(find_file(CORRECTED_LABELS_FILE).read_text())
        need = set(tr["rec"]) | set(va["rec"])
        missing = sorted(need - set(fixed))
        if missing:
            raise SystemExit(f"{CORRECTED_LABELS_FILE} lacks {len(missing)} recordings, first {missing[:3]}")
        tr = tr.assign(stage_sequence=[fixed[r] for r in tr["rec"]])
        va = va.assign(stage_sequence=[fixed[r] for r in va["rec"]])
        n_q = sum(seq.split().count("?") for seq in list(tr["stage_sequence"]) + list(va["stage_sequence"]))
        print(f"  CORRECTED LABELS: train+val labels replaced from {CORRECTED_LABELS_FILE} "
              f"({n_q} never-scored epochs ignored). Not comparable with stored-label runs.")
    dropped = []
    if EXCLUDE_MISALIGNED:
        dropped = sorted(set(MISALIGNED) & (set(tr["rec"]) | set(va["rec"])))
        tr = tr[~tr["rec"].isin(MISALIGNED)]; va = va[~va["rec"].isin(MISALIGNED)]
        print(f"  EXCLUDE_MISALIGNED: dropped {len(dropped)} recordings with time-shifted labels: {dropped}")

    name = (f"{EXPERIMENT}_{ARM}_s{SEED}" + (f"_cv{CV_FOLD}" if CV_FOLD is not None else "")
            + ("_clean" if EXCLUDE_MISALIGNED else "") + ("_fixlabels" if USE_CORRECTED_LABELS else ""))
    out = Path(OUT_ROOT) / f"student_{name}"; out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\nEXPERIMENT {name}   components {comp}\n{'='*72}")
    print(f"  cv_fold {CV_FOLD} | alpha {ALPHA} | warm start {WARM_START} | test {len(splits['test'])} subj NEVER LOADED")

    feat_cache = mu = sd = None
    eog_cache, feat_cache, quality_cache = build_recording_caches(
        pd.concat([tr, va]), root, want_eog=comp["eog"], want_quality=recipe)
    if comp["eog"]:
        mu, sd = fit_standardiser(feat_cache, list(tr["rec"]))       # TRAIN recordings only
    tdir = None
    if ALPHA < 1.0:
        tdir = find_file(f"teacher_logits_{TEACHER_TAG}").resolve()
        miss = sorted(set(tr["rec"]) - {p.stem for p in tdir.glob("*.npz")})
        if miss:
            raise SystemExit(f"{len(miss)} training recordings have no teacher logits, first {miss[:3]}")

    eeg_cache, spec_cache = preload_signals(pd.concat([tr, va]), root)
    dkw = dict(with_eog=comp["eog"], feat_cache=feat_cache, feat_mu=mu, feat_sd=sd,
               eog_cache=eog_cache, quality_cache=quality_cache,
               eeg_cache=eeg_cache, spec_cache=spec_cache)
    ds_tr = SleepDatasetAdv(tr, root, WINDOW_SIZE, OVERLAP, teacher_dir=tdir, **dkw)
    ds_va = SleepDatasetAdv(va, root, WINDOW_SIZE, 0, **dkw)
    lkw = dict(collate_fn=collate, num_workers=NUM_WORKERS, pin_memory=amp)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, **lkw)
    dl_va = DataLoader(ds_va, batch_size=MICRO_BATCH, shuffle=False, **lkw)
    steps = len(dl_tr)

    cw = None
    if CLASS_WEIGHT_POWER > 0:
        c = dict.fromkeys(STAGES, 0)
        for s in tr["stage_sequence"]:
            for x in str(s).split():
                if x in c:                                   # "?" never counts
                    c[x] += 1
        counts = torch.tensor([float(c[s]) for s in STAGES])
        cw = (1.0 / counts) ** CLASS_WEIGHT_POWER
        cw = (cw / cw.sum() * NUM_CLASSES).to(device)

    model = build_model(ARM).to(device)
    if WARM_START:
        ck = torch.load(find_file(WARM_START_FILE), map_location=device, weights_only=False)
        if ck.get("eeg_scale") != EEG_SCALE:
            raise SystemExit(f"Warm start: checkpoint eeg_scale {ck.get('eeg_scale')} != {EEG_SCALE}")
        fresh = warm_start_from(model, ck["model"])
        print(f"  warm start from N4kd: backbone loaded, {len(fresh)} new tensors fresh")
    npar = sum(p.numel() for p in model.parameters())
    nnew = sum(p.numel() for n, p in model.named_parameters() if is_new_param(n))
    print(f"  parameters {npar:,} (new {nnew:,}; N4kd 139,606; budget {PARAM_BUDGET:,})")

    groups = [{"params": [p for n, p in model.named_parameters() if not is_new_param(n)]}]
    new_params = [p for n, p in model.named_parameters() if is_new_param(n)]
    lambdas = make_lr_lambdas(steps, EPOCHS, WARM_START, FREEZE_EPOCHS, EEG_LR_MULT)
    if new_params:
        groups.append({"params": new_params})
    else:
        lambdas = lambdas[:1]
    opt = torch.optim.AdamW(groups, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)
    scaler = torch.amp.GradScaler(enabled=amp)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    ema = EMA(model) if recipe else None

    config = {"experiment": name, "arm": ARM, "components": comp, "seed": SEED, "cv_fold": CV_FOLD,
              "alpha": ALPHA, "T": T, "warm_start": WARM_START, "eeg_scale": EEG_SCALE, "eog_scale": EOG_SCALE,
              "smooth_eps": SMOOTH_EPS if recipe else 0.0, "aux_weight": AUX_WEIGHT if recipe else 0.0,
              "gain_range": GAIN_RANGE if recipe else None, "ema_decay": EMA_DECAY if recipe else None,
              "modality_dropout": MODALITY_DROPOUT, "splits": splits,
              "exclude_misaligned": EXCLUDE_MISALIGNED, "excluded_recordings": dropped,
              "corrected_labels": USE_CORRECTED_LABELS,
              "schedule_shape": {"epochs": EPOCHS, "steps": steps}, "n_parameters": npar}

    def eval_model():
        if ema is None:
            return model
        m2 = build_model(ARM).to(device); m2.load_state_dict(ema.state_dict()); return m2

    start, best, since = 0, -1.0, 0
    last = out / "student_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        keys = ("arm", "seed", "cv_fold", "alpha", "warm_start", "schedule_shape", "exclude_misaligned",
                "corrected_labels")
        if {k: ck["config"].get(k) for k in keys} != {k: config[k] for k in keys}:
            raise SystemExit(f"Cannot resume: config changed. Delete {out}.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema") is not None:
            ema.shadow = ck["ema"]
        start, best, since = ck["epoch"] + 1, ck["best_macro_f1"], ck.get("since_best", 0)
        print(f"  RESUMED at epoch {start} (best {best:.4f})")

    t0 = time.time()
    for ep in range(start, EPOCHS):
        model.train()
        frozen = WARM_START and ep < FREEZE_EPOCHS
        set_backbone_bn_eval(model, frozen)
        tot, lr_sum = 0.0, [0.0] * len(opt.param_groups)
        for xe, xo, xs, xf, tl, y, q, mask in dl_tr:
            for gi, g in enumerate(opt.param_groups):
                lr_sum[gi] += g["lr"]
            xe, xs, tl = xe.to(device), xs.to(device), tl.to(device)
            y, q, mask = y.to(device), q.to(device), mask.to(device)
            xo, xf = xo.to(device), xf.to(device)
            if recipe:
                y = torch.where(q, y, torch.full_like(y, IGNORE_INDEX))       # QMASK: loss only
                xe, xs, xo = augment(xe, xs, xo, gen)
            B = xe.size(0)
            eog_on = (torch.rand(B, generator=gen) >= MODALITY_DROPOUT).float().to(device)
            tf_on = (torch.rand(B, generator=gen) >= MODALITY_DROPOUT).float().to(device)
            nv = (y != IGNORE_INDEX).sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            for c0 in range(0, B, MICRO_BATCH):
                sl = slice(c0, c0 + MICRO_BATCH)
                nvc = (y[sl] != IGNORE_INDEX).sum()
                if nvc == 0:
                    continue
                kw = {}
                if comp["eog"]:
                    kw.update(xo=xo[sl], xf=xf[sl], eog_on=eog_on[sl])
                if comp["tf"]:
                    kw["tf_on"] = tf_on[sl]
                with torch.amp.autocast(device_type="cuda", enabled=amp):
                    lg, ax = model(xe[sl], xs[sl], mask[sl], return_aux=True, **kw)
                    loss = distillation_loss(lg, tl[sl], y[sl], T=T, alpha=ALPHA, class_weights=cw,
                                             smooth_eps=SMOOTH_EPS if recipe else 0.0)
                    if recipe and ax is not None:
                        loss = loss + AUX_WEIGHT * aux_loss(ax, y[sl])
                    loss = loss * (nvc / nv).float()
                scaler.scale(loss).backward()
                tot += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if ema is not None:
                ema.update(model)

        em = eval_model()
        m = evaluate(em, dl_va, device, amp)
        rec = {"experiment": name, "epoch": ep, "frozen_backbone": frozen, "loss": tot / steps,
               "lr_backbone": lr_sum[0] / steps, "lr_new": lr_sum[1] / steps if len(lr_sum) > 1 else None,
               "elapsed_min": round((time.time() - t0) / 60, 2), **m}
        if em.gates:
            rec["mean_gate"] = {k: float(v.float().mean()) for k, v in em.gates.items()}
        with open(out / "training_metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        is_best = m["macro_f1"] > best
        best, since = (m["macro_f1"], 0) if is_best else (best, since + 1)
        state = {"epoch": ep, "model": model.state_dict(), "ema": ema.state_dict() if ema else None,
                 "eval_weights": em.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "best_macro_f1": best,
                 "since_best": since, "config": config,
                 "eog_feature_standardiser": None if mu is None else {"mu": mu.tolist(), "sd": sd.tolist()}}
        torch.save(state, last)
        if is_best:
            torch.save(state, out / "student_best.pt")
        pc = " ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())
        gs = (" | gate " + " ".join(f"{k}={v:.2f}" for k, v in rec["mean_gate"].items())) if "mean_gate" in rec else ""
        print(f"ep {ep:3d}{' F' if frozen else '  '}| loss {rec['loss']:.4f} | mF1 {m['macro_f1']:.4f} | "
              f"k {m['kappa']:.4f} | acc {m['accuracy']:.4f} | {pc}{gs}{'  <- best' if is_best else ''}", flush=True)
        if EARLY_STOP_PATIENCE and since >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stop: no gain for {EARLY_STOP_PATIENCE} epochs."); break

    ck = torch.load(out / "student_best.pt", map_location=device, weights_only=False)
    final = build_model(ARM).to(device); final.load_state_dict(ck["eval_weights"])
    m = evaluate(final, dl_va, device, amp)
    result = {"experiment": name, "arm": ARM, "components": comp, "seed": SEED, "cv_fold": CV_FOLD,
              "exclude_misaligned": EXCLUDE_MISALIGNED, "excluded_recordings": dropped,
              "corrected_labels": USE_CORRECTED_LABELS,
              "alpha": ALPHA, "warm_start": WARM_START, "best_epoch": ck["epoch"], "n_parameters": npar, "val": m}
    print(f"\n{'='*72}\nRESULT  {name}  (validation)\n{'='*72}")
    print(f"  acc {m['accuracy']:.4f} | kappa {m['kappa']:.4f} | macro-F1 {m['macro_f1']:.4f}")
    print("  per-class F1  " + " ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items()))
    for branch, kw in (("eog", dict(eog=False)), ("tf", dict(tf=False))):
        if comp[branch]:
            mo = evaluate(final, dl_va, device, amp, **kw)
            result[f"val_{branch}_off"] = mo
            print(f"\n  {branch.upper()} switched off (same weights): kappa {mo['kappa']:.4f} -> {m['kappa']:.4f}")
            for s in STAGES:
                print(f"    {s:<4} F1 {mo['per_class_f1'][s]:.3f} -> {m['per_class_f1'][s]:.3f} "
                      f"({m['per_class_f1'][s]-mo['per_class_f1'][s]:+.3f})")
    if CV_FOLD is None:
        print(f"\n  vs N4kd on the same validation split: kappa {m['kappa']:.4f} vs {N4KD_VAL['kappa']:.4f}; "
              + "; ".join(f"{s} {m['per_class_f1'][s]:.3f} vs {N4KD_VAL['per_class_f1'][s]:.3f}" for s in ("N1", "REM")))
    else:
        print("\n  One fold decides nothing. Run all 5 folds for this arm AND for 'base', then summarise_cv.py.")
    (out / f"result_{name}.json").write_text(json.dumps(result, indent=1))
    print("  Test split was NEVER loaded.")


if __name__ == "__main__":
    main()
