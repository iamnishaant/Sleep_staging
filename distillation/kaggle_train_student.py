"""
================================================================================
KAGGLE NOTEBOOK - STUDENT DISTILLATION (runs on a SECOND account, in parallel)
================================================================================
It only READS a frozen teacher checkpoint, writes to its own directories, and
never loads the test split. It cannot affect any teacher run it is run beside.

CURRENTLY CONFIGURED FOR THE E1b RUNG (TEACHER_TAG = "E1b").
The E0 rung is already complete; its results are in
distillation/results/students/ and distillation/results/eval_students.json.

SETUP
-----
1. Attach your existing sleepedf dataset (processed_sleepedf/...).
2. Create a SECOND small dataset containing ONLY the teacher you are distilling
   from - for this run, the E1b checkpoint:
       teacher_best.pt      (~8 MB, from distillation/results/E1b/)
   Attach it too. The script finds it by filename anywhere under /kaggle/input.
   Attach ONE teacher only: every rung saves a file with this same name, and
   find_file() takes the first match. identify_teacher() cross-checks the
   checkpoint's own metadata against TEACHER_TAG and hard-fails on a mismatch,
   so a wrong attachment stops the run instead of mislabelling it.
3. Accelerator -> GPU T4 x2 (single GPU is used; T4 required, P100 has no
   kernels in Kaggle's PyTorch build).
4. Run.

WHAT IT DOES
------------
  Step 1  Compute and cache the teacher's soft targets ONCE (~2 min on GPU).
          Running the teacher inside the training loop would recompute
          identical outputs every epoch.
  Step 2  Train the DISTILLED student   (alpha=0.5, T=3.0)
  Step 3  Train the BASELINE student    (alpha=1.0 -> soft term disabled)
          SKIPPED unless RUN_BASELINE=True. See the config note: alpha=1.0 never
          reads the teacher, so with SEED=42 it reproduces student_baseline_E0
          bit-for-bit. It is the teacher-independent arm - trained once, reused.

Steps 2 and 3 use the SAME trainer, architecture, data, splits, schedule and
seed. The only difference is alpha. That is what makes the comparison isolate
the training signal rather than a pile of confounds - a separate baseline
script could not guarantee it, and it is also why the stored baseline stays
valid across teachers.

THE T**2 TERM IN THE KD LOSS IS MANDATORY. Softening by T shrinks the soft
gradient by 1/T^2; without the correction training still runs and converges but
barely distils, and nothing in the loss curve reveals it. Verified by
distillation/test_kd_loss.py (13/13).

EXPECT ~3 h with RUN_BASELINE=False (~5 h if True). Resumes from /kaggle/working
if the session dies.
================================================================================
"""

import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# ==============================================================================
# CONFIG
# ==============================================================================
TEACHER_TAG    = "E1b"       # must match the attached checkpoint's experiment tag
#                ^^^^^ The E0 rung is DONE - student_distilled_E0 and
#                student_baseline_E0 are already in distillation/results/students/.
#                To reproduce that rung instead, set this back to "E0", set
#                RUN_BASELINE = True, and attach the E0 teacher_best.pt.
#                identify_teacher() below hard-fails on a mismatch, so a wrong
#                tag stops the run rather than mislabelling it.
T              = 3.0         # KD temperature
ALPHA_DISTILL  = 0.5         # hard/soft mix for the distilled run

# The baseline (alpha=1.0) does not read teacher logits at all: train_one passes
# `cache if distilling else None`, and KDSleepDataset builds its window index from
# sequence length and stride only - never from the cache. With SEED fixed at 42,
# re-running it for a second teacher reproduces student_baseline_E0 bit-for-bit
# and costs ~2.5 GPU-hours to learn nothing. It is the teacher-independent arm of
# the comparison, so it is trained ONCE and reused.
#
# Set True only if the student architecture, CLASS_WEIGHT_POWER, USE_SPECTRAL,
# EPOCHS or SEED change - that invalidates the stored baseline and it must be
# retrained for the comparison to stay honest.
RUN_BASELINE   = False
BASELINE_REF_VAL_MF1  = 0.6881   # student_baseline_E0, validation macro-F1
BASELINE_REF_TEST_KAPPA = 0.6449  # student_baseline_E0, held-out test kappa
# sqrt-inverse-frequency class weighting on the HARD term only.
#
# This was 0.0 ("no rebalancing, honest per-class view") in the first attempt and
# the student collapsed to the majority classes: N3 and REM F1 exactly 0.000,
# baseline never predicting N1 at all, loss flat at ~1.41 for 32 epochs.
#
# The cause was an asymmetry with the teacher, not a bug. The teacher trains with
# focal loss AND inverse-frequency alpha (N1 1.37, N3 1.77) - that is why it
# over-predicts N1 at 2.4x. The student had NO rebalancing, less capacity, and no
# spectral input. With N2 at 41% of val epochs, plain CE on an under-capacity
# model just predicts the majority.
#
# 0.5 is the same sqrt-inverse choice validated on the teacher's prior
# correction (macro-F1 0.6335 vs a test-tuned oracle of 0.6342). It applies to
# the hard term only - the soft/KD term is untouched - so the distilled-vs-
# baseline comparison stays clean: both runs get identical weighting and alpha
# remains the only difference between them.
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

# student architecture: 121,099 params with USE_SPECTRAL=True -> 5.36x smaller
# than the teacher (110,475 / 5.88x if USE_SPECTRAL=False)
EMBED_DIM      = 64
HEADS          = 2
LAYERS         = 2
N_TOKENS       = 1           # 8 applies the E4 multi-token treatment (+65 params)
DROPOUT        = 0.2

# Give the student the 34-dim spectral vector, as the teacher has.
#
# Attempt 1 (no spectral, no class weights): N3 and REM F1 exactly 0.000.
# Attempt 2 (no spectral, sqrt class weights): N3 and REM STILL exactly 0.000 at
# epoch 11, hard-CE 1.627 against a chance level of ln(5)=1.609 - i.e. not
# learning at all. Class balance was never the binding constraint.
#
# The teacher reaches N3 F1 = 0.406 at EPOCH 0 on identical data. The difference
# is the input: N3 is DEFINED by delta-band power, which is directly one of the
# 34 precomputed spectral features. Without them the student must rediscover
# delta power from a raw 3000-sample trace through 2 conv branches and a global
# average pool - a strictly harder problem than its teacher solves, while being
# 6x smaller.
#
# Cost: Linear(34->64) plus concat fusion, ~10.6K params, so compression goes
# 5.88x -> 5.36x. Set False to reproduce the temporal-only ablation.
USE_SPECTRAL   = True

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


# ----------------------------------------------------------------- DISCOVERY --
def find_file(pattern, base="/kaggle/input"):
    hits = sorted(Path(base).glob(f"**/{pattern}"))
    if not hits:
        print(f"Could not find '{pattern}'. Contents of {base}:")
        for p in sorted(Path(base).rglob("*"))[:50]:
            print("   ", p)
        raise SystemExit(f"Missing required file: {pattern}")
    if len(hits) > 1:
        print(f"  WARNING: {len(hits)} files match '{pattern}'; using the first:")
        for h in hits:
            print("     ", h)
    return hits[0]


def identify_teacher(ckpt_path):
    """
    Work out WHICH teacher a checkpoint actually is, from its own metadata.

    Every rung of the ladder saves to a file called 'teacher_best.pt'. If the
    wrong one is attached, or both are, the run would train against a different
    teacher than the output directories claim - and the mislabelling would only
    surface much later, in the RQ3 comparison, as an inexplicable result.

      E0   : saved by kaggle_train_teacher.py, which predates the 'experiment'
             key. Identified by its absence. epoch 118, val macro-F1 0.5862.
      E1a+ : saved by kaggle_train_improved.py, which stores 'experiment'.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    exp = ck.get("experiment") or "E0"
    return exp, ck.get("epoch"), ck.get("best_macro_f1"), ck.get("config")


# -------------------------------------------------------------- TEACHER MODEL --
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Linear(ch, max(1, ch // r)); self.fc2 = nn.Linear(max(1, ch // r), ch)
    def forward(self, x):
        s = x.mean(-1); s = F.relu(self.fc1(s)); s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)

class AtrousPyramid(nn.Module):
    def __init__(self, inc=1, hid=64, dil=(1, 2, 4, 8)):
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
    def __init__(self, embed=128, hid=64, dil=(1, 2, 4, 8), n_tokens=1):
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
        if self.n_tokens > 1:
            f = (f * torch.softmax(self.token_attn(f), 1)).sum(1)
        else:
            f = f.squeeze(1)
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

class SleepTransformer(nn.Module):
    def __init__(self, embed=128, heads=4, layers=3, dropout=0.2, max_len=512):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed, max_len)
        l = nn.TransformerEncoderLayer(d_model=embed, nhead=heads,
                                       dim_feedforward=embed*4, dropout=dropout,
                                       batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(l, layers)
        self.cls = nn.Linear(embed, NUM_CLASSES)
    def forward(self, x, mask=None):
        return self.cls(self.encoder(self.positional_encoding(x), src_key_padding_mask=mask))

class FusedSleepStagingModel(nn.Module):
    """Teacher. Must match the E0 checkpoint exactly."""
    def __init__(self, spec_dim=34, embed=128, heads=4, layers=3, dropout=0.2):
        super().__init__()
        self.temporal_encoder = EpochEncoder(embed)
        self.spectral_encoder = nn.Linear(spec_dim, embed)
        self.fusion = nn.Sequential(nn.Linear(embed*2, embed), nn.LayerNorm(embed), nn.GELU())
        self.context = SleepTransformer(embed, heads, layers, dropout)
    def forward(self, xt, xs, mask=None):
        h = torch.cat([self.temporal_encoder(xt), self.spectral_encoder(xs)], -1)
        return self.context(self.fusion(h), mask)


# -------------------------------------------------------------- STUDENT MODEL --
class StudentSleepStagingModel(nn.Module):
    """
    2 dilation branches (teacher 4), embed 64 (128), 2 heads (4), 2 layers (3).
    use_spectral mirrors the teacher's second stream: Linear(34->embed) then
    concat+fuse. Without it the student cannot learn N3 or REM at all.
    """
    def __init__(self, embed=64, heads=2, layers=2, dropout=0.2,
                 hid=64, dil=(1, 4), n_tokens=1, max_len=512,
                 use_spectral=True, spec_dim=34):
        super().__init__()
        self.use_spectral = use_spectral
        self.temporal_encoder = EpochEncoder(embed, hid, dil, n_tokens)
        if use_spectral:
            self.spectral_encoder = nn.Linear(spec_dim, embed)
            self.fusion = nn.Sequential(nn.Linear(embed*2, embed),
                                        nn.LayerNorm(embed), nn.GELU())
        self.positional_encoding = PositionalEncoding(embed, max_len)
        l = nn.TransformerEncoderLayer(d_model=embed, nhead=heads,
                                       dim_feedforward=embed*4, dropout=dropout,
                                       batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(l, layers)
        self.cls = nn.Linear(embed, NUM_CLASSES)
    def forward(self, xt, xs=None, mask=None):
        h = self.temporal_encoder(xt)
        if self.use_spectral:
            if xs is None:
                raise ValueError("use_spectral=True but no spectral input given")
            h = self.fusion(torch.cat([h, self.spectral_encoder(xs)], -1))
        h = self.positional_encoding(h)
        return self.cls(self.encoder(h, src_key_padding_mask=mask))


# ------------------------------------------------------------------- KD LOSS --
def distillation_loss(s_logits, t_logits, targets, T=3.0, alpha=0.5,
                      class_weights=None):
    C = s_logits.size(-1)
    s = s_logits.reshape(-1, C); t = t_logits.reshape(-1, C); y = targets.reshape(-1)
    m = y != IGNORE_INDEX
    s, t, y = s[m], t[m], y[m]
    if s.numel() == 0:
        z = s_logits.sum() * 0.0
        return z, z, z
    hard = F.cross_entropy(s, y, weight=class_weights)
    # float() guards fp16 autocast: the soft targets' information lives in the
    # small off-argmax probabilities, which half precision would blur.
    s_log_p = F.log_softmax(s.float() / T, -1)
    t_p = F.softmax(t.float() / T, -1)
    soft = F.kl_div(s_log_p, t_p, reduction="batchmean") * (T ** 2)   # T**2 MANDATORY
    return alpha * hard + (1 - alpha) * soft, hard.detach(), soft.detach()


# ----------------------------------------------------------------- DATA --------
class KDSleepDataset(Dataset):
    def __init__(self, df, root, tdir=None, window=256, overlap=0):
        self.df = df.reset_index(drop=True); self.root = Path(root)
        self.tdir = Path(tdir) if tdir else None
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
        if self.tdir is not None:
            tl = torch.from_numpy(np.load(self.tdir / f"{row['rec']}.npz")["logits"][st:en].astype(np.float32))
            n = min(n, tl.shape[0]); tl = tl[:n]
        else:
            tl = torch.zeros(n, NUM_CLASSES)
        return xt[:n].float(), xs[:n].float(), tl, y[:n]

def collate(batch):
    L, B = max(b[0].shape[0] for b in batch), len(batch)
    xt = torch.zeros(B, L, batch[0][0].shape[1])
    xs = torch.zeros(B, L, batch[0][1].shape[1])
    tl = torch.zeros(B, L, NUM_CLASSES)
    y = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    for i, (a, sp, t, c) in enumerate(batch):
        n = a.shape[0]
        xt[i, :n], xs[i, :n], tl[i, :n], y[i, :n], mask[i, :n] = a, sp, t, c, False
    return xt, xs, tl, y, mask


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval(); yt, yp = [], []
    for xt, xs, _, y, mask in loader:
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


# =============================================================== STEP 1: CACHE =
def cache_teacher(root, df_train, ckpt_path, device, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    todo = [r for r in df_train["rec"] if not (cache / f"{r}.npz").exists()]
    if not todo:
        print(f"  teacher logits already cached ({len(list(cache.glob('*.npz')))} recordings)")
        return
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    tm = FusedSleepStagingModel()
    missing, unexpected = tm.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"Teacher checkpoint mismatch:\n missing {sorted(missing)}\n"
                         f" unexpected {sorted(unexpected)}")
    tm.to(device).eval()
    print(f"  teacher loaded: {sum(p.numel() for p in tm.parameters()):,} params "
          f"(expect 649,229 trainable)")
    t0 = time.time()
    with torch.no_grad():
        for i, rec in enumerate(todo):
            row = df_train[df_train["rec"] == rec].iloc[0]
            xt = torch.load(root / row["tensor_path"], map_location="cpu").float()
            xs = torch.load(root / row["spectral"], map_location="cpu").float()
            y = np.array([STAGE_TO_IDX[s] for s in str(row["stage_sequence"]).split()], dtype=np.int64)
            n = min(xt.shape[0], xs.shape[0], len(y))
            out = np.empty((n, NUM_CLASSES), dtype=np.float32)
            for st in range(0, n, WINDOW_SIZE):
                en = min(st + WINDOW_SIZE, n)
                out[st:en] = tm(xt[st:en].unsqueeze(0).to(device),
                                xs[st:en].unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            np.savez_compressed(cache / f"{rec}.npz",
                                logits=out.astype(np.float16), labels=y[:n].astype(np.int8))
            if (i+1) % 25 == 0 or i == len(todo)-1:
                print(f"    [{i+1}/{len(todo)}] {(time.time()-t0)/60:.1f} min", flush=True)
    del tm
    torch.cuda.empty_cache()
    print(f"  cached {len(list(cache.glob('*.npz')))} recordings in {(time.time()-t0)/60:.1f} min")


# =============================================================== STEP 2: TRAIN =
def train_one(tag, alpha, root, tr, va, cache, device, amp):
    out = Path(OUT_ROOT) / tag
    out.mkdir(parents=True, exist_ok=True)
    distilling = alpha < 1.0
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*72}\n{tag}   alpha={alpha}  "
          f"({'DISTILLING (T=%.1f)' % T if distilling else 'HARD LABELS ONLY - baseline'})"
          f"\n{'='*72}")

    ds_tr = KDSleepDataset(tr, root, cache if distilling else None, WINDOW_SIZE, OVERLAP)
    ds_va = KDSleepDataset(va, root, None, WINDOW_SIZE, 0)
    kw = dict(collate_fn=collate, num_workers=NUM_WORKERS, pin_memory=amp)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, **kw)
    dl_va = DataLoader(ds_va, batch_size=MICRO_BATCH, shuffle=False, **kw)
    steps = len(dl_tr)
    print(f"  train windows {len(ds_tr):,} ({steps} steps/epoch) | val {len(ds_va):,}")

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
                                     n_tokens=N_TOKENS,
                                     use_spectral=USE_SPECTRAL).to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  student parameters: {npar:,}  (teacher 649,229 -> {649229/npar:.2f}x)")

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
        shape = {"epochs": EPOCHS, "steps": steps}
        # A stale checkpoint from a previous session silently ruined the first
        # attempt: the distilled run resumed at epoch 41 with since_best already
        # >= 24, trained for ONE epoch, and early-stopped after 2.8 minutes.
        # Refuse to resume if the training signal changed, and say so loudly.
        prev_cfg = (ck.get("alpha"), ck.get("T"), ck.get("class_weight_power"),
                    ck.get("use_spectral"))
        now_cfg = (alpha, T, CLASS_WEIGHT_POWER, USE_SPECTRAL)
        if prev_cfg != now_cfg:
            raise SystemExit(
                f"\nCannot resume {tag}: the training configuration changed.\n"
                f"  checkpoint was (alpha, T, cw_power, use_spectral) = {prev_cfg}\n"
                f"  this run needs                                    = {now_cfg}\n"
                f"  Resuming would blend two different objectives.\n"
                f"  FIX: delete {out} and start fresh:\n"
                f"      import shutil; shutil.rmtree('{out}', ignore_errors=True)\n")
        if ck.get("schedule_shape") != shape:
            raise SystemExit(f"Cannot resume {tag}: schedule changed "
                             f"{ck.get('schedule_shape')} -> {shape}. Delete {out}.")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best, since = ck["epoch"]+1, ck["best_macro_f1"], ck.get("since_best", 0)
        print(f"  RESUMED at epoch {start} (best macro-F1 {best:.4f})")

    t0 = time.time()
    for ep in range(start, EPOCHS):
        model.train(); tot = th = ts = 0.0
        for xt, xs, tl, y, mask in dl_tr:
            xt, xs = xt.to(device, non_blocking=True), xs.to(device, non_blocking=True)
            tl = tl.to(device, non_blocking=True)
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
                    loss, hard, soft = distillation_loss(lg, tl[sl], y[sl], T, alpha, cw)
                w = (nvc/nv).float()
                scaler.scale(loss*w).backward()
                tot += loss.item()*w.item(); th += hard.item()*w.item(); ts += soft.item()*w.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()

        m = evaluate(model, dl_va, device, amp)
        rec = {"run": tag, "epoch": ep, "alpha": alpha, "T": T,
               "loss": tot/steps, "loss_hard": th/steps, "loss_soft": ts/steps,
               "lr": opt.param_groups[0]["lr"],
               "elapsed_min": round((time.time()-t0)/60, 2), **m}
        with open(out / "training_metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

        is_best = m["macro_f1"] > best
        if is_best:
            best, since = m["macro_f1"], 0
        else:
            since += 1
        state = {"run": tag, "epoch": ep, "model": model.state_dict(),
                 "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                 "scaler": scaler.state_dict(), "best_macro_f1": best,
                 "since_best": since, "n_parameters": npar,
                 "alpha": alpha, "T": T, "splits": SPLITS,
                 "class_weight_power": CLASS_WEIGHT_POWER,
                 "use_spectral": USE_SPECTRAL,
                 "schedule_shape": {"epochs": EPOCHS, "steps": steps}}
        torch.save(state, last)
        if is_best:
            torch.save(state, out / "student_best.pt")

        pc = " ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())
        print(f"ep {ep:3d} | loss {rec['loss']:.4f} (hard {rec['loss_hard']:.3f} "
              f"soft {rec['loss_soft']:.3f}) | mF1 {m['macro_f1']:.4f} | "
              f"k {m['kappa']:.4f} | acc {m['accuracy']:.4f} | {pc} | "
              f"N1r {m['prediction_ratio']['N1']:.2f}x{'  <- best' if is_best else ''}",
              flush=True)
        if EARLY_STOP_PATIENCE and since >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stop: no gain for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\nDONE {tag} in {(time.time()-t0)/60:.1f} min | best val macro-F1 {best:.4f}")
    return best


# ===================================================================== MAIN ====
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0); sm = f"sm_{cap[0]}{cap[1]}"
        sup = torch.cuda.get_arch_list()
        if sup and sm not in sup:
            raise SystemExit(f"UNSUPPORTED GPU {torch.cuda.get_device_name(0)} ({sm}); "
                             f"build supports {sup}. Use 'GPU T4 x2'.")
        print(f"GPU: {torch.cuda.get_device_name(0)} ({sm})")

    index = find_file("index.csv")
    root = index.parent.parent
    ckpt = find_file("teacher_best.pt")
    print(f"data root : {root}")
    print(f"teacher   : {ckpt}")

    exp, ep, vmf1, cfg = identify_teacher(ckpt)
    print(f"  checkpoint identifies as: {exp}  (epoch {ep}, val macro-F1 "
          f"{vmf1:.4f})" + (f"  config={cfg}" if cfg else ""))
    if exp != TEACHER_TAG:
        raise SystemExit(
            f"\nTEACHER MISMATCH.\n"
            f"  TEACHER_TAG is set to '{TEACHER_TAG}', but the attached "
            f"checkpoint is '{exp}'.\n"
            f"  Every rung saves a file called 'teacher_best.pt', so this is "
            f"easy to get wrong,\n"
            f"  and the run would be labelled as distilling from "
            f"'{TEACHER_TAG}' while actually\n"
            f"  using '{exp}'. That would corrupt the RQ3 comparison.\n"
            f"  FIX: set TEACHER_TAG = \"{exp}\" at the top, or attach the "
            f"'{TEACHER_TAG}' checkpoint.\n")
    print(f"  provenance OK: distilling from {exp} as intended")

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
    print(f"train {len(tr)} rec / {tr['subject'].nunique()} subj | "
          f"val {len(va)} rec / {va['subject'].nunique()} subj | "
          f"test {len(SPLITS['test'])} subj HELD OUT (never loaded)\n")

    print("STEP 1 - cache teacher soft targets")
    cache = Path(OUT_ROOT) / f"teacher_logits_{TEACHER_TAG}"
    cache_teacher(root, tr, ckpt, device, cache)

    print("\nSTEP 2 - distilled student")
    b1 = train_one(f"student_distilled_{TEACHER_TAG}", ALPHA_DISTILL,
                   root, tr, va, cache, device, amp)

    if RUN_BASELINE:
        print("\nSTEP 3 - baseline student (same trainer, alpha=1.0)")
        b2 = train_one(f"student_baseline_{TEACHER_TAG}", 1.0,
                       root, tr, va, cache, device, amp)
        b2_src = "trained in this run"
    else:
        b2 = BASELINE_REF_VAL_MF1
        b2_src = "stored student_baseline_E0 (teacher-independent, not retrained)"
        print("\nSTEP 3 - baseline student: SKIPPED")
        print("  alpha=1.0 ignores the teacher entirely, so with SEED=42 this would")
        print("  reproduce student_baseline_E0 bit-for-bit at a cost of ~2.5 GPU-hours.")
        print(f"  Reusing the stored result: validation macro-F1 {b2:.4f}, "
              f"test kappa {BASELINE_REF_TEST_KAPPA:.4f}")

    print(f"\n{'='*72}\nSUMMARY (validation macro-F1)\n{'='*72}")
    print(f"  distilled (alpha={ALPHA_DISTILL}, T={T}) : {b1:.4f}")
    print(f"  baseline  (alpha=1.0, hard only)  : {b2:.4f}   [{b2_src}]")
    print(f"  distillation gain                 : {b1-b2:+.4f}")
    print(f"\n  RQ3 second data point: teacher {TEACHER_TAG} -> "
          f"distillation gain {b1-b2:+.4f} on validation.")
    print("  The E0 rung gave -0.0134 on validation and -0.0523 on held-out test.")
    print("  A negative number here is a RESULT, not a failed run - it shows the")
    print("  effect persists across a 0.093 kappa range of teacher quality.")
    print("\nDownload from /kaggle/working:")
    print(f"  student_distilled_{TEACHER_TAG}/student_best.pt + training_metrics.jsonl")
    if RUN_BASELINE:
        print(f"  student_baseline_{TEACHER_TAG}/student_best.pt + training_metrics.jsonl")
    print("Test split was NEVER loaded. Evaluate it once, afterwards.")


main()
