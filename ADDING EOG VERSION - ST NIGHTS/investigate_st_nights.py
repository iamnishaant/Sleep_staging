"""
Why do three ST (telemetry) validation nights fail for EVERY model?

    ST7201J0-PSG, ST7202J0-PSG (same subject, ST720), ST7151J0-PSG
    kappa ~0.1 for N4kd, its EEG-only twin, the EEG+EOG model and the teacher.

Two parts:

A. SIGNAL. Per-stage EEG/EOG statistics for every TRAIN and VALIDATION
   recording (168 of 197). Amplitude, absolute and relative band power,
   recording quality. Where do the bad nights sit relative to SC and other ST?
   Which signal property predicts per-night kappa across the 31 validation
   nights? TEST recordings are never opened.

B. COUNTERFACTUAL. Re-run the shipped N4kd on CPU, first reproducing its
   cached validation probabilities, then with each night's EEG amplitude
   multiplied by a gain g (and every spectral feature recomputed for that
   gain). If the bad nights recover at some g while good nights do not
   collapse, the failure is an amplitude mismatch the model was never trained
   to handle. A label-free per-recording amplitude normalisation is also
   scored.

Reads only existing files. Writes only inside this folder. Nothing trained.

    python investigate_st_nights.py            # both parts, ~10-15 min on CPU
    python investigate_st_nights.py --signal   # part A only
"""

import sys
sys.dont_write_bytecode = True

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import stft, welch
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIST = ROOT / "distillation"
RES = DIST / "results"
MC = ROOT / "processed_sleepedf_mc"
SPEC = ROOT / "processed_sleepedf" / "spectral"
CKPT = RES / "students" / "student_N4kd" / "student_best.pt"
VAL_CACHE = RES / "probs_student_N4kd_val"
OUT = HERE / "results"
FIG = HERE / "figures"

STAGES = ["W", "N1", "N2", "N3", "REM"]
S2I = {s: i for i, s in enumerate(STAGES)}
LAB = list(range(5))
FS = 100
UV = 1e6
BAD = ["ST7201J0-PSG", "ST7202J0-PSG", "ST7151J0-PSG"]
BANDS = {"slow 0.5-2": (0.5, 2), "delta 0.5-4": (0.5, 4), "theta 4-8": (4, 8), "alpha 8-12": (8, 12),
         "sigma 12-15": (12, 15), "beta 15-30": (15, 30), "high 30-45": (30, 45)}
GAINS = (0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0)
WINDOW = 256


# ============================================================ PART A: SIGNAL
def epoch_stats(x_uv):
    """x_uv: (n, 3000) microvolts -> dict of (n,) arrays."""
    f, P = welch(x_uv, fs=FS, nperseg=400, axis=-1)            # 0.25 Hz resolution
    tot = P[:, (f >= 0.5) & (f <= 45)].sum(1) + 1e-12
    out = {"amp_std_uv": x_uv.std(1)}
    for name, (lo, hi) in BANDS.items():
        bp = P[:, (f >= lo) & (f < hi)].sum(1)
        out[f"logpow {name}"] = np.log10(bp + 1e-6)
        out[f"rel {name}"] = bp / tot
    out["flat"] = (x_uv.std(1) < 1.0).astype(float)
    d = np.diff(x_uv, axis=1)
    out["max_run_identical"] = np.array([_max_zero_run(r) for r in d], float)
    return out


def _max_zero_run(d):
    z = np.concatenate([[0], (d == 0).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(z))
    return int((edges[1::2] - edges[::2]).max()) if len(edges) else 0


def stuck_flags(x_volts, min_run=100):
    """(n, 3000) -> (n,) bool: the epoch contains >= min_run consecutive identical
    samples (1 s at 100 Hz), or is flat (std < 1 uV). Labels are not used."""
    x = torch.as_tensor(x_volts, dtype=torch.float64)
    z = (x.diff(dim=-1) == 0).float().unsqueeze(1)                      # (n, 1, 2999)
    run = F.avg_pool1d(z, kernel_size=min_run - 1, stride=1).squeeze(1)  # 1.0 where min_run-1 diffs are all 0
    return ((run >= 1.0 - 1e-9).any(-1) | (x.std(-1) < 1e-6)).numpy()


def quality_table(df):
    rows = []
    for _, r in df.iterrows():
        x = torch.load(MC / "tensors" / f"{r['rec']}.pt", map_location="cpu").numpy()[:, 0]
        n = len(str(r["stage_sequence"]).split())
        fl = stuck_flags(x[:n])
        row = {"rec": r["rec"], "group": r["group"], "split": r["split"], "bad_epoch_frac": float(fl.mean())}
        if r["split"] == "val":
            c = np.load(VAL_CACHE / f"{r['rec']}.npz")
            y, pr = c["labels"], c["probs"].argmax(1)
            m = len(y); flm = fl[:m]
            row["kappa_all"] = float(cohen_kappa_score(y, pr, labels=LAB))
            row["kappa_clean"] = float(cohen_kappa_score(y[~flm], pr[~flm], labels=LAB)) if (~flm).sum() > 50 else None
            row["acc_on_flagged"] = float((y[flm] == pr[flm]).mean()) if flm.sum() else None
            row["acc_on_clean"] = float((y[~flm] == pr[~flm]).mean())
            row["flags"] = fl[:m]
        rows.append(row)
    return rows


def offset_scan(cache, rec, max_k=180):
    """kappa between N4kd predictions and the expert labels shifted by k epochs.
    k < 0: prediction at t+|k| is compared with the label at t, i.e. the labels
    run AHEAD of the signal by |k| epochs. Uses cached predictions only."""
    d = np.load(cache / f"{rec}.npz"); y = d["labels"]; p = d["probs"].argmax(1)
    best, k0 = (-9.0, 0), None
    for k in range(-max_k, max_k + 1):
        a, b = (p[: len(p) - k], y[k:]) if k >= 0 else (p[-k:], y[: len(y) + k])
        kk = float(cohen_kappa_score(b, a, labels=LAB))
        if k == 0:
            k0 = kk
        best = max(best, (kk, k))
    return {"rec": rec, "kappa_at_0": k0, "best_kappa": best[0], "best_offset_epochs": best[1]}


def load_index():
    df = pd.read_csv(MC / "index.csv")
    df["rec"] = [str(p).replace("\\", "/").rsplit("/", 1)[-1][:-3] for p in df["tensor_path"]]
    sp = json.loads((DIST / "splits.json").read_text(encoding="utf-8"))
    split_of = {r: k for k, subs in sp["splits"].items() for s in subs for r in sp["recordings_by_subject"][s]}
    df["split"] = df["rec"].map(split_of)
    df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)      # TEST never opened
    df["cohort"] = df["rec"].str[:2]
    df["group"] = np.where(df["rec"].isin(BAD), "BAD (3 ST val)",
                           df["cohort"] + "-" + df["split"])
    return df


def signal_table(df):
    rows = []
    t0 = time.time()
    for k, r in df.iterrows():
        x = torch.load(MC / "tensors" / f"{r['rec']}.pt", map_location="cpu").numpy().astype(np.float64)
        y = np.array([S2I[s] for s in str(r["stage_sequence"]).split()])
        n = min(len(y), x.shape[0]); x, y = x[:n], y[:n]
        eeg, eog = x[:, 0] * UV, x[:, 1] * UV
        se, so = epoch_stats(eeg), {"eog_amp_std_uv": eog.std(1)}
        rec = {"rec": r["rec"], "cohort": r["cohort"], "split": r["split"], "group": r["group"],
               "epochs": n, "hours": n / 120,
               "rec_amp_median_uv": float(np.median(se["amp_std_uv"])),        # label-free
               "flat_frac": float(se["flat"].mean()),
               "max_run_identical": float(se["max_run_identical"].max()),
               "transitions_per_hour": float((np.diff(y) != 0).sum() / (n / 120))}
        for i, s in enumerate(STAGES):
            m = y == i
            rec[f"share {s}"] = float(m.mean())
            if m.sum() >= 10:
                for key, v in se.items():
                    if key in ("flat", "max_run_identical"):
                        continue
                    rec[f"{s} {key}"] = float(np.median(v[m]))
                rec[f"{s} eog_amp_std_uv"] = float(np.median(so["eog_amp_std_uv"][m]))
        rows.append(rec)
        if (k + 1) % 40 == 0:
            print(f"  signal stats {k + 1}/{len(df)}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def robust_z(value, ref):
    ref = np.asarray(ref, float); ref = ref[np.isfinite(ref)]
    med = np.median(ref); mad = 1.4826 * np.median(np.abs(ref - med)) or 1e-9
    return (value - med) / mad


# ================================================ PART B: COUNTERFACTUAL (N4kd)
# N4kd's model, copied verbatim (single-channel path) from
# distillation/kaggle_train_student_kd.py so the checkpoint loads unchanged.
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
    def __init__(self, embed=64, n_tokens=4, fine_ch=(24, 48), coarse_ch=(24, 48), dropout=0.1):
        super().__init__()
        self.n_tokens = n_tokens
        f1, f2 = fine_ch
        self.fine = nn.Sequential(ConvBlock(1, f1, 50, 6, 25), nn.MaxPool1d(8, 8), nn.Dropout(dropout),
                                  ConvBlock(f1, f2, 8), nn.MaxPool1d(4, 4))
        c1, c2 = coarse_ch
        self.coarse = nn.Sequential(ConvBlock(1, c1, 200, 25, 100), nn.MaxPool1d(4, 4), nn.Dropout(dropout),
                                    ConvBlock(c1, c2, 6), nn.MaxPool1d(2, 2))
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.proj = nn.Linear(f2 + c2, embed)
        if n_tokens > 1:
            self.token_attn = nn.Linear(embed, 1)

    def forward(self, x):
        B, T, L = x.shape
        z = x.reshape(B * T, 1, L)
        h = torch.cat([self.pool(self.fine(z)), self.pool(self.coarse(z))], dim=1)
        h = self.proj(h.transpose(1, 2))
        h = (h * torch.softmax(self.token_attn(h), dim=1)).sum(1) if self.n_tokens > 1 else h.squeeze(1)
        return h.view(B, T, -1)


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


class N4kdModel(nn.Module):
    def __init__(self, embed=64, heads=2, layers=2, dropout=0.2, n_tokens=4, spec_dim=34):
        super().__init__()
        self.temporal_encoder = MultiScaleEpochEncoder(embed=embed, n_tokens=n_tokens)
        self.spectral_encoder = nn.Linear(spec_dim, embed)
        self.fusion = nn.Sequential(nn.Linear(embed * 2, embed), nn.LayerNorm(embed), nn.GELU())
        self.positional_encoding = PositionalEncoding(embed)
        l = nn.TransformerEncoderLayer(embed, heads, embed * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(l, layers)
        self.cls = nn.Linear(embed, 5)

    def forward(self, xt, xs):
        h = self.fusion(torch.cat([self.temporal_encoder(xt), self.spectral_encoder(xs)], -1))
        return self.cls(self.encoder(self.positional_encoding(h)))


def stft_block(x_volts):
    """The 10 STFT/ratio features of code/Phase2_48/spectral_preprocess.py,
    vectorised over epochs, on raw volts exactly as the cache was built."""
    f, _, Z = stft(x_volts, fs=FS, window="hann", nperseg=2 * FS, noverlap=FS, axis=-1)
    pm = (np.abs(Z) ** 2).mean(-1)
    tot = pm.sum(1)
    bp = lambda lo, hi: pm[:, (f >= lo) & (f <= hi)].sum(1)
    d, th, a, b, g = bp(.5, 4), bp(4, 8), bp(8, 13), bp(13, 30), bp(30, 45)
    ps = pm / (pm.sum(1, keepdims=True) + 1e-12)
    ent = -(ps * np.log(ps + 1e-12)).sum(1)
    cum = np.cumsum(pm, 1)
    sef = f[(cum >= 0.95 * cum[:, -1:]).argmax(1)]
    return np.stack([d / tot, th / tot, a / tot, b / tot, g / tot, ent, sef,
                     d / (b + 1e-12), th / (a + 1e-12), (d + th) / (a + b + 1e-12)], 1)


def spectral_for_gain(S_cached, x_volts, g):
    """Spectral features of (g * signal). g == 1 returns the cache untouched.
    DWT block: energy and variance scale by g^2 (exact - the DWT is linear);
    log_energy = log(g^2 * energy + 1e-12) (exact); DWT entropy held fixed
    (its 1e-12 term moves it by < 0.1% at these magnitudes). STFT/ratio block
    recomputed exactly."""
    if g == 1.0:
        return S_cached
    S = S_cached.astype(np.float64).copy()
    k = np.arange(24)
    S[:, k[k % 4 == 0]] *= g * g
    S[:, k[k % 4 == 3]] *= g * g
    S[:, k[k % 4 == 1]] = np.log(S[:, k[k % 4 == 0]] + 1e-12)
    S[:, 24:] = stft_block(x_volts * g)
    return S.astype(np.float32)


@torch.no_grad()
def predict(model, x_volts, S, eeg_scale):
    t = torch.from_numpy(x_volts.astype(np.float32)) * eeg_scale
    s = torch.from_numpy(S.astype(np.float32))
    P = np.empty((len(t), 5), np.float32)
    for st in range(0, len(t), WINDOW):                         # as evaluate_student.predict
        en = min(st + WINDOW, len(t))
        P[st:en] = F.softmax(model(t[st:en].unsqueeze(0), s[st:en].unsqueeze(0)).float(), -1).squeeze(0).numpy()
    return P


def counterfactual(df_val, sig):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = N4kdModel(); model.load_state_dict(ck["model"]); model.eval()
    scale = ck["eeg_scale"]
    torch.set_num_threads(max(1, torch.get_num_threads()))
    sc_train_amp = float(sig.loc[sig["group"] == "SC-train", "rec_amp_median_uv"].median())

    data, repro = {}, []
    for _, r in df_val.iterrows():
        x = torch.load(MC / "tensors" / f"{r['rec']}.pt", map_location="cpu").numpy()[:, 0].astype(np.float64)
        S = torch.load(SPEC / f"{r['rec']}_spectral.pt", map_location="cpu").numpy()
        c = np.load(VAL_CACHE / f"{r['rec']}.npz")
        n = len(c["labels"]); x, S = x[:n], S[:n]
        P = predict(model, x, S, scale)
        repro.append(float(np.abs(P - c["probs"]).max()))
        data[r["rec"]] = (x, S, c["labels"].astype(int), r["cohort"])
    worst = max(repro)
    if worst > 1e-3:
        raise SystemExit(f"Cannot reproduce cached N4kd validation probabilities (max diff {worst:.2e}).")
    print(f"  reproduced cached N4kd validation probabilities: max |diff| {worst:.1e}")

    def group_of(rec, cohort):
        return "BAD (3 ST val)" if rec in BAD else ("ST-val (other)" if cohort == "ST" else "SC-val")

    def score(preds):
        out = {}
        for gname in ("BAD (3 ST val)", "ST-val (other)", "SC-val", "ALL 31"):
            ys, ps = [], []
            for rec, (x, S, y, coh) in data.items():
                if gname == "ALL 31" or group_of(rec, coh) == gname:
                    ys.append(y); ps.append(preds[rec].argmax(1))
            yy, pp = np.concatenate(ys), np.concatenate(ps)
            out[gname] = {"kappa": float(cohen_kappa_score(yy, pp, labels=LAB)), "accuracy": float((yy == pp).mean()),
                          "pred_share_N3": float((pp == 3).mean()), "pred_share_REM": float((pp == 4).mean())}
        return out

    sweep = {}
    t0 = time.time()
    for g in GAINS:
        preds = {rec: predict(model, x * g, spectral_for_gain(S, x, g), scale) for rec, (x, S, y, coh) in data.items()}
        sweep[str(g)] = score(preds)
        print(f"  gain {g:<4}  " + "  ".join(f"{k}: k={v['kappa']:.3f}" for k, v in sweep[str(g)].items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        if g == 1.0:
            per_rec_base = {rec: float(cohen_kappa_score(data[rec][2], preds[rec].argmax(1), labels=LAB)) for rec in data}

    # Label-free per-recording normalisation: scale each night so its median
    # epoch amplitude equals the SC-train median. Uses no labels.
    norm_preds, gains_used = {}, {}
    for rec, (x, S, y, coh) in data.items():
        g = sc_train_amp / float(np.median(x.std(1) * UV))
        gains_used[rec] = g
        norm_preds[rec] = predict(model, x * g, spectral_for_gain(S, x, g), scale)
    per_rec_norm = {rec: float(cohen_kappa_score(data[rec][2], norm_preds[rec].argmax(1), labels=LAB)) for rec in data}
    return {"reproduction_max_abs_diff": worst, "gain_sweep": sweep,
            "label_free_amplitude_normalisation": {
                "target_median_epoch_std_uv": sc_train_amp, "gain_per_recording": gains_used,
                "scores": score(norm_preds), "per_recording_kappa": per_rec_norm,
                "per_recording_kappa_baseline": per_rec_base}}


# ================================================================= REPORT
def fmt(v, d=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def build_report(sig, kap, cf):
    L = ["# ST nights investigation", "",
         "_Generated by `investigate_st_nights.py`. Train + validation recordings only; test recordings are never opened._", ""]
    groups = ["SC-train", "ST-train", "SC-val", "ST-val", "BAD (3 ST val)"]
    g = sig.copy()
    g.loc[(g["cohort"] == "ST") & (g["split"] == "val") & ~g["rec"].isin(BAD), "group"] = "ST-val"
    present = [x for x in groups if (g["group"] == x).any()]
    counts = {x: int((g["group"] == x).sum()) for x in present}
    L += ["Groups (recordings): " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".", ""]

    L += ["## A1. Recording level", "", "| group | median epoch amplitude (µV) | transitions / hour | hours | W | N1 | N2 | N3 | REM |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in present:
        d = g[g["group"] == x]
        L.append(f"| {x} | {d['rec_amp_median_uv'].median():.1f} | {d['transitions_per_hour'].median():.1f} | "
                 f"{d['hours'].median():.1f} | " + " | ".join(f"{100*d[f'share {s}'].median():.0f}%" for s in STAGES) + " |")
    L += ["", "_Stage shares are the expert's. Medians across recordings._", ""]

    for stage in ("N2", "REM", "N3", "W"):
        L += [f"## A2. {stage} epochs — EEG by group (medians across recordings)", "",
              "| group | amplitude µV | " + " | ".join(f"log10 {b}" for b in BANDS) + " | rel sigma | rel delta | EOG µV |",
              "|---" * (len(BANDS) + 5) + "|"]
        for x in present:
            d = g[g["group"] == x]
            if f"{stage} amp_std_uv" not in d:
                continue
            L.append(f"| {x} | {d[f'{stage} amp_std_uv'].median():.1f} | " +
                     " | ".join(fmt(d[f'{stage} logpow {b}'].median()) for b in BANDS) +
                     f" | {fmt(d[f'{stage} rel sigma 12-15'].median(), 3)} | {fmt(d[f'{stage} rel delta 0.5-4'].median(), 3)}"
                     f" | {fmt(d[f'{stage} eog_amp_std_uv'].median(), 1)} |")
        L.append("")

    L += ["## A3. How far are the bad nights from SC training nights? (robust z-scores)", "",
          "z = (value − SC-train median) / (1.4826 × MAD). |z| > 3 is far outside the SC range.", ""]
    ref = g[g["group"] == "SC-train"]
    feats = [c for c in g.columns if any(c.startswith(s + " ") for s in STAGES) and "share" not in c] + \
            ["rec_amp_median_uv", "transitions_per_hour"]
    L += ["| recording | the five largest deviations |", "|---|---|"]
    for rec in BAD:
        row = g[g["rec"] == rec].iloc[0]
        zs = [(robust_z(row[c], ref[c]), c) for c in feats if c in row and np.isfinite(row[c])]
        zs = sorted(zs, key=lambda t: -abs(t[0]))[:5]
        L.append(f"| {rec} | " + "; ".join(f"{c} z={z:+.1f}" for z, c in zs) + " |")
    st_ref = g[g["group"] == "ST-train"]
    L += ["", "| recording | the five largest deviations from **ST-train** |", "|---|---|"]
    for rec in BAD:
        row = g[g["rec"] == rec].iloc[0]
        zs = [(robust_z(row[c], st_ref[c]), c) for c in feats if c in row and np.isfinite(row[c])]
        zs = sorted(zs, key=lambda t: -abs(t[0]))[:5]
        L.append(f"| {rec} | " + "; ".join(f"{c} z={z:+.1f}" for z, c in zs) + " |")

    L += ["", "## A4. What predicts a night's κ? (Spearman across the 31 validation nights)", "",
          "| signal feature | ρ with N4kd κ | p |", "|---|---:|---:|"]
    v = g[g["split"] == "val"].merge(kap, on="rec")
    cand = []
    for c in feats + ["flat_frac", "max_run_identical"] + [f"share {s}" for s in STAGES]:
        if c in v and v[c].notna().sum() >= 20 and v[c].nunique() > 3:
            rho, p = spearmanr(v[c], v["kappa"], nan_policy="omit")
            cand.append((rho, p, c))
    for rho, p, c in sorted(cand, key=lambda t: -abs(t[0]))[:12]:
        L.append(f"| {c} | {rho:+.2f} | {p:.3g} |")
    L += ["", "_31 nights, many features: treat individual p-values as descriptive, not confirmatory._", ""]

    q = sig[["rec", "group", "flat_frac", "max_run_identical"]]
    L += ["## A5. Recording quality", "",
          f"Recordings with any flat epoch (std < 1 µV): {int((q['flat_frac'] > 0).sum())} of {len(q)}. "
          f"Bad nights' flat fraction: " + ", ".join(f"{r} {100*q.loc[q['rec']==r,'flat_frac'].iloc[0]:.1f}%" for r in BAD) + ". "
          f"Longest run of identical consecutive samples (saturation/dropout): bad nights "
          + ", ".join(f"{int(q.loc[q['rec']==r,'max_run_identical'].iloc[0])}" for r in BAD)
          + f"; median over all {len(q)} recordings {int(q['max_run_identical'].median())}.", ""]

    if cf:
        L += ["## B1. Counterfactual: N4kd re-run with every night's EEG amplitude × g", "",
              f"Cached N4kd validation probabilities reproduced first (max |diff| {cf['reproduction_max_abs_diff']:.1e}). "
              "Spectral features are recomputed for each gain. κ is plain argmax (the shipped decoder moves κ by < 0.001).", "",
              "| gain g | BAD 3 κ | other ST-val κ | SC-val κ | all 31 κ | all 31 acc | BAD: predicted N3 share | BAD: predicted REM share |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for gk, s in cf["gain_sweep"].items():
            L.append(f"| {gk} | {s['BAD (3 ST val)']['kappa']:.3f} | {s['ST-val (other)']['kappa']:.3f} | {s['SC-val']['kappa']:.3f} | "
                     f"{s['ALL 31']['kappa']:.3f} | {100*s['ALL 31']['accuracy']:.1f}% | {100*s['BAD (3 ST val)']['pred_share_N3']:.0f}% | "
                     f"{100*s['BAD (3 ST val)']['pred_share_REM']:.0f}% |")
        n = cf["label_free_amplitude_normalisation"]
        L += ["", "## B2. Label-free per-recording amplitude normalisation (inference only, no retraining)", "",
              f"Each night scaled so its median epoch amplitude equals the SC-train median ({n['target_median_epoch_std_uv']:.1f} µV). No labels used.", "",
              "| group | κ baseline (g = 1) | κ normalised | acc normalised |", "|---|---:|---:|---:|"]
        base = cf["gain_sweep"]["1.0"]
        for k, s in n["scores"].items():
            L.append(f"| {k} | {base[k]['kappa']:.3f} | {s['kappa']:.3f} | {100*s['accuracy']:.1f}% |")
        L += ["", "| bad night | gain applied | κ before | κ after |", "|---|---:|---:|---:|"]
        for r in BAD:
            L.append(f"| {r} | {n['gain_per_recording'][r]:.2f} | {n['per_recording_kappa_baseline'][r]:.3f} | {n['per_recording_kappa'][r]:.3f} |")
    L += ["", "Figures: `figures/psd_by_group.png`, `figures/kappa_vs_gain.png`.", ""]
    return "\n".join(L)


def figures(df, cf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    groups = {"SC-train": df[(df["cohort"] == "SC") & (df["split"] == "train")]["rec"].tolist()[:40],
              "ST-train": df[(df["cohort"] == "ST") & (df["split"] == "train")]["rec"].tolist(),
              "BAD (3 ST val)": BAD}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, stage in zip(axes, ("N2", "REM")):
        for gname, recs in groups.items():
            psds = []
            for rec in recs:
                r = df[df["rec"] == rec].iloc[0]
                x = torch.load(MC / "tensors" / f"{rec}.pt", map_location="cpu").numpy()[:, 0].astype(np.float64) * UV
                y = np.array([S2I[s] for s in str(r["stage_sequence"]).split()])[: len(x)]
                m = y[: len(x)] == S2I[stage]
                if m.sum() >= 10:
                    f, P = welch(x[: len(y)][m], fs=FS, nperseg=400, axis=-1)
                    psds.append(np.log10(np.median(P, 0) + 1e-6))
            if psds:
                ax.plot(f, np.median(psds, 0), label=f"{gname} ({len(psds)})")
        ax.set_xlim(0.5, 35); ax.set_xlabel("Hz"); ax.set_title(f"{stage} epochs: EEG power spectrum")
    axes[0].set_ylabel("log10 power (µV²/Hz)"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "psd_by_group.png", dpi=150); plt.close(fig)

    if cf:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        gs = [float(k) for k in cf["gain_sweep"]]
        for gname in ("BAD (3 ST val)", "ST-val (other)", "SC-val", "ALL 31"):
            ax.plot(gs, [v[gname]["kappa"] for v in cf["gain_sweep"].values()], marker="o", label=gname)
        ax.set_xscale("log"); ax.set_xticks(gs, [str(k) for k in gs])
        ax.axvline(1.0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("EEG amplitude multiplied by g"); ax.set_ylabel("κ (N4kd, argmax)")
        ax.set_title("Does a different amplitude rescue the bad nights?"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "kappa_vs_gain.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", action="store_true", help="part A only")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_index()
    print(f"{len(df)} train+val recordings (test never opened); bad nights: {BAD}")
    sig = signal_table(df)
    sig.to_csv(OUT / "signal_stats_per_recording.csv", index=False)

    dec = json.loads((RES / "sequence_decoding.json").read_text(encoding="utf-8"))
    kap = []
    for rec in sig.loc[sig["split"] == "val", "rec"]:
        c = np.load(VAL_CACHE / f"{rec}.npz")
        kap.append({"rec": rec, "kappa": float(cohen_kappa_score(c["labels"], c["probs"].argmax(1), labels=LAB))})
    kap = pd.DataFrame(kap)

    qrows = quality_table(df)
    qv = [r for r in qrows if r["split"] == "val"]
    yy, pp, ff = [], [], []
    for r in qv:
        c = np.load(VAL_CACHE / f"{r['rec']}.npz"); yy.append(c["labels"]); pp.append(c["probs"].argmax(1)); ff.append(r["flags"])
    yy, pp, ff = np.concatenate(yy), np.concatenate(pp), np.concatenate(ff)
    quality = {"min_stuck_run_s": 1.0,
               "per_recording": [{k: v for k, v in r.items() if k != "flags"} for r in qrows],
               "val_pooled": {"flagged_epochs": int(ff.sum()), "flagged_share": float(ff.mean()),
                              "kappa_all": float(cohen_kappa_score(yy, pp, labels=LAB)),
                              "kappa_clean_epochs": float(cohen_kappa_score(yy[~ff], pp[~ff], labels=LAB)),
                              "acc_flagged": float((yy[ff] == pp[ff]).mean()) if ff.any() else None,
                              "acc_clean": float((yy[~ff] == pp[~ff]).mean())}}
    (OUT / "recording_quality.json").write_text(json.dumps(quality, indent=1), encoding="utf-8")

    scan = []
    for split, cache in (("train", RES / "probs_student_N4kd_train"), ("val", VAL_CACHE)):
        for f in sorted(cache.glob("*.npz")):
            r = offset_scan(cache, f.stem); r["split"] = split; scan.append(r)
    (OUT / "label_offset_scan.json").write_text(json.dumps(scan, indent=1), encoding="utf-8")

    cf = None
    if not a.signal:
        print("counterfactual inference (N4kd on CPU)...")
        cf = counterfactual(df[df["split"] == "val"], sig)
        (OUT / "counterfactual_gain.json").write_text(json.dumps(cf, indent=1), encoding="utf-8")
    rep = build_report(sig, kap, cf)
    Q = ["", "## A6. Recording faults: epochs with a stuck (>= 1 s identical samples) or flat signal", "",
         "Flags use the raw signal only, never the labels.", "",
         "| group | recordings | recordings with any flagged epoch | median flagged share | max flagged share |",
         "|---|---:|---:|---:|---:|"]
    for gname in ["SC-train", "ST-train", "SC-val", "ST-val", "BAD (3 ST val)"]:
        rr = [r for r in qrows if (r["group"] == gname) or (gname == "ST-val" and r["split"] == "val"
              and r["rec"].startswith("ST") and r["rec"] not in BAD)]
        if not rr:
            continue
        fr = np.array([r["bad_epoch_frac"] for r in rr])
        Q.append(f"| {gname} | {len(rr)} | {int((fr > 0).sum())} | {100*np.median(fr):.1f}% | {100*fr.max():.1f}% |")
    Q += ["", "| validation night | flagged epochs | κ all epochs | κ clean epochs only | accuracy on flagged | accuracy on clean |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(qv, key=lambda r: -r["bad_epoch_frac"]):
        if r["bad_epoch_frac"] > 0:
            Q.append(f"| {r['rec']} | {100*r['bad_epoch_frac']:.1f}% | {r['kappa_all']:.3f} | {fmt(r['kappa_clean'], 3)} | "
                     f"{fmt(r['acc_on_flagged'], 3)} | {r['acc_on_clean']:.3f} |")
    vp = quality["val_pooled"]
    Q += ["", f"Pooled over the 31 validation nights: {vp['flagged_epochs']:,} flagged epochs ({100*vp['flagged_share']:.1f}%). "
          f"Accuracy on flagged epochs {100*vp['acc_flagged']:.1f}% vs {100*vp['acc_clean']:.1f}% on clean ones; "
          f"κ {vp['kappa_all']:.4f} on all epochs, {vp['kappa_clean_epochs']:.4f} on clean epochs only "
          "(diagnostic - no reported figure excludes anything).", ""]
    sus = sorted([r for r in scan if r["best_offset_epochs"] != 0 and r["best_kappa"] - r["kappa_at_0"] > 0.05],
                 key=lambda r: r["kappa_at_0"] - r["best_kappa"])
    Q += ["## A7. Are the expert labels shifted in time? (offset scan)", "",
          "For every train and validation recording, the expert hypnogram is slid against N4kd's predictions "
          "by k = -180..+180 epochs (+-90 min). A correctly aligned night agrees best at k = 0. "
          "k < 0 means the labels run AHEAD of the signal. Train predictions are in-sample.", "",
          f"{len(scan)} recordings scanned; {len(sus)} agree better (> +0.05 kappa) at a non-zero offset:", "",
          "| split | recording | κ at offset 0 | best κ | at offset | minutes |", "|---|---|---:|---:|---:|---:|"]
    for r in sus:
        Q.append(f"| {r['split']} | {r['rec']} | {r['kappa_at_0']:.3f} | {r['best_kappa']:.3f} | "
                 f"{r['best_offset_epochs']:+d} | {r['best_offset_epochs'] * 0.5:+.1f} |")
    Q += ["", "Every large offset is NEGATIVE (labels ahead of the signal), and almost all are ST recordings. "
          "`distillation/preprocess_multichannel.py` (copied unchanged from `code/Phase1_48/Phase1reworked/preprocess.py`) "
          "builds labels from `mne.read_annotations(hyp_path).onset`, whose onsets count from the HYPNOGRAM file's start, "
          "not from `raw.annotations` (re-aligned to the PSG start by `set_annotations`). If a hypnogram starts Δ later "
          "than its PSG, every label lands Δ early - exactly this signature. Confirm with "
          "`verify_hypnogram_alignment.py` on a machine that has the raw Sleep-EDF files.", ""]
    rep = rep.replace("Figures: `figures/psd_by_group.png`",
                      "\n".join(Q) + "\nFigures: `figures/psd_by_group.png`")
    (OUT / "ST_NIGHTS_REPORT.md").write_text(rep, encoding="utf-8")
    figures(df, cf)
    print(f"wrote {OUT / 'ST_NIGHTS_REPORT.md'}")


if __name__ == "__main__":
    main()
