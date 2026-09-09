"""Roadmap item 3.1: evaluate the SC-trained model on the held-out ST cohort.

This is the one generalisation claim available inside Sleep-EDFx: a model that
has never seen a telemetry recording, scored on all 22 ST subjects.

REPORTING RULES, built in rather than left to the reader
-------------------------------------------------------
1. Bootstrap clusters by SUBJECT, not recording. Two nights of one person are
   not independent evidence; resampling recordings understates every interval.
2. Per-class F1 and the confusion matrix are reported alongside kappa. The
   cohorts differ sharply in stage architecture - ST has ~2x the N3 and ~1/3
   the wake - and kappa is prevalence-sensitive, so kappa alone cannot separate
   "the model did not transfer" from "the priors moved underneath it".
3. This kappa is NOT comparable to the headline 0.7001. Different population.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
S2I = {s: i for i, s in enumerate(STAGES)}
LAB = list(range(5))
NBOOT = 10000


# ----- model, mirrored from the trainer (single-channel multiscale student) ---
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
    def __init__(self, embed=64, n_tokens=4, fine_ch=(24, 48), coarse_ch=(24, 48),
                 dropout=0.1, in_channels=1):
        super().__init__()
        self.n_tokens, self.in_channels = n_tokens, in_channels
        f1, f2 = fine_ch
        self.fine = nn.Sequential(
            ConvBlock(in_channels, f1, 50, 6, 25), nn.MaxPool1d(8, 8),
            nn.Dropout(dropout), ConvBlock(f1, f2, 8), nn.MaxPool1d(4, 4))
        c1, c2 = coarse_ch
        self.coarse = nn.Sequential(
            ConvBlock(in_channels, c1, 200, 25, 100), nn.MaxPool1d(4, 4),
            nn.Dropout(dropout), ConvBlock(c1, c2, 6), nn.MaxPool1d(2, 2))
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.proj = nn.Linear(f2 + c2, embed)
        if n_tokens > 1:
            self.token_attn = nn.Linear(embed, 1)

    def forward(self, x):
        B, T, L = x.shape
        z = x.reshape(B * T, 1, L)
        h = torch.cat([self.pool(self.fine(z)), self.pool(self.coarse(z))], 1)
        h = self.proj(h.transpose(1, 2))
        h = (h * torch.softmax(self.token_attn(h), 1)).sum(1) if self.n_tokens > 1 else h.squeeze(1)
        return h.view(B, T, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class StudentSleepStagingModel(nn.Module):
    def __init__(self, embed=64, heads=2, layers=2, dropout=0.2, n_tokens=4,
                 max_len=512, use_spectral=True, spec_dim=34):
        super().__init__()
        self.use_spectral = use_spectral
        self.temporal_encoder = MultiScaleEpochEncoder(embed=embed, n_tokens=n_tokens)
        if use_spectral:
            self.spectral_encoder = nn.Linear(spec_dim, embed)
            self.fusion = nn.Sequential(nn.Linear(embed * 2, embed),
                                        nn.LayerNorm(embed), nn.GELU())
        self.positional_encoding = PositionalEncoding(embed, max_len)
        l = nn.TransformerEncoderLayer(embed, heads, embed * 4, dropout,
                                       batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(l, layers)
        self.cls = nn.Linear(embed, 5)

    def forward(self, xt, xs=None, mask=None):
        h = self.temporal_encoder(xt)
        if self.use_spectral:
            h = self.fusion(torch.cat([h, self.spectral_encoder(xs)], -1))
        return self.cls(self.encoder(self.positional_encoding(h), src_key_padding_mask=mask))


def kappa(y, p):
    return float(cohen_kappa_score(y, p, labels=LAB))


def boot_subject(per_rec, subjects, rec2subj, stat, n=NBOOT, seed=20260830):
    """Clustered bootstrap: resample SUBJECTS, take all their recordings."""
    rng = np.random.default_rng(seed)
    by = {s: [r for r in per_rec if rec2subj[r] == s] for s in subjects}
    vals = np.empty(n)
    for b in range(n):
        sel = rng.integers(0, len(subjects), len(subjects))
        recs = [r for i in sel for r in by[subjects[i]]]
        y = np.concatenate([per_rec[r][0] for r in recs])
        p = np.concatenate([per_rec[r][1] for r in recs])
        vals[b] = stat(y, p)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="student_N4kd_sc2st")
    ap.add_argument("--data", default=str(ROOT / "processed_sleepedf"))
    ap.add_argument("--out", default=str(HERE / "results" / "cohort_transfer.json"))
    a = ap.parse_args()

    d = HERE / "results" / "students" / a.model
    ck = torch.load(d / "student_best.pt", map_location="cpu", weights_only=False)
    cs = json.loads((HERE / "cohort_split.json").read_text())
    sp = json.loads((HERE / "splits.json").read_text())
    rbs = sp["recordings_by_subject"]

    # refuse to score a model that saw the cohort
    tr = set(ck["splits"]["train"]) | set(ck["splits"]["val"])
    st_subj = sorted(cs["test"])
    if tr & set(st_subj):
        raise SystemExit(f"{a.model} trained on {len(tr & set(st_subj))} ST subjects. "
                         "This is not a transfer measurement.")
    scale = ck["eeg_scale"]
    print(f"model {a.model}  |  {ck['n_parameters']:,} params  |  eeg_scale {scale}")
    print(f"trained on {len(ck['splits']['train'])} SC subjects, never saw ST\n")

    model = StudentSleepStagingModel(use_spectral=ck.get("use_spectral", True),
                                     n_tokens=ck.get("n_tokens", 4))
    model.load_state_dict(ck["model"])
    model.eval()

    data = Path(a.data)
    idx = pd.read_csv(data / "index.csv")
    idx["rec"] = [Path(str(p)).stem for p in idx["tensor_path"]]
    idx = idx.set_index("rec")

    recs = [r for s in st_subj for r in rbs[s]]
    rec2subj = {r: s for s in st_subj for r in rbs[s]}
    print(f"scoring {len(recs)} ST recordings from {len(st_subj)} subjects")

    per_rec = {}
    with torch.no_grad():
        for r in recs:
            row = idx.loc[r]
            xt = torch.load(data / "tensors" / f"{r}.pt", map_location="cpu")
            xs = torch.load(data / "spectral" / f"{r}_spectral.pt", map_location="cpu")
            y = np.array([S2I[s] for s in str(row["stage_sequence"]).split()])
            n = min(len(y), xt.shape[0], xs.shape[0])
            xt, xs, y = xt[:n].float() * scale, xs[:n].float(), y[:n]
            preds = []
            for i in range(0, n, 256):                       # window, no overlap
                lg = model(xt[i:i + 256].unsqueeze(0), xs[i:i + 256].unsqueeze(0))
                preds.append(lg.argmax(-1)[0].numpy())
            per_rec[r] = (y, np.concatenate(preds)[:n])

    Y = np.concatenate([per_rec[r][0] for r in recs])
    P = np.concatenate([per_rec[r][1] for r in recs])
    k = kappa(Y, P)
    mf1 = float(f1_score(Y, P, average="macro", labels=LAB, zero_division=0))
    _, _, f1, _ = precision_recall_fscore_support(Y, P, labels=LAB, zero_division=0)
    lo, hi = boot_subject(per_rec, st_subj, rec2subj, kappa)

    rows = [json.loads(l) for l in (d / "training_metrics.jsonl").read_text().splitlines() if l.strip()]
    best = max(rows, key=lambda r: r["macro_f1"])

    print("\n" + "=" * 72)
    print("COHORT TRANSFER  -  trained on SC, scored on ST")
    print("=" * 72)
    print(f"  held-out SC validation (same model)   kappa {best['kappa']:.4f}")
    print(f"  held-out ST cohort                    kappa {k:.4f}   "
          f"95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  transfer gap                          {k - best['kappa']:+.4f}")
    print(f"\n  macro-F1 on ST {mf1:.4f}   ({len(Y):,} epochs, "
          f"{len(recs)} recordings, {len(st_subj)} subjects)")
    print("\n  interval clusters by SUBJECT (22), not recording (44).")

    print(f"\n  {'stage':<6}{'F1 on ST':>10}{'ST prevalence':>16}{'predicted':>12}")
    for i, s in enumerate(STAGES):
        print(f"  {s:<6}{f1[i]:>10.4f}{(Y == i).mean():>15.1%}{(P == i).mean():>12.1%}")

    cm = confusion_matrix(Y, P, labels=LAB)
    print(f"\n  confusion (rows = expert, cols = model)")
    print("        " + "".join(f"{s:>8}" for s in STAGES))
    for i, s in enumerate(STAGES):
        print(f"  {s:<6}" + "".join(f"{cm[i][j]:>8}" for j in range(5)))

    out = {
        "item": "roadmap 3.1 - leave-one-cohort-out, SC -> ST",
        "model": a.model, "n_parameters": ck["n_parameters"],
        "trained_on": {"cohort": "SC", "subjects": len(ck["splits"]["train"])},
        "scored_on": {"cohort": "ST", "subjects": len(st_subj),
                      "recordings": len(recs), "epochs": int(len(Y))},
        "kappa_ST": round(k, 4),
        "kappa_ST_ci_subject_clustered": [round(lo, 4), round(hi, 4)],
        "macro_f1_ST": round(mf1, 4),
        "kappa_SC_validation_same_model": round(best["kappa"], 4),
        "transfer_gap": round(k - best["kappa"], 4),
        "per_class_f1_ST": {s: round(float(f1[i]), 4) for i, s in enumerate(STAGES)},
        "prevalence_ST_true": {s: round(float((Y == i).mean()), 4) for i, s in enumerate(STAGES)},
        "prevalence_ST_predicted": {s: round(float((P == i).mean()), 4) for i, s in enumerate(STAGES)},
        "confusion_matrix": cm.tolist(),
        "bootstrap": {"unit": "subject", "n": NBOOT, "seed": 20260830},
        "not_comparable_to": ("the headline test kappa 0.7001, which is measured on a "
                              "15-subject subset of BOTH cohorts - a different population"),
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
