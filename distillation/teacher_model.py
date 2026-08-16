"""
Teacher model — extracted from code/Phase2_48/temporal-spectral-fusion.ipynb (cells 5-6).

PROVENANCE
----------
The class hierarchy below reproduces the revision of the notebook model that
produced the best committed checkpoint:

    results/sleep staging/temporal spectral fusion/concat/best_model_fusion.pt
    validation kappa 0.6663 | accuracy 0.7575 | macro-F1 0.6814 | 714,765 params

That checkpoint predates the notebook's *current* class, which differs in three
ways (all of which scored WORSE - see README_V2.md section 4):
    - spectral_encoder is an MLP (Linear->GELU->LN->Linear->GELU->LN), not a Linear
    - context uses 6 transformer layers, not 3
    - auxiliary heads `temporal_cls` / `spectral_cls` are present

This module deliberately reproduces the OLDER revision, because it is the
stronger teacher. Do not "upgrade" it to match the notebook.

NOTE: code/Phase1_48/Phase1reworked/model.py defines a *different*
`SleepStagingModel` (attributes encoder/fusion/context). No checkpoint in the
repository matches it. It is stale; do not use it as the teacher.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5
STAGE_TO_IDX = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
IDX_TO_STAGE = {v: k for k, v in STAGE_TO_IDX.items()}


class SEBlock(nn.Module):
    """Squeeze-and-excitation over the channel axis of a 1-D feature map."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class AdaptiveAtrousPyramid(nn.Module):
    """Four dilated conv branches merged by a learned softmax gate."""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 64,
                 dilations: tuple = (1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7,
                      dilation=d, padding=3 * d)
            for d in dilations
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_channels * len(dilations), len(dilations), 1),
            nn.Softmax(dim=1),
        )
        self.se = SEBlock(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

    def forward(self, x):
        feats = [F.relu(b(x)) for b in self.branches]
        stacked = torch.cat(feats, dim=1)
        weights = self.gate(stacked)

        out = 0
        for i, f in enumerate(feats):
            out = out + f * weights[:, i:i + 1]

        return self.proj(self.se(out))


class EpochEncoder(nn.Module):
    """Per-epoch temporal encoder: [B, T, 3000] -> [B, T, embed_dim]."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.pyramid = AdaptiveAtrousPyramid()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        B, T, L = x.shape
        x = x.reshape(B * T, 1, L)
        f = self.pyramid(x)
        f = self.pool(f).squeeze(-1)
        f = self.fc(f)
        return f.view(B, T, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SleepTransformer(nn.Module):
    def __init__(self, embed_dim: int = 128, heads: int = 4, layers: int = 3,
                 dropout: float = 0.2, max_len: int = 512):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.cls = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x, padding_mask=None):
        x = self.positional_encoding(x)
        h = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.cls(h)


class FusedSleepStagingModel(nn.Module):
    """
    Teacher. Concat fusion of a CNN temporal branch and a linear spectral branch,
    followed by a shared transformer over the epoch sequence.

    Inputs
        x_temporal   : [B, T, 3000]  raw EEG, 30 s epochs @ 100 Hz
        x_spectral   : [B, T, 34]    spectral feature vector per epoch
        padding_mask : [B, T] bool, True at padded positions

    Output
        logits : [B, T, 5]
    """

    def __init__(self, spectral_input_dim: int = 34, embed_dim: int = 128,
                 heads: int = 4, layers: int = 3, dropout: float = 0.2,
                 fusion_type: str = "concat"):
        super().__init__()
        if fusion_type != "concat":
            raise ValueError(
                "Only 'concat' is reproduced here - it is the best committed "
                "checkpoint. See README_V2.md section 4."
            )
        self.fusion_type = fusion_type

        self.temporal_encoder = EpochEncoder(embed_dim=embed_dim)
        self.spectral_encoder = nn.Linear(spectral_input_dim, embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.context = SleepTransformer(
            embed_dim=embed_dim, heads=heads, layers=layers, dropout=dropout
        )

    def forward(self, x_temporal, x_spectral, padding_mask=None):
        h_temp = self.temporal_encoder(x_temporal)
        h_spec = self.spectral_encoder(x_spectral)
        fused = self.fusion(torch.cat([h_temp, h_spec], dim=-1))
        return self.context(fused, padding_mask)


# Checkpoint corresponding to this class revision.
DEFAULT_TEACHER_CKPT = (
    "results/sleep staging/temporal spectral fusion/concat/best_model_fusion.pt"
)


def load_teacher(checkpoint_path: str | Path,
                 device: str | torch.device = "cpu",
                 strict: bool = True) -> FusedSleepStagingModel:
    """
    Build the teacher and load weights.

    Handles the `module.` prefix left by nn.DataParallel. Raises if the
    checkpoint does not match this class exactly, rather than silently
    loading a partial state dict.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload: {type(state)}")

    # Strip DataParallel prefix.
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}

    model = FusedSleepStagingModel()
    missing, unexpected = model.load_state_dict(state, strict=False)

    if strict and (missing or unexpected):
        raise RuntimeError(
            "Teacher checkpoint does not match FusedSleepStagingModel.\n"
            f"  missing:    {sorted(missing)}\n"
            f"  unexpected: {sorted(unexpected)}"
        )

    model.to(device).eval()
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import argparse

    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Verify the teacher loads exactly.")
    ap.add_argument("--checkpoint", default=str(repo_root / DEFAULT_TEACHER_CKPT))
    args = ap.parse_args()

    model = load_teacher(args.checkpoint, strict=True)
    n = count_parameters(model)
    print(f"Teacher loaded cleanly from: {args.checkpoint}")
    print(f"  trainable parameters : {n:,}")
    print(f"  expected             : 714,765")
    print(f"  match                : {n == 714765}")

    B, T = 2, 16
    with torch.no_grad():
        out = model(torch.randn(B, T, 3000), torch.randn(B, T, 34))
    print(f"  forward [2,16,3000]+[2,16,34] -> {tuple(out.shape)} (expect (2, 16, 5))")
