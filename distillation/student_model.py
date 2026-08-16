"""
Phase 4 - the compact student.

Temporal-only, narrower and shallower than the teacher.

                        teacher        student
  transformer layers    3              2
  attention heads       4              2
  embedding dim         128            64
  atrous branches       4 (1,2,4,8)    2 (1,4)
  spectral branch       present        REMOVED

NOTE ON THE SPEC: the original plan listed the teacher at 4 transformer layers.
It is 3 - verified against the checkpoint's state_dict
(context.encoder.layers.0..2). The table above uses the measured value.

NOTE ON "no architectural surgery": the plan assumed SleepStagingModel.forward
already accepting temporal-only input meant the student needed no new class.
That was based on code/Phase1_48/Phase1reworked/model.py, which turns out to be
stale - no checkpoint in the repo matches it (README_V2.md s2.3). The real
teacher class always takes both streams. So the student IS a purpose-written
class; it is just a small one.

Dropping the spectral branch is the largest single saving and is defensible:
the spectral features are DERIVED from the same single EEG channel (DWT + STFT
band powers), so the student is not denied an independent information source -
it is required to learn from the raw signal what the teacher was handed
pre-computed. That is a real distillation question rather than a handicap.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5
STAGES = ["W", "N1", "N2", "N3", "REM"]


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)

    def forward(self, x):
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class StudentAtrousPyramid(nn.Module):
    """Same gated-pyramid idea as the teacher, fewer dilation branches."""

    def __init__(self, in_channels=1, hidden_channels=64, dilations=(1, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels, hidden_channels, 7, dilation=d, padding=3 * d)
            for d in dilations])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_channels * len(dilations), len(dilations), 1),
            nn.Softmax(dim=1))
        self.se = SEBlock(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels, 1)

    def forward(self, x):
        feats = [F.relu(b(x)) for b in self.branches]
        w = self.gate(torch.cat(feats, dim=1))
        out = 0
        for i, f in enumerate(feats):
            out = out + f * w[:, i:i + 1]
        return self.proj(self.se(out))


class StudentEpochEncoder(nn.Module):
    """
    n_tokens=1 mirrors the teacher's global average pool.
    n_tokens=K keeps K temporal sub-windows and attention-pools them, so the
    student can be given the E4 treatment independently of the teacher.
    """

    def __init__(self, embed_dim=64, hidden_channels=64, dilations=(1, 4),
                 n_tokens=1):
        super().__init__()
        self.n_tokens = n_tokens
        self.pyramid = StudentAtrousPyramid(1, hidden_channels, dilations)
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.fc = nn.Linear(hidden_channels, embed_dim)
        if n_tokens > 1:
            self.token_attn = nn.Linear(embed_dim, 1)

    def forward(self, x):
        B, T, L = x.shape
        f = self.pyramid(x.reshape(B * T, 1, L))
        f = self.pool(f)
        f = self.fc(f.transpose(1, 2))
        if self.n_tokens > 1:
            w = torch.softmax(self.token_attn(f), dim=1)
            f = (f * w).sum(dim=1)
        else:
            f = f.squeeze(1)
        return f.view(B, T, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class StudentSleepStagingModel(nn.Module):
    """
    Input:  x_temporal [B, T, 3000]  (raw EEG only - no spectral stream)
            padding_mask [B, T] bool, True at padded positions
    Output: logits [B, T, 5]

    forward() tolerates a (temporal, spectral) tuple and ignores the spectral
    element, so the same evaluation and distillation code paths work for both
    teacher and student without branching.
    """

    def __init__(self, embed_dim=64, heads=2, layers=2, dropout=0.2,
                 hidden_channels=64, dilations=(1, 4), n_tokens=1, max_len=512):
        super().__init__()
        self.temporal_encoder = StudentEpochEncoder(
            embed_dim=embed_dim, hidden_channels=hidden_channels,
            dilations=dilations, n_tokens=n_tokens)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.cls = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x_temporal, x_spectral=None, padding_mask=None):
        if isinstance(x_temporal, (tuple, list)):
            x_temporal = x_temporal[0]
        h = self.temporal_encoder(x_temporal)
        h = self.positional_encoding(h)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.cls(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    TEACHER_PARAMS = 649_229
    print(f"{'config':<44}{'params':>10}{'vs teacher':>12}")
    print("-" * 66)
    print(f"{'TEACHER (embed128 h4 L3, 4 dil, +spectral)':<44}"
          f"{TEACHER_PARAMS:>10,}{'1.00x':>12}")

    configs = [
        ("student embed64  h2 L2, 2 dil", dict(embed_dim=64, heads=2, layers=2)),
        ("student embed64  h2 L2, 2 dil, 8 tokens",
         dict(embed_dim=64, heads=2, layers=2, n_tokens=8)),
        ("student embed48  h2 L2, 2 dil", dict(embed_dim=48, heads=2, layers=2)),
        ("student embed32  h2 L2, 2 dil, hidden32",
         dict(embed_dim=32, heads=2, layers=2, hidden_channels=32)),
    ]
    for name, kw in configs:
        m = StudentSleepStagingModel(**kw)
        n = count_parameters(m)
        print(f"{name:<44}{n:>10,}{TEACHER_PARAMS/n:>11.2f}x")

    m = StudentSleepStagingModel()
    out = m(torch.randn(2, 16, 3000))
    print(f"\nforward [2,16,3000] -> {tuple(out.shape)} (expect (2, 16, 5))")
    out2 = m((torch.randn(2, 16, 3000), torch.randn(2, 16, 34)))
    print(f"tuple input accepted, spectral ignored -> {tuple(out2.shape)}")
    mask = torch.zeros(2, 16, dtype=torch.bool); mask[0, 10:] = True
    print(f"padding mask accepted -> {tuple(m(torch.randn(2,16,3000), None, mask).shape)}")
