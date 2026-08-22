"""
A multi-scale epoch encoder that can actually see sleep microstructure.

WHY THE EXISTING ONE CANNOT
---------------------------
`eeg_normalization_ablation.json` concluded that the raw EEG carries no
predictive information beyond the 34 precomputed spectral features, on the
evidence that normalising it (which revived a provably inert branch) moved
validation macro-F1 by -0.006. That experiment held the architecture fixed, and
the architecture is the binding constraint. The current encoder is:

    AtrousPyramid: Conv1d(1, 64, kernel=7, dilation in {1, 4})
    AdaptiveAvgPool1d(1)
    Linear(64, 64)

Two facts follow, and `python distillation/student_encoder.py` measures both
rather than asserting them:

  1. RECEPTIVE FIELD 25 SAMPLES = 0.25 s at 100 Hz.
     kernel 7 at dilation 4 spans 1 + (7-1)*4 = 25 samples. The 1x1 projection
     adds none.

  2. THE POOL IS GLOBAL. AdaptiveAvgPool1d(1) averages all 3000 positions into
     one vector.

So the encoder computes the mean, over 30 seconds, of a 0.25-second texture
detector. The events that define the stages this model is worst at are all
longer than that:

    sleep spindle    0.5 - 2 s   at 11-16 Hz    defines N2 against N1
    K-complex        >= 0.5 s                   defines N2 against N1
    slow wave        0.5 - 2 s   at 0.5-2 Hz    defines N3
    sawtooth wave    ~ 1 - 3 s   at 2-6 Hz      marks REM

None of them fit in 0.25 s, and averaging over the whole epoch discards where
in the epoch anything happened. A band-power feature vector beats this encoder
because it summarises 30 seconds of structure, while the encoder summarises
120 repetitions of a quarter-second. That the ablation found no gain is
therefore expected, and is evidence about THIS ENCODER, not about the EEG.

WHAT THIS REPLACES IT WITH
--------------------------
Two branches at deliberately different time scales, the standard
DeepSleepNet/TinySleepNet arrangement, plus multi-token output:

    fine    Conv1d(kernel=50, stride=6)    0.5 s stride 0.06 s   -> spindles, K-complexes
    coarse  Conv1d(kernel=200, stride=25)  2.0 s stride 0.25 s   -> slow waves, delta

each followed by pooling and a second convolution, so the effective receptive
field reaches several seconds. Outputs are concatenated and attention-pooled to
`n_tokens` positions per 30-second epoch instead of one, so within-epoch
structure survives into the transformer.

`n_tokens > 1` is not new machinery: `EpochEncoder` already supported it and it
was never switched on (N_TOKENS = 1 in every trainer).

WHAT THIS DOES NOT CLAIM
------------------------
That it will beat the spectral features. It might not. The point is that the
question "does the raw EEG add anything?" has not yet been asked with an
encoder capable of representing an answer, and this makes it askable. The
comparison to run is stated in `ABLATION` below.

THE INPUT MUST BE NORMALISED
----------------------------
The preprocessed EEG is in volts (~2e-5) against spectral features at ~7.
`check_branch_live.py` measured the resulting branch as INERT: zeroing the
entire EEG input changed 0.000 kappa and 0% of predictions. EEG_SCALE is the
1/std fitted on the TRAINING split in eeg_normalization.py. Training this
encoder on unnormalised input would reproduce the dead branch with more
parameters.

USAGE
    python distillation/student_encoder.py     # measured receptive fields, params
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5
STAGES = ["W", "N1", "N2", "N3", "REM"]

SAMPLE_RATE = 100
EPOCH_SAMPLES = 3000
TEACHER_PARAMS = 649_229

# 1/std of the raw EEG over the TRAINING split. See eeg_normalization.py.
EEG_SCALE = 15849.46

ABLATION = """
The three-way comparison this encoder exists to make possible. One variable,
same data, splits, schedule, seed and evaluation:

  A  current encoder, EEG_SCALE = 1          the stored M0. Branch INERT.
                                             val macro-F1 0.6881
  B  current encoder, EEG_SCALE = 15849.46   branch LIVE, 0.25 s receptive field.
                                             val macro-F1 0.6821  (-0.0060)
  C  THIS encoder,    EEG_SCALE = 15849.46   branch LIVE, multi-second field.
                                             not yet run

A vs B is what eeg_normalization_ablation.json measured, and it varied the
input scale while holding the encoder fixed. B vs C varies the encoder while
holding the scale fixed, which is the comparison that actually tests whether
the raw waveform carries anything.
"""


# --------------------------------------------------------------------- blocks
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


class StudentSleepStagingModelV2(nn.Module):
    """
    Same shape of model as the shipped student - spectral stream, concat fusion,
    transformer context, linear head - with the temporal encoder replaced.

    Keeping everything else identical is the point: it makes the encoder the
    single changed variable against the stored student, so B vs C in ABLATION
    is a clean comparison rather than a confounded one.
    """

    def __init__(self, embed=64, heads=2, layers=2, dropout=0.2, n_tokens=4,
                 use_spectral=True, spec_dim=34, max_len=512,
                 fine_ch=(24, 48), coarse_ch=(24, 48), in_channels=1):
        super().__init__()
        self.use_spectral = use_spectral
        self.temporal_encoder = MultiScaleEpochEncoder(
            embed=embed, n_tokens=n_tokens, fine_ch=fine_ch, coarse_ch=coarse_ch,
            in_channels=in_channels)
        if use_spectral:
            self.spectral_encoder = nn.Linear(spec_dim, embed)
            self.fusion = nn.Sequential(nn.Linear(embed * 2, embed),
                                        nn.LayerNorm(embed), nn.GELU())
        self.positional_encoding = PositionalEncoding(embed, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=embed, nhead=heads, dim_feedforward=embed * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.cls = nn.Linear(embed, NUM_CLASSES)

    def forward(self, xt, xs=None, mask=None):
        if isinstance(xt, (tuple, list)):
            xt, xs = xt[0], (xs if xs is not None else xt[1])
        h = self.temporal_encoder(xt)
        if self.use_spectral:
            if xs is None:
                raise ValueError("use_spectral=True but no spectral input given")
            h = self.fusion(torch.cat([h, self.spectral_encoder(xs)], -1))
        h = self.positional_encoding(h)
        return self.cls(self.encoder(h, src_key_padding_mask=mask))


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------- receptive field
@torch.no_grad()
def _zero_grads(m):
    for p in m.parameters():
        p.grad = None


def measured_receptive_field(encoder: nn.Module, length: int = EPOCH_SAMPLES) -> int:
    """
    Empirical receptive field: how many input samples the centre output actually
    depends on, found by backpropagating from one output position and counting
    non-zero input gradients.

    Measured rather than derived, because a derivation can be wrong in exactly
    the way that produced the inert-branch bug - a number that looks right and
    describes something the code does not do.
    """
    encoder = encoder.eval()
    x = torch.randn(1, 1, length, requires_grad=True)

    # reach inside to the pre-pool feature map, so the global pool of the OLD
    # encoder does not trivially report "everything"
    if isinstance(encoder, MultiScaleEpochEncoder):
        feats = torch.cat([encoder.fine(x), encoder.coarse(x)], dim=1)
    else:
        feats = encoder(x)

    mid = feats.shape[-1] // 2
    feats[0, :, mid].sum().backward()
    g = x.grad.detach().abs().squeeze()
    nz = torch.nonzero(g > 0).flatten()
    return int(nz.max() - nz.min() + 1) if len(nz) else 0


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print(__doc__.split("USAGE")[0].strip()[:0] or "", end="")
    print("=" * 78)
    print("RECEPTIVE FIELD, MEASURED BY BACKPROPAGATION (not derived)")
    print("=" * 78)

    # the encoder currently in use, lifted out of the shipped student
    class OldPyramid(nn.Module):
        def __init__(self, hid=64, dil=(1, 4)):
            super().__init__()
            self.branches = nn.ModuleList(
                [nn.Conv1d(1, hid, 7, dilation=d, padding=3 * d) for d in dil])
            self.proj = nn.Conv1d(hid, hid, 1)

        def forward(self, x):
            return self.proj(sum(F.relu(b(x)) for b in self.branches))

    old_rf = measured_receptive_field(OldPyramid())
    new = MultiScaleEpochEncoder()
    new_rf = measured_receptive_field(new)
    fine_rf = measured_receptive_field(
        type("F", (nn.Module,), {"forward": lambda s, x: new.fine(x),
                                 "__init__": lambda s: nn.Module.__init__(s)})())
    coarse_rf = measured_receptive_field(
        type("C", (nn.Module,), {"forward": lambda s, x: new.coarse(x),
                                 "__init__": lambda s: nn.Module.__init__(s)})())

    print(f"{'encoder':<34}{'samples':>10}{'seconds':>10}")
    print(f"{'current (atrous k=7, dil 1,4)':<34}{old_rf:>10}{old_rf / SAMPLE_RATE:>10.2f}")
    print(f"{'new: fine branch':<34}{fine_rf:>10}{fine_rf / SAMPLE_RATE:>10.2f}")
    print(f"{'new: coarse branch':<34}{coarse_rf:>10}{coarse_rf / SAMPLE_RATE:>10.2f}")
    print(f"{'new: combined':<34}{new_rf:>10}{new_rf / SAMPLE_RATE:>10.2f}")

    print(f"\nevents the model must distinguish, and whether they fit:")
    for name, lo, hi, why in (
            ("sleep spindle", 0.5, 2.0, "N2 vs N1"),
            ("K-complex", 0.5, 1.5, "N2 vs N1"),
            ("slow wave (0.5-2 Hz)", 0.5, 2.0, "N3"),
            ("sawtooth wave", 1.0, 3.0, "REM")):
        o = "yes" if old_rf / SAMPLE_RATE >= lo else "NO"
        n = "yes" if new_rf / SAMPLE_RATE >= hi else "partial"
        print(f"  {name:<24}{lo:>4.1f}-{hi:<4.1f}s  {why:<10} current: {o:<4} new: {n}")

    print(f"\n{'=' * 78}\nPARAMETERS\n{'=' * 78}")
    from student_model import StudentSleepStagingModel as OldStudent  # noqa: E402

    old_enc = OldStudent().temporal_encoder
    print(f"{'component':<40}{'params':>10}")
    print(f"{'current temporal encoder':<40}{count_parameters(old_enc):>10,}")
    print(f"{'new temporal encoder':<40}{count_parameters(new):>10,}")
    print(f"{'  fine branch':<40}{count_parameters(new.fine):>10,}")
    print(f"{'  coarse branch':<40}{count_parameters(new.coarse):>10,}")
    print(f"{'  projection':<40}{count_parameters(new.proj):>10,}")

    print(f"\n{'config':<44}{'params':>10}{'vs teacher':>12}")
    print(f"{'TEACHER':<44}{TEACHER_PARAMS:>10,}{'1.00x':>12}")
    print(f"{'shipped student (M0)':<44}{121099:>10,}{TEACHER_PARAMS / 121099:>11.2f}x")
    for label, kw in (
            ("V2 n_tokens=1, ch (24,48)", dict(n_tokens=1)),
            ("V2 n_tokens=4, ch (24,48)", dict(n_tokens=4)),
            ("V2 n_tokens=4, ch (16,32)", dict(n_tokens=4, fine_ch=(16, 32),
                                               coarse_ch=(16, 32))),
            ("V2 n_tokens=8, ch (24,48)", dict(n_tokens=8)),
            ("V2 + EOG      (2 channels)", dict(n_tokens=4, in_channels=2)),
            ("V2 + EOG + Pz-Oz (3 channels)", dict(n_tokens=4, in_channels=3)),
    ):
        m = StudentSleepStagingModelV2(**kw)
        n = count_parameters(m)
        print(f"{label:<44}{n:>10,}{TEACHER_PARAMS / n:>11.2f}x")

    print(f"\n{'=' * 78}\nFORWARD PASS\n{'=' * 78}")
    m = StudentSleepStagingModelV2()
    xt = torch.randn(2, 16, EPOCH_SAMPLES) * (1.0 / EEG_SCALE) * EEG_SCALE
    xs = torch.randn(2, 16, 34)
    out = m(xt, xs)
    print(f"  (2,16,3000) + (2,16,34) -> {tuple(out.shape)}  expect (2, 16, 5)")
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[0, 10:] = True
    print(f"  padding mask accepted    -> {tuple(m(xt, xs, mask).shape)}")
    m2 = StudentSleepStagingModelV2(use_spectral=False)
    print(f"  use_spectral=False       -> {tuple(m2(xt).shape)}")
    m3 = StudentSleepStagingModelV2(in_channels=2)
    xt2 = torch.randn(2, 16, 2, EPOCH_SAMPLES)
    print(f"  (2,16,2,3000) EEG+EOG    -> {tuple(m3(xt2, xs).shape)}")
    try:
        m3(xt, xs)
    except ValueError as e:
        print(f"  channel mismatch rejected: {str(e)[:52]}")

    loss = out.sum()
    loss.backward()
    n_no_grad = sum(1 for p in m.parameters() if p.requires_grad and p.grad is None)
    print(f"  parameters with no gradient: {n_no_grad}  (expect 0)")

    print(ABLATION)
