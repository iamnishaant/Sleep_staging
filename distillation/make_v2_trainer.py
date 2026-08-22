"""
Derive kaggle_train_student_v2.py from kaggle_train_student_norm.py.

WHY DERIVE RATHER THAN WRITE
----------------------------
The claim the next experiment has to support is "the ENCODER is the only thing
that changed". A hand-written 460-line trainer cannot support that claim - a
stray hyperparameter, a reordered seed, a different early-stop patience, and
the comparison silently becomes a comparison of two pipelines. Deriving it
mechanically means the untouched regions are byte-identical by construction,
and this script is the record of exactly which regions were touched.

The same reasoning produced kaggle_train_teacher_norm.py, which
eeg_normalization_ablation.json records as "derived programmatically from
kaggle_train_improved.py".

WHAT IT CHANGES
---------------
  1. the module docstring
  2. EXPERIMENT       "N1norm" -> "N2multiscale"
  3. N_TOKENS         1 -> 4          (attention-pooled sub-epoch tokens)
  4. the MODEL section: the atrous pyramid and EpochEncoder are replaced by
     MultiScaleEpochEncoder, copied verbatim out of student_encoder.py so the
     two cannot drift apart.

EEG_SCALE, splits, seed, schedule, batch size, class weighting, early stopping,
the dataset, the evaluation and the zeroed-EEG ablation check are all untouched.

USAGE
    python distillation/make_v2_trainer.py            # writes the trainer
    python distillation/make_v2_trainer.py --check    # verify it is up to date
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "kaggle_train_student_norm.py"
DST = HERE / "kaggle_train_student_v2.py"
ENC = HERE / "student_encoder.py"

MODEL_START = "# ------------------------------------------------------------------- MODEL ---"
MODEL_END = "# -------------------------------------------------------------------- DATA ---"

DOCSTRING = '''"""
================================================================================
KAGGLE NOTEBOOK - MULTI-SCALE TEMPORAL ENCODER  (experiment "N2multiscale")
================================================================================
A/B TEST OF ONE CHANGE against student_N1norm. Same data, splits, schedule,
class weighting, seed 42 and EEG_SCALE. The temporal encoder is replaced.

WHY
---
eeg_normalization_ablation.json concluded that the raw EEG carries no
predictive information beyond the 34 spectral features, because normalising it
(which revived a provably inert branch) moved validation macro-F1 by -0.0060.
That experiment held the ARCHITECTURE fixed, and the architecture is the
binding constraint.

Measured by backpropagation in distillation/student_encoder.py, the encoder
used for that result has a receptive field of 25 samples - 0.25 SECONDS at
100 Hz - followed by AdaptiveAvgPool1d(1), which averages all 3000 positions
into one vector. It computes the mean, over 30 seconds, of a quarter-second
texture detector.

Every event that defines the stages this model is worst at is longer than that:

    sleep spindle   0.5 - 2 s    defines N2 against N1
    K-complex       >= 0.5 s     defines N2 against N1
    slow wave       0.5 - 2 s    defines N3
    sawtooth wave   1 - 3 s      marks REM

So a band-power vector beat the encoder because it summarises 30 seconds of
structure while the encoder summarised 120 repetitions of a quarter-second.
The -0.0060 is evidence about that encoder, not about the EEG.

THE CHANGE
----------
Two convolutional branches at deliberately different time scales:

    fine    Conv1d(kernel=50,  stride=6)    0.5 s window   -> spindles, K-complexes
    coarse  Conv1d(kernel=200, stride=25)   2.0 s window   -> slow waves, delta

Measured receptive field 875 samples = 8.75 s, against 0.25 s. Output is
attention-pooled to N_TOKENS=4 positions per epoch rather than 1, so
within-epoch structure survives into the transformer. N_TOKENS was always
supported and never switched on.

Cost: temporal encoder 10,182 -> 28,689 params; student 121,099 -> 139,606.
Compression against the teacher 5.36x -> 4.65x. If the encoder does not earn
that, it should be reverted - say so rather than keeping it for its own sake.

WHAT SUCCESS LOOKS LIKE
-----------------------
    beat  : validation macro-F1 0.6821, kappa 0.6323   (student_N1norm - the
            SAME encoder question with the SAME EEG scale, so it is the honest
            comparison)
    watch : 0.6881 / 0.6389 (student_baseline_E0, EEG inert) is the number to
            beat before claiming the raw waveform contributes anything at all

Its held-out TEST kappa 0.6449 must NOT be used to decide anything here.

This script re-runs the zeroed-EEG ablation on its own checkpoint at the end.
Agreement well below 1.0 means the branch is live. If macro-F1 still does not
improve with a live branch AND a multi-second receptive field, that is a real
result about the EEG rather than about the encoder - report it, do not tune
around it.

SETUP
-----
1. Attach the sleepedf dataset (processed_sleepedf/...).
2. Accelerator -> GPU T4 x2 (single GPU is used; P100 has no kernels here).
3. Run. ~3 h. Resumes from /kaggle/working if the session dies.

GENERATED FILE - do not edit by hand.
Regenerate with: python distillation/make_v2_trainer.py
================================================================================
"""'''


def extract_encoder_source() -> str:
    """Lift the encoder classes verbatim out of student_encoder.py."""
    src = ENC.read_text(encoding="utf-8")
    out = []
    for name in ("ConvBlock", "MultiScaleEpochEncoder", "PositionalEncoding"):
        m = re.search(rf"^class {name}\b.*?(?=\n\n\nclass |\n\n\ndef |\Z)",
                      src, re.S | re.M)
        if not m:
            raise SystemExit(f"Could not find class {name} in {ENC.name}")
        out.append(m.group(0).rstrip())
    return "\n\n\n".join(out)


STUDENT_V2 = '''class StudentSleepStagingModel(nn.Module):
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
        self.temporal_encoder = MultiScaleEpochEncoder(embed=embed, n_tokens=n_tokens)
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
        return self.cls(self.encoder(self.positional_encoding(h), src_key_padding_mask=mask))'''


def build() -> str:
    src = SRC.read_text(encoding="utf-8")

    # 1. docstring
    end = src.index('"""', src.index('"""') + 3) + 3
    src = DOCSTRING + src[end:]

    # 2/3. config
    src, n1 = re.subn(r'^EXPERIMENT\s*=\s*"N1norm"',
                      'EXPERIMENT     = "N2multiscale"', src, flags=re.M)
    src, n2 = re.subn(r'^N_TOKENS\s*=\s*1\b',
                      'N_TOKENS       = 4', src, flags=re.M)
    if n1 != 1 or n2 != 1:
        raise SystemExit(f"config substitution failed (EXPERIMENT {n1}, N_TOKENS {n2}) "
                         f"- {SRC.name} has changed shape; update this script.")

    # 4. model section
    a, b = src.index(MODEL_START), src.index(MODEL_END)
    body = (f"{MODEL_START}\n"
            f"# Lifted verbatim from student_encoder.py by make_v2_trainer.py so the\n"
            f"# two cannot drift apart. Receptive field measured there, not derived.\n"
            f"{extract_encoder_source()}\n\n\n{STUDENT_V2}\n\n\n")
    src = src[:a] + body + src[b:]

    # the config block the checkpoint records, so a stray checkpoint is identifiable
    src = src.replace('"use_spectral": USE_SPECTRAL,',
                      '"use_spectral": USE_SPECTRAL, "encoder": "multiscale",\n'
                      '                   "n_tokens": N_TOKENS,', 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Exit non-zero if the generated file is stale.")
    args = ap.parse_args()

    text = build()
    if args.check:
        if not DST.exists():
            print(f"{DST.name} does not exist. Run without --check.")
            return 1
        stale = DST.read_text(encoding="utf-8") != text
        print(f"{DST.name}: {'STALE - regenerate' if stale else 'up to date'}")
        return 1 if stale else 0

    DST.write_text(text, encoding="utf-8")
    n_src = len(SRC.read_text(encoding="utf-8").splitlines())
    n_dst = len(text.splitlines())
    print(f"wrote {DST.name}  ({n_dst} lines, from {SRC.name} at {n_src})")
    print("changed: docstring, EXPERIMENT, N_TOKENS, the MODEL section, "
          "checkpoint config tag")
    print("unchanged: EEG_SCALE, splits, seed, schedule, class weighting, "
          "dataset, evaluation, ablation check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
