"""
Derive kaggle_train_teacher_v2.py from kaggle_train_teacher_norm.py.

WHY THIS EXISTS
---------------
The teacher has the SAME defect the student had. Measured by backpropagation
in student_encoder.py:

    student encoder   Conv1d(k=7, dil 1,4)      RF 25 samples = 0.25 s
    TEACHER encoder   Conv1d(k=7, dil 1,2,4,8)  RF 49 samples = 0.49 s

0.49 s is twice the student's and still below the 0.5-2 s a sleep spindle
occupies, and it is followed by the same AdaptiveAvgPool1d(1) over all 3000
positions. Replacing that encoder in the student took validation macro-F1 from
0.6821 to 0.7236 and N1 F1 from 0.431 to 0.523. The teacher has 4.6x the
capacity to exploit the same fix.

WHY THE TEACHER SPECIFICALLY MATTERS
------------------------------------
Distillation currently COSTS accuracy in this project, because the student
already beats every honest teacher:

    student_baseline_E0    test kappa 0.6449   121K params
    teacher_E1b            test kappa 0.6055   649K params

`fig8_rq3` extrapolates that soft targets only stop costing anything once the
teacher reaches roughly kappa 0.75. The N2multiscale student widened that gap
rather than closing it - it now beats teacher_E1b by +0.0259 validation
macro-F1. The teacher's encoder is the one lever that could close it, which
makes this the run that decides whether soft labels are ever worth using here.

WHAT IT CHANGES
---------------
  1. the module docstring
  2. EXPERIMENT      "E1b" -> "E1bmulti"   (a new ladder rung, EPOCH_TOKENS=4)
  3. OUT_DIR         a separate directory, so no stored teacher is overwritten
  4. the MODEL section: AdaptiveAtrousPyramid and EpochEncoder are replaced by
     MultiScaleEpochEncoder, copied verbatim out of student_encoder.py. An
     `EpochEncoder(embed_dim, n_tokens)` shim keeps FusedSleepStagingModel's
     call site unchanged.

EEG_SCALE, splits, seed, schedule, the focal loss, OneCycleLR, the dataset, the
evaluation and the zeroed-EEG ablation check are untouched.

ENCODER WIDTH IS DELIBERATELY NOT RAISED
----------------------------------------
The teacher could afford wider branches - it is not the deployed model and has
no compression budget to defend. It is kept at the student's (24, 48) anyway so
that this run changes the encoder FAMILY and nothing else, and remains a clean
A/B against E1bnorm. Widening is a separate experiment; run it second, not
mixed in, or a gain cannot be attributed.

USAGE
    python distillation/make_teacher_v2.py            # writes the trainer
    python distillation/make_teacher_v2.py --check    # verify it is up to date
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "kaggle_train_teacher_norm.py"
DST = HERE / "kaggle_train_teacher_v2.py"
ENC = HERE / "student_encoder.py"

MODEL_START = "# ------------------------------------------------------------------- MODEL ---"
MODEL_END = "class PositionalEncoding(nn.Module):"

DOCSTRING = '''"""
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
"""'''


def extract_encoder_source() -> str:
    """Lift the encoder classes verbatim out of student_encoder.py."""
    src = ENC.read_text(encoding="utf-8")
    out = []
    for name in ("ConvBlock", "MultiScaleEpochEncoder"):
        m = re.search(rf"^class {name}\b.*?(?=\n\n\nclass |\n\n\ndef |\Z)",
                      src, re.S | re.M)
        if not m:
            raise SystemExit(f"Could not find class {name} in {ENC.name}")
        out.append(m.group(0).rstrip())
    return "\n\n\n".join(out)


SHIM = '''class EpochEncoder(nn.Module):
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
        return self.encoder(x)'''


def build() -> str:
    src = SRC.read_text(encoding="utf-8")

    # 1. docstring
    end = src.index('"""', src.index('"""') + 3) + 3
    src = DOCSTRING + src[end:]

    # 2. a new ladder rung, so EPOCH_TOKENS=4 arrives through the existing
    #    mechanism rather than as a loose override
    src, n1 = re.subn(
        r'^    "E4":  dict\(OVERLAP=192, ALPHA_POWER=0\.5, FIX_FOCAL_PT=True,  EPOCH_TOKENS=8\),$',
        '    "E4":  dict(OVERLAP=192, ALPHA_POWER=0.5, FIX_FOCAL_PT=True,  EPOCH_TOKENS=8),\n'
        '    # E1b recipe exactly, with the multi-scale encoder and 4 sub-epoch tokens.\n'
        '    "E1bmulti": dict(OVERLAP=192, ALPHA_POWER=0.5, FIX_FOCAL_PT=True, EPOCH_TOKENS=4),',
        src, flags=re.M)
    src, n2 = re.subn(r'^EXPERIMENT = "E1b"', 'EXPERIMENT = "E1bmulti"', src, flags=re.M)
    src, n3 = re.subn(r'^OUT_DIR        = "/kaggle/working/E1bnorm"',
                      'OUT_DIR        = "/kaggle/working/E1bmulti"', src, flags=re.M)
    if n1 != 1 or n2 != 1 or n3 != 1:
        raise SystemExit(f"config substitution failed (ladder {n1}, EXPERIMENT {n2}, "
                         f"OUT_DIR {n3}) - {SRC.name} has changed shape.")

    # 3. model section: AdaptiveAtrousPyramid + EpochEncoder -> multi-scale.
    #    SEBlock goes with them; nothing else references it.
    a, b = src.index(MODEL_START), src.index(MODEL_END)
    body = (f"{MODEL_START}\n"
            f"# Lifted verbatim from student_encoder.py by make_teacher_v2.py so the\n"
            f"# two cannot drift apart. Receptive field measured there, not derived:\n"
            f"# 875 samples = 8.75 s, against 49 samples = 0.49 s for the encoder\n"
            f"# this replaces.\n"
            f"{extract_encoder_source()}\n\n\n{SHIM}\n\n\n")
    src = src[:a] + body + src[b:]

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
    print(f"wrote {DST.name}  ({len(text.splitlines())} lines, from {SRC.name} at "
          f"{len(SRC.read_text(encoding='utf-8').splitlines())})")
    print("changed: docstring, ladder rung E1bmulti, EXPERIMENT, OUT_DIR, MODEL section")
    print("unchanged: EEG_SCALE, splits, seed, schedule, focal loss, OneCycleLR, "
          "dataset, evaluation, ablation check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
