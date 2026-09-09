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

CV_BLOCK = '\n\n# --------------------------------------------------------------- CV FOLDS ---\n# Roadmap item 0.1. Generated from distillation/cv_folds.json by\n# make_cv_folds.py; see distillation/test_cv_folds.py (35 assertions).\n#\n# CV_FOLD = None  -> the original 69/16 split. Identical behaviour to every\n#                    run before 29 Aug 2026, so existing checkpoints reproduce.\n# CV_FOLD = 0..4  -> that fold\'s 68/17 subject-level split.\n#\n# Only train and val move. "test" is never touched by any fold, and that is\n# asserted below rather than trusted.\n#\n# This is NOT nested CV: one 5-fold CV for selection, plus the frozen 15-subject\n# test split for the final report.\nCV_FOLD = None\n\n# Each fold\'s VALIDATION subjects. Train is derived as (train + val) minus\n# these, so the pool is stated once and cannot drift out of step with itself.\nCV_VAL_SUBJECTS = {\n    0: [\n        \'SC400\', \'SC403\', \'SC406\', \'SC410\', \'SC411\', \'SC417\', \'SC425\',\n        \'SC443\', \'SC447\', \'SC457\', \'SC463\', \'SC465\', \'SC467\', \'SC477\',\n        \'ST701\', \'ST713\', \'ST720\'\n    ],\n    1: [\n        \'SC404\', \'SC422\', \'SC428\', \'SC430\', \'SC441\', \'SC442\', \'SC445\',\n        \'SC450\', \'SC459\', \'SC471\', \'SC473\', \'SC481\', \'SC482\', \'ST708\',\n        \'ST719\', \'ST721\', \'ST722\'\n    ],\n    2: [\n        \'SC413\', \'SC415\', \'SC418\', \'SC427\', \'SC434\', \'SC435\', \'SC436\',\n        \'SC438\', \'SC440\', \'SC449\', \'SC454\', \'SC456\', \'SC476\', \'ST704\',\n        \'ST709\', \'ST711\', \'ST712\'\n    ],\n    3: [\n        \'SC408\', \'SC416\', \'SC419\', \'SC426\', \'SC431\', \'SC432\', \'SC444\',\n        \'SC446\', \'SC448\', \'SC455\', \'SC464\', \'SC470\', \'SC480\', \'ST706\',\n        \'ST707\', \'ST715\', \'ST717\'\n    ],\n    4: [\n        \'SC405\', \'SC407\', \'SC409\', \'SC412\', \'SC424\', \'SC433\', \'SC451\',\n        \'SC458\', \'SC460\', \'SC462\', \'SC466\', \'SC472\', \'SC475\', \'ST702\',\n        \'ST705\', \'ST718\', \'ST724\'\n    ],\n}\n\nif CV_FOLD is not None:\n    _pool = list(SPLITS["train"]) + list(SPLITS["val"])\n    assert len(_pool) == 85, f"CV pool should be 85 subjects, got {len(_pool)}"\n    _va = CV_VAL_SUBJECTS[CV_FOLD]\n    assert set(_va) <= set(_pool), "fold validation subjects must come from the pool"\n    assert not (set(_va) & set(SPLITS["test"])), "a CV fold may never contain a test subject"\n    SPLITS = {\n        "train": [s for s in _pool if s not in set(_va)],\n        "val":   list(_va),\n        "test":  SPLITS["test"],\n    }\n    # Suffix the experiment name so a fold can never write into the delivered\n    # model\'s directory, and folds can never overwrite each other. EXPERIMENT is\n    # defined above this block, so every out-dir expression picks this up.\n    EXPERIMENT = f"{EXPERIMENT}_cv{CV_FOLD}"\n    print(f"CV_FOLD={CV_FOLD}: train {len(SPLITS[\'train\'])} subj / "\n          f"val {len(SPLITS[\'val\'])} subj  (test untouched)")\n    print(f"  -> writes to student_{EXPERIMENT}; the un-suffixed directory is untouched")\n\n# ---------------------------------------------------------- COHORT SPLIT ---\n# Roadmap item 3.1. Generated from distillation/cohort_split.json by\n# make_cohort_split.py. Train on the SC cohort, test on the whole ST cohort:\n# different population (mild insomnia, temazepam) on different hardware.\n#\n# COHORT_SPLIT = False -> untouched.\n# COHORT_SPLIT = True  -> train 62 SC / val 16 SC / test 22 ST subjects.\n#\n# NOT comparable to the headline kappa: the main 15-subject test split is a\n# subset of both cohorts, so that number and this one measure different\n# populations. Expect this to be LOWER; the size of the drop is the finding.\n#\n# Do NOT rebalance SC\'s classes toward the dataset overall - the overall\n# includes ST, the test cohort, so it leaks, and the prior shift is part of\n# what is being measured (ST has ~2x the N3 and ~1/3 the wake).\nCOHORT_SPLIT = False\n\nCOHORT_TRAIN = [\n        \'SC400\', \'SC401\', \'SC402\', \'SC403\', \'SC404\', \'SC405\', \'SC406\',\n        \'SC407\', \'SC409\', \'SC410\', \'SC411\', \'SC412\', \'SC413\', \'SC414\',\n        \'SC416\', \'SC417\', \'SC418\', \'SC419\', \'SC420\', \'SC421\', \'SC422\',\n        \'SC424\', \'SC425\', \'SC426\', \'SC427\', \'SC428\', \'SC429\', \'SC430\',\n        \'SC431\', \'SC432\', \'SC434\', \'SC435\', \'SC436\', \'SC437\', \'SC438\',\n        \'SC440\', \'SC442\', \'SC443\', \'SC444\', \'SC445\', \'SC446\', \'SC447\',\n        \'SC450\', \'SC455\', \'SC456\', \'SC458\', \'SC459\', \'SC460\', \'SC461\',\n        \'SC463\', \'SC464\', \'SC465\', \'SC466\', \'SC467\', \'SC472\', \'SC473\',\n        \'SC474\', \'SC475\', \'SC476\', \'SC480\', \'SC481\', \'SC482\'\n]\nCOHORT_VAL = [\n        \'SC408\', \'SC415\', \'SC423\', \'SC433\', \'SC441\', \'SC448\', \'SC449\',\n        \'SC451\', \'SC452\', \'SC453\', \'SC454\', \'SC457\', \'SC462\', \'SC470\',\n        \'SC471\', \'SC477\'\n]\nCOHORT_TEST = [\n        \'ST701\', \'ST702\', \'ST704\', \'ST705\', \'ST706\', \'ST707\', \'ST708\',\n        \'ST709\', \'ST710\', \'ST711\', \'ST712\', \'ST713\', \'ST714\', \'ST715\',\n        \'ST716\', \'ST717\', \'ST718\', \'ST719\', \'ST720\', \'ST721\', \'ST722\',\n        \'ST724\'\n]\n\nif COHORT_SPLIT:\n    if CV_FOLD is not None:\n        raise SystemExit("Set either CV_FOLD or COHORT_SPLIT, not both - they are "\n                         "different experiments.")\n    assert not (set(COHORT_TRAIN) & set(COHORT_TEST)), "SC train leaked into the ST test set"\n    assert not (set(COHORT_VAL) & set(COHORT_TEST)), "SC val leaked into the ST test set"\n    assert all(s.startswith("ST") for s in COHORT_TEST), "the test cohort must be all ST"\n    assert all(s.startswith("SC") for s in COHORT_TRAIN + COHORT_VAL), "train/val must be all SC"\n    SPLITS = {"train": COHORT_TRAIN, "val": COHORT_VAL, "test": COHORT_TEST}\n    EXPERIMENT = f"{EXPERIMENT}_sc2st"\n    print(f"COHORT_SPLIT: train {len(COHORT_TRAIN)} SC / val {len(COHORT_VAL)} SC "\n          f"-> test {len(COHORT_TEST)} ST subjects (held out, never loaded here)")\n    print(f"  -> writes to student_{EXPERIMENT}")\n    print("  NOTE: not comparable to the headline kappa - different population.")\n\n# The CV reference for a hard-label run of this architecture, measured 30 Aug\n# 2026 over all five folds (results/cv_analysis.json). Use THIS, not\n# BASELINE_VAL_*, when judging anything run under CV_FOLD.\nCV_BASELINE_MACRO_F1 = 0.7419\nCV_BASELINE_KAPPA    = 0.7143\nCV_BASELINE_SD_KAPPA = 0.0193\n'
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

    # ---- CV folds (roadmap 0.1) ----
    # Injected after the SPLITS literal so a run can select a subject-level
    # fold. Inert unless CV_FOLD is set, which keeps prior runs reproducible.
    _splits_end = src.index('\n}\n', src.index('SPLITS = {')) + len('\n}\n')
    src = src[:_splits_end] + CV_BLOCK + src[_splits_end:]

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

    # The inherited strings describe the run this was derived FROM. Left alone
    # they make the log read as a failed assertion ("expect 121,099" beside
    # 139,606) and mislabel the experiment, which is how a correct result gets
    # pasted into a report looking like a broken one.
    src, n3 = re.subn(r'print\(f"  parameters: \{npar:,\}  \(expect 121,099\)"\)',
                      'print(f"  parameters: {npar:,}  '
                      '(N1norm was 121,099; the encoder costs +18,507)")', src)
    src, n4 = re.subn(r'EXPERIMENT \{EXPERIMENT\}  -  EEG normalisation A/B',
                      'EXPERIMENT {EXPERIMENT}  -  temporal encoder A/B', src)
    if n3 != 1 or n4 != 1:
        raise SystemExit(f"log-string substitution failed (params {n3}, header {n4}).")

    # GRADIENT-ACCUMULATION BUG, inherited from kaggle_train_student_norm.py.
    #
    #     loss = F.cross_entropy(..., reduction="sum") / nv    # already the
    #                                                         # micro-batch's share
    #                                                         # of the full-batch MEAN
    #     w = (nvc / nv)
    #     scaler.scale(loss * w).backward()                    # divides by nv AGAIN
    #
    # summing loss over chunks already gives total_sum / nv, which IS the
    # full-batch mean - linearity of the sum is the whole reason accumulation
    # works. The extra w is a second division.
    #
    # Measured against a true full-batch backward:
    #   balanced chunks    gradient norm exactly 0.5x correct (1/K, K=2 chunks)
    #   unbalanced chunks  cosine similarity 0.9916 - the DIRECTION is wrong,
    #                      because each chunk ends up weighted by nvc^2 rather
    #                      than nvc. AdamW normalises away a uniform rescale;
    #                      it cannot fix a direction.
    # The fixed form reproduces a full-batch backward to 2.5e-07 relative error.
    #
    # kaggle_train_student.py is NOT affected: distillation_loss returns a MEAN,
    # so multiplying by nvc/nv is correct there. The defect appeared when
    # _norm.py switched to reduction="sum" / nv and kept the weight.
    # The annotation is part of what gets replaced. Left behind, every generated
    # trainer carried a comment reading "KNOWN BUG - LEFT IN PLACE DELIBERATELY
    # ... not fixed here" directly above the FIXED code - which is worse than no
    # comment, since a reader would trust it over the line beneath it.
    _buggy = ('                # KNOWN BUG - LEFT IN PLACE DELIBERATELY.\n'
              '                # `loss` is already sum/nv, i.e. this chunk\'s share of the\n'
              '                # full-batch mean. Multiplying by nvc/nv divides by nv a second\n'
              '                # time: gradients come out 0.5x too small with balanced chunks,\n'
              '                # and MIS-DIRECTED (cosine 0.9916) when padding makes chunks\n'
              '                # uneven, since each chunk is then weighted by nvc^2 not nvc.\n'
              '                #\n'
              '                # Not fixed here because this file must keep reproducing the\n'
              '                # stored student_N1norm checkpoint. The fix is applied by\n'
              '                # make_v2_trainer.py to every generated trainer; run those.\n'
              '                w = (nvc/nv).float()\n'
              '                scaler.scale(loss*w).backward()\n'
              '                tot_loss += loss.item()*w.item()')
    _fixed = ("                # `loss` is ALREADY this chunk's share of the full-batch\n"
              "                # mean (its sum / nv). Summing over chunks therefore gives\n"
              "                # the full-batch mean exactly. Do NOT reweight it again.\n"
              "                scaler.scale(loss).backward()\n"
              "                tot_loss += loss.item()")
    if src.count(_buggy) != 1:
        raise SystemExit(f"gradient-accumulation fix did not apply "
                         f"({src.count(_buggy)} matches).")
    src = src.replace(_buggy, _fixed, 1)

    # Resolve the index EXPLICITLY, as the mc trainer does. find_file("index.csv")
    # takes the first sorted match, and with processed_sleepedf_mc also attached
    # that is ambiguous - it picks the right one here only because '/' (0x2F)
    # sorts before '_' (0x5F). Which dataset a run trains on should not depend on
    # a sort order, and a multi-channel index handed to this single-channel
    # trainer would otherwise be read without complaint.
    _old_idx = '    index = find_file("index.csv"); root = index.parent.parent'
    _new_idx = (
        '    index = find_file("processed_sleepedf/index.csv")\n'
        '    root = index.parent.parent\n'
        '    print(f"  index     {index}")\n'
        '    if "channels" in pd.read_csv(index, nrows=1).columns:\n'
        '        raise SystemExit(f"{index} is a MULTI-CHANNEL index, but this "\n'
        '                         f"trainer is single-channel.\\n"\n'
        '                         f"Attach processed_sleepedf, or run "\n'
        '                         f"kaggle_train_student_mc.py instead.")')
    if src.count(_old_idx) != 1:
        raise SystemExit(f"index-resolution fix did not apply ({src.count(_old_idx)}).")
    src = src.replace(_old_idx, _new_idx, 1)

    # Compare against BOTH bars. 0.6881 is student_baseline_E0 with the EEG
    # inert; 0.6821 is student_N1norm, which is the same EEG scale as this run
    # and therefore the comparison that isolates the encoder.
    src, n5 = re.subn(
        r'^BASELINE_VAL_KAPPA    = 0\.6389$',
        'BASELINE_VAL_KAPPA    = 0.6389\n\n'
        '# student_N1norm: SAME EEG_SCALE, old encoder. This is the bar that\n'
        '# isolates the encoder; the one above also carries the scaling change.\n'
        'N1NORM_VAL_MACRO_F1   = 0.6821\n'
        'N1NORM_VAL_KAPPA      = 0.6323', src, flags=re.M)
    # No backslashes inside the injected f-string expressions: escapes in an
    # f-string replacement field are a SyntaxError before Python 3.12, and
    # Kaggle is not on 3.12. Bind to locals first instead.
    n1norm_block = (
        '    _mf1, _kap = m["macro_f1"], m["kappa"]\n'
        '    print("\\n  vs student_N1norm (same EEG scale, old encoder) - "\n'
        '          "the bar that isolates the encoder:")\n'
        '    print(f"    macro-F1 {_mf1:.4f} vs {N1NORM_VAL_MACRO_F1:.4f}"\n'
        '          f"   ({_mf1 - N1NORM_VAL_MACRO_F1:+.4f})")\n'
        '    print(f"    kappa    {_kap:.4f} vs {N1NORM_VAL_KAPPA:.4f}"\n'
        '          f"   ({_kap - N1NORM_VAL_KAPPA:+.4f})")\n'
        '    better = m["macro_f1"] > BASELINE_VAL_MACRO_F1')
    src, n6 = re.subn(r'^ +better = m\["macro_f1"\] > BASELINE_VAL_MACRO_F1$',
                      n1norm_block.replace("\\", "\\\\"), src, flags=re.M)
    if n5 != 1 or n6 != 1:
        raise SystemExit(f"baseline-bar substitution failed ({n5}, {n6}).")

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
    # ---- cv_fold in the checkpoint and the resume guard (roadmap 0.1) ----
    _old_guard = ('        if ck.get("eeg_scale") != EEG_SCALE or ck.get("schedule_shape")'
                  ' != {"epochs": EPOCHS, "steps": steps}:\n')
    _new_guard = ('        if (ck.get("eeg_scale") != EEG_SCALE\n'
                  '                or ck.get("schedule_shape") != {"epochs": EPOCHS, "steps": steps}\n'
                  '                or ck.get("cv_fold") != CV_FOLD):\n')
    assert src.count(_old_guard) == 1, "resume guard not found"
    src = src.replace(_old_guard, _new_guard, 1)

    _old_msg = '                             f"(scale {ck.get(\'eeg_scale\')} -> {EEG_SCALE}). Delete {out}.")\n'
    _new_msg = ('                             f"(scale {ck.get(\'eeg_scale\')} -> {EEG_SCALE}, "\n'
                '                             f"cv_fold {ck.get(\'cv_fold\')} -> {CV_FOLD}). Delete {out}.")\n')
    assert src.count(_old_msg) == 1, "resume message not found"
    src = src.replace(_old_msg, _new_msg, 1)

    _old_state = '"n_parameters": npar, "eeg_scale": EEG_SCALE,'
    assert src.count(_old_state) == 1, "checkpoint state dict not found"
    src = src.replace(_old_state,
                      '"n_parameters": npar, "eeg_scale": EEG_SCALE, "cv_fold": CV_FOLD,', 1)

    # ---- the verdict is invalid under CV (measured 30 Aug) ----
    # BASELINE_VAL_* come from the original 16-subject val split, which the
    # 5-fold CV showed to be a pessimistic draw (CV mean macro-F1 0.7419 vs a
    # bar of 0.7249; all five folds cleared it). Comparing one fold to that bar
    # prints "BEATS the baseline" for free.
    _old_verdict = (
        '    better = m["macro_f1"] > BASELINE_VAL_MACRO_F1\n'
        '    print(f"\\n  VERDICT: {\'BEATS\' if better else \'DOES NOT BEAT\'} '
        'the baseline on validation.")\n')
    _new_verdict = (
        '    if CV_FOLD is not None:\n'
        '        print("\\n  NO VERDICT FROM ONE FOLD.")\n'
        '        print("    BASELINE_VAL_* were measured on the original 16-subject val")\n'
        '        print("    split, which the CV showed to be a pessimistic draw - every")\n'
        '        print("    fold clears it for free. A single fold means nothing on its own.")\n'
        '        print(f"    This fold: macro-F1 {m[\'macro_f1\']:.4f}, kappa {m[\'kappa\']:.4f}")\n'
        '        print("    Run all 5, then: python distillation/cv_summary.py")\n'
        '        print(f"    CV reference (hard labels): macro-F1 {CV_BASELINE_MACRO_F1:.4f}, "\n'
        '              f"kappa {CV_BASELINE_KAPPA:.4f}, across-fold sd {CV_BASELINE_SD_KAPPA:.4f}")\n'
        '    else:\n'
        '        better = m["macro_f1"] > BASELINE_VAL_MACRO_F1\n'
        '        print(f"\\n  VERDICT: {\'BEATS\' if better else \'DOES NOT BEAT\'} '
        'the baseline on validation.")\n')
    assert src.count(_old_verdict) == 1, "verdict block not found"
    src = src.replace(_old_verdict, _new_verdict, 1)


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
