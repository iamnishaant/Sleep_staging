
# ADDING EOG VERSION — ADVANCED STUDENT

**Written 10 October 2026.** This is part of the EOG testing work, and it is
separate from the shipped pipeline. **Nothing outside this folder was changed.**

**The goal:** improve REM and N1, and overall accuracy, with the final student
model kept **at or under 200,000 parameters**.

| File | What it is |
|---|---|
| `kaggle_train_student_advanced.py` | Kaggle trainer: 7 arms, two protocols, guards against data leakage |
| `summarise_cv.py` | Compares every arm with the `base` control over the cross-validation runs, and applies the success bar |
| `test_advanced_student.py` | 52 tests (CPU, ~10 s) |

---

## 1. What has already been tried, and what it taught

Every row below is a measured result already in this repository (mostly
`PROJECT_REPORT.md` and `distillation/results/`). The design in §3 builds on
these results and doesn't repeat what already failed.

| Technique | Result | Lesson |
|---|---|---|
| Focal loss + class weights (teacher E1a → E1b) | κ 0.51 → 0.61 | the loss mattered early on |
| Bigger teacher (649K parameters) | train κ 0.85, test 0.61 | **more parameters overfit** ~69 subjects |
| Distilling from a weaker teacher | −0.03 to −0.05 κ | a teacher must beat its student |
| **Multi-scale encoder** (0.25 s → 8.75 s receptive field) | **κ 0.645 → 0.699** | **the biggest gain: let the model *see* spindles and slow waves** |
| EOG + Pz-Oz, fused early | +0.003 κ, p = 0.87 | the error analysis shows REM +, Wake + but N2/N3 −; **the fusion method was the problem** |
| 3-model ensemble teacher → distilled N4kd | κ 0.700 (tie); calibration improved | distillation transfers calibration, not accuracy |
| Correcting class priors | +0.002 even with perfect priors | not the cause of the ST gap |
| Per-recording amplitude normalisation | **−0.018 κ** | amplitude defines N3; don't normalise it away |
| Sequence decoding (Viterbi, minimum-run rules) | ≤ +0.002 κ | the transformer already models sequences |
| N1 decision thresholds | ≤ +0.009 N1 F1 | N1 is limited by the input, not the threshold |

**What the error and ST analyses then measured on N4kd:**
- **REM:** errors sit *inside* REM periods (only 35% at an edge) and are
  confident; 59% go to N2. That's **missing information**, the kind a new
  signal can fix and smoothing can't.
- **N1:** 65% of errors are at stage edges, and N1 was the runner-up 90% of the
  time. These are **boundary and near-miss** errors, where experts disagree
  too.
- **The 34 hand-made spectral features have no sigma band (12–15 Hz).** Sleep
  spindles, the textbook N2-vs-REM marker, are split across "alpha 8–13" and
  "beta 13–30".
- **The old EOG fusion helped REM** (21/31 nights, p = 0.011) and Wake
  (+272 epochs), but **damaged N2 (−541) and N3 (−123).**
- **ST recordings** are ~1.8× SC amplitude; N4kd scores them best at ×0.7–0.85.
- **A preprocessing bug misaligns labels and signal** in 54 of 137 training and
  13 of 31 validation recordings (5.65% / 7.10% of labels wrong). It's confirmed
  against the raw files in `ADDING EOG VERSION - ST NIGHTS/`. Scored against
  corrected labels, the same N4kd gets **79.2%** validation accuracy instead of
  76.1%.
- **Stuck-signal epochs** occur in 29 of 30 ST training recordings; N4kd is
  right on only 21.6% of them.

**The conclusion that shapes the design:** more parameters and more
post-processing are exhausted. The gains left come from **better input views**
(spindles, eye movements), **local boundary context**, and **cleaner training
data**.

## 2. Should the model just be bigger?

No.
- **The evidence:** the 649K teacher overfit, and the 139K, 151K and ~410K
  models all sit within one confidence interval.
- **The budget is used deliberately instead:** about 36K extra parameters, each
  block aimed at a measured error.

| Arm | Parameters |
|---|---:|
| `base` | 139,606 (= N4kd) |
| `full_eeg` | 160,511 |
| `full` | **175,599** (≤ 200K, enforced in the code) |

## 3. The design — each component aimed at a measured error

| Component | What it is | Aimed at | Params |
|---|---|---|---:|
| **TF** — time-frequency branch | STFT per epoch (2 s window, 1 s hop, 0.5 Hz bins, 0.5–30 Hz → 60 bins), log power, small CNN, gated | REM↔N2 (**spindles at 12–15 Hz**), N1↔W (alpha), REM (theta) | 16,168 |
| **EOG** — eye-movement branch | separate encoder: 0.5 s saccade + 4 s slow-roll kernels + 8 features, gated, Pz-Oz dropped | REM (rapid eye movements), Wake; built not to touch N2/N3 | 15,088 |
| **LOCAL** — local context | depthwise conv over neighbouring epochs (k = 7), residual, before the transformer | N1 / boundary errors | 4,672 |
| **AUX** — transition head | predicts "the stage changes here", loss weight 0.1, unused at inference | boundaries | 65 |
| **SMOOTH** — boundary label smoothing | at a change epoch, the hard target gives 0.2 to the neighbouring stage | N1 near-misses (expert uncertainty) | 0 |
| **QMASK** — quality mask | flat or stuck (≥ 1 s identical) epochs left out of the *loss* (never out of evaluation) | ST recorder faults | 0 |
| **GAIN** — amplitude augmentation | ×0.75–1.33 per sequence, spectral features moved consistently, + 2% noise | ST amplitude shift, without normalising | 0 |
| **EMA** — weight averaging | decay 0.999 with warm-up | stability | 0 |

**Every new path is closed at the start:** projections are zero-initialised and
gates start at 0.018.
- **Tested:** with every component on, warm-started from the real N4kd
  checkpoint, the model's output is **bit-identical to N4kd's.** Each component
  has to earn its influence during training.
- **Modality dropout:** TF and EOG are each switched off for 25% of training
  sequences, so the model never depends on one entirely.

## 4. Arms and protocols

| Arm | Components | Why it's there |
|---|---|---|
| `base` | none | the control: N4kd's architecture and recipe |
| `recipe` | SMOOTH AUX QMASK GAIN EMA | is it the training recipe alone? |
| `tf` | TF | the spindle/theta view alone |
| `eog` | EOG | eye movements alone |
| `local` | LOCAL | local context alone |
| **`full_eeg`** | TF LOCAL + recipe | **the best candidate for an EEG-only device** |
| `full` | EOG TF LOCAL + recipe | everything; needs an EOG electrode |

**Protocol A — which components help?** `--cv-fold 0..4`. This forces
`ALPHA = 1.0` and no warm start: the teacher and N4kd were both trained on the
original split, which contains most of each fold's validation subjects.
**Protocol B — the production candidate:** original split, ENSmc teacher,
warm start from N4kd (the defaults). Run it only for the arm Protocol A
selects.

**The success bar, fixed in advance (`summarise_cv.py`):** an arm *helps* if,
paired by fold and seed against `base`, mean Δκ ≥ +0.01 **and** the 95% CI is
above zero. N1 and REM changes are reported for every arm.

## 5. Run plan (Kaggle, GPU T4 x2)

Attach the same datasets as the EOG trainer: `processed_sleepedf_mc`,
`processed_sleepedf` (spectral), and for Protocol B also `teacher_logits_ENSmc`
and `student_N4kd`.

**Stage 1 — 15 runs, about 9–11 GPU-hours:**
```python
for fold in range(5):
    for arm in ("base", "full_eeg", "full"):
        !python "{script}" --arm {arm} --seed 0 --cv-fold {fold}
```
Then run `python summarise_cv.py <folder of result files>`.

**Stage 2 — only if `full_eeg` or `full` helps:** run the single-component
arms (`tf`, `local`, `recipe`, `eog`) to see which parts carry the gain. Add
seeds 1–2 if the effect is borderline.

**Use the corrected labels (recommended for every new run).**
1. Upload `ADDING EOG VERSION - ST NIGHTS/results/corrected_labels_train_val.json`
   as a small Kaggle dataset.
2. Add `--corrected-labels` to **every** run, `base` included:
   ```python
   for fold in range(5):
       for arm in ("base", "full_eeg", "full"):
           !python "{script}" --arm {arm} --seed 0 --cv-fold {fold} --corrected-labels
   ```

Train and validation then use the true labels. "?" marks never-scored epochs,
which are ignored. The signal data is unchanged. The summariser pairs
`_fixlabels` runs only with a `_fixlabels` base. These numbers are **not**
comparable with stored-label runs; compare `base_fixlabels` with N4kd's
re-scored 79.2%.

`--exclude-misaligned` (leave out 10 recordings) is the older, cruder option.
Prefer `--corrected-labels`.

**Stage 3:** run Protocol B for the chosen arm, then a single, final test
evaluation, written only then.

**What each run prints:**
- per-class F1 every epoch, plus the mean gate value of TF and EOG;
- at the end, the same weights re-scored with **TF off** and **EOG off**, stage
  by stage. That shows exactly what each branch contributes.

## 6. What to expect

These are estimates, not results.

| | Expected change |
|---|---|
| REM F1 | **+2 to +5 points**, from TF (spindles) and EOG |
| N1 F1 | **+1 to +3 points** (it stays capped by expert agreement, ~63%) |
| Accuracy | **+1 to +2 points** |

**Separately, the label fix:** scored against corrected labels, the existing
N4kd is already at **79.2% validation accuracy** (κ 0.722, REM F1 80.2, N1 53.8),
with no retraining. Training on corrected labels (5.65% of training labels are
wrong today) should add to that. **`base --corrected-labels` measures exactly
how much.**

**Reaching 80%:** with correct labels, validation is already at 79.2%.
Retraining on corrected labels, plus the best arm here, is a realistic route
past 80%. The test set's labels are also affected (13 of 29 recordings); see
the ST NIGHTS README.

## 7. Caveats

- **Deployment:** the `full` arm needs an EOG electrode. `full_eeg` works on
  today's single-electrode setup.
- **Downstream:** promoting any new staging model changes every evidence packet,
  the report model's training data and `deploy/frozen.json`. Treat it as a v2
  track.
- **The misaligned list and the corrected labels:** the list
  (`--exclude-misaligned`) came from model-vs-label agreement. The corrected
  labels come from the raw hypnograms, with the preprocessing reproduced on
  197/197 recordings, so they are exact.

## 8. Verification done

- **52 tests pass.** Among them:
  - `base` is exactly N4kd: 139,606 parameters, same keys;
  - every arm stays within 200K, and the budget is enforced;
  - every arm, untrained, equals `base` exactly; the warm-started `full` model
    **equals the real N4kd** exactly;
  - each component changes the output once opened, and switching branches off
    recovers the backbone exactly;
  - the local block ignores padded content;
  - the STFT resolves a 13 Hz tone in the sigma band;
  - boundary smoothing reduces to N4kd's loss at ε = 0;
  - the quality mask flags 1.00 s stuck runs but not 0.98 s;
  - the gain transform follows the DWT algebra;
  - EMA warm-up works as designed;
  - frozen training steps move no backbone weight;
  - the guards and the summariser behave as specified;
  - "?" labels are ignored, and the corrected-labels file covers exactly the
    train and validation subjects, never test.
- **End-to-end CPU smoke runs** of `full` (warm start + teacher), `full_eeg`
  (a CV fold) and `base`, plus `--exclude-misaligned` and `--corrected-labels`. They exercised training,
  the freeze, EMA, the auxiliary loss, branch-off evaluation and the result
  files.

```
cd "ADDING EOG VERSION - ADVANCED STUDENT"
python -m pytest -q -p no:cacheprovider test_advanced_student.py
```
