# ADDING EOG VERSION — ST NIGHTS

**Written 10–11 October 2026.** This is part of the EOG testing work, and it is
separate from the shipped pipeline.

**The question:** three validation nights (ST7201J0, ST7202J0, ST7151J0) scored
κ ≈ 0.1 for *every* model. Why?

**The answer: a label-alignment bug in the preprocessing.** It affects about a
third of all recordings, not just ST:
- train: **54 of 137** recordings, with **5.65%** of training labels wrong;
- validation: **13 of 31**, with **7.10%** wrong;
- test: **13 of 29**, with **5.17%** wrong.

Re-scored against corrected labels, the deployed N4kd's validation accuracy is
**79.2%, not 76.1%.**

**What this folder touches:**
- It **only reads** existing files and the raw Sleep-EDF files in
  `F:\sleep edf signals\`.
- It writes **only inside this folder**.
- Nothing is trained, and no existing file is changed.
- Corrected labels are written for **train and validation only**.

---

## 1. The bug

`code/Phase1_48/Phase1reworked/preprocess.py :: extract_epochs_and_trim`
(copied unchanged into `distillation/preprocess_multichannel.py :: epoch_bounds`):

```python
labels = labels[labels != "INVALID"]        # unscored epochs REMOVED from the labels
start_ep, end_ep = ...                      # trim points computed on the SHORTENED labels
data = raw[start_ep * spe : end_ep * spe]   # ...but applied to the FULL signal
```

**What goes wrong:**
- Every unscored epoch is dropped from the label list but stays in the signal.
  Unscored epochs include an unscored lead-in, "Movement time" and
  "Sleep stage ?".
- So after each one, the labels drift one more epoch **ahead** of the signal.
- Signal and label lengths still match, so the pipeline's own check
  (`assert epochs.shape[0] == len(labels)`) can't catch it.

**How it was established, in three independent steps:**
1. **Offset scan** (`investigate_st_nights.py`, section A7). Sliding each
   recording's labels against N4kd's predictions, 14 recordings agree far better
   at a negative offset. ST7191J0 goes from κ 0.11 to 0.77 at −72 epochs.
2. **Code audit** (`verify_hypnogram_alignment.py`). It rebuilds the labels from
   the raw hypnograms with the preprocessing's exact rules, and **reproduces the
   stored labels on all 197 of 197 recordings.** From there it computes each
   epoch's true offset.
3. **The two agree:** ST7191J0 72 vs scan −72; ST7182J0 66 vs −66; ST7202J0 44
   vs −44; ST7201J0 47 vs −48. The offset equals the number of unscored epochs
   at the start of the hypnogram.

**A correction to my earlier note.** I first suggested the hypnogram files might
start later than their recordings, with the preprocessing using
hypnogram-relative times. The raw files refute that: the ST hypnograms store no
separate start time, and their onsets are already relative to the recording.
**The real cause is the drop-then-cut step above.**

## 2. How much it matters

**The same N4kd predictions, scored against corrected labels** (validation;
never-scored epochs dropped):

| | Accuracy | κ | Macro F1 | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stored labels (as reported so far) | 76.11% | 0.681 | 72.5 | 88.8 | 50.4 | 78.2 | 69.0 | 76.2 |
| **Corrected labels** | **79.18%** | **0.722** | **76.0** | 90.1 | **53.8** | 81.0 | **74.8** | **80.2** |
| SC nights: stored → corrected | 79.8 → 80.3% | 0.729 → 0.736 | | | | | | |
| **ST nights: stored → corrected** | **62.4 → 74.8%** | **0.476 → 0.645** | | 59.5 → 67.6 | 39.3 → 54.9 | | | 67.2 → 82.7 |

**What it means:**
- **The model is better than reported.** About 3 accuracy points, +4 REM F1 and
  +3 N1 F1 were lost to the answer key, not to the model.
- **Much of the "SC→ST cohort gap" was this bug.** `PROJECT_REPORT.md` §6h–6i
  attributed it to covariate shift. That conclusion should be revisited.
- **Training was also on wrong labels** (5.65%), and so was the ENSmc teacher.
  Retraining on corrected labels is the obvious next improvement.
- **Test:** 13 of 29 recordings are affected, so the reported test result
  (77.7% / κ 0.6993) is very likely an underestimate. **The test labels were not
  corrected here,** because the locked set should only change by deliberate
  decision.

## 3. The other findings, which still hold

**ST7151J0 is also a genuinely poor recording.** Its labels are off by 226 epochs
(113 min), and 26% of its epochs are flat or stuck.

**Stuck-signal segments are an ST recorder fault.**
- 29 of 30 ST training recordings contain epochs with at least 1 s of identical
  samples; no SC recording does.
- N4kd is right on only 21.6% of them.
- So: exclude them from training loss, and flag them at inference.

**Amplitude is not the cause.** ST recordings are ~1.8× SC amplitude.
- Scaling the bad nights' amplitude anywhere from ×0.5 to ×2 leaves them at
  κ ≈ 0.1.
- Full per-recording normalisation hurts overall (κ 0.681 → 0.668).
- **Conclusion:** mild amplitude augmentation helps; normalising amplitude away
  does not.

## 4. Files

| File | What it is |
|---|---|
| `verify_hypnogram_alignment.py` | **The label audit.** `python verify_hypnogram_alignment.py "F:/sleep edf signals/physionet.org/files/sleep-edfx/1.0.0"` (about 1 min). Writes `results/label_alignment_audit.csv` and `results/corrected_labels_train_val.json` |
| `rescore_with_corrected_labels.py` | Re-scores N4kd's cached validation predictions against the corrected labels → `results/rescore_corrected_labels.json` |
| `investigate_st_nights.py` | Signal statistics, recording quality, the offset scan and the amplitude counterfactual → `results/ST_NIGHTS_REPORT.md` |
| `test_investigate_st_nights.py` | 4 tests |
| `figures/` | EEG spectra by group; κ against amplitude gain |

## 5. What to do

1. **Use the corrected labels for every new training run.** The advanced trainer
   supports it: `--corrected-labels`, with `corrected_labels_train_val.json`
   attached as a Kaggle dataset.
2. **Decide about the preprocessing fix.** The one-line idea: compute the trim
   points on the *full* label array and mark unscored epochs as ignored,
   instead of deleting them.
   - **What it touches:** every dataset, evidence packet and model trained
     since.
   - **Why it's your decision:** the corrected-labels file avoids reprocessing,
     because it relabels the signal that already exists.
3. **Decide about the test labels.** Re-scoring the *frozen* N4kd once against
   corrected test labels would give its true test accuracy. It doesn't select
   or tune anything; it only corrects the answer key. It does change a reported
   number, so it should be an explicit, documented decision. Say the word and I
   can produce the corrected test labels and that single re-score.
4. **Downstream:** the evidence packets are built from model predictions, not
   ground truth, so the report model's 29/29 result is unaffected. The
   reliability tables fitted on validation labels (per-stage reliability, the N1
   flag threshold) were fitted against partly wrong labels and could be refitted
   later.
