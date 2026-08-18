# Project Update & Audit Report

**Generated:** 2026-08-10 07:15  (revision 2 — adds the E1b teacher rung)
**Repository:** `f:\Sleep_project\Sleep-Staging` · branch `main` · base commit `b00a428`
**Scope:** everything done in this working session — Group 48 lineage only (sleep staging → distillation). Group 47 (NAS) untouched throughout.

---

## 0. M0 and the scope boundary

> **M0 = `distillation/results/students/student_baseline_E0/student_best.pt`** — 121,099 params, **un-distilled**, held-out test κ **0.6449**.

The deployed model was **not** distilled. Distillation was attempted, measured as a significant negative (p = 0.00262) with an identified mechanism, and the un-distilled compact student was deployed. Everything downstream builds on M0.

**Two distillations exist in this project and must never be conflated:**

| | Status |
|---|---|
| **Staging-model** distillation (649K teacher → 121K student) | **Complete** — significant negative, mechanism identified (§6.3) |
| **Language-model** distillation (O5 / C4 / RQ4) | **Not started** — separate question, separate data |

The result on the first says nothing about the second.

---

## 0b. Late finding (2026-08-17): the models never used the raw EEG

Found while smoke-testing the attribution pipeline. Zeroing the **entire** raw EEG input changes **zero predictions** in every trained model — agreement exactly 1.00000 for the 121K student and both 649K teachers. Cause: the preprocessed tensors are in volts (~2e-5) against spectral features at ~7, a ~450,000× scale mismatch that leaves the temporal branch numerically inert.

Normalising it (scale 15849.46, fitted on the training split only) **did** revive the branch — deleting the EEG then costs 0.2366 κ in the student and 0.4306 κ in the teacher — but **accuracy did not improve**: −0.0060 and −0.0069 val macro-F1 respectively.

**Why:** the 34 spectral features are computed *from* the same EEG. They are a lossy summary of identical data, not a second modality, so the raw trace adds no independent information.

**M0 unchanged.** `student_baseline_E0` remains the deliverable. Full figures: `distillation/results/eeg_normalization_ablation.json`; narrative in `PROJECT_REPORT.md` §8.4b.

*Correction:* an earlier note in commit `4b8b82f` said the inert branch held "most of the parameters". It does not — 2.5% of the teacher, 8.4% of the student. The transformer is 82–92% of both.

---

## 1. Executive Summary

We set out to build a compact distilled student from Group 48's inherited sleep-staging teacher. Before distilling we audited the inherited evidence, and that audit changed the project.

**The single most important finding:** the inherited teacher's published κ = 0.6663 was not a generalization number. It was measured with a recording-level split that leaked 86% of its validation subjects into training, and the committed confusion matrices were computed on the training set. Under honest, subject-level evaluation the same design scores **κ ≈ 0.51**.

From that corrected baseline we improved the model and produced a working compact student.

| Model | Params | Held-out test κ |
|---|---|---|
| Inherited teacher, on the 1 subject it never saw | 649,229 | 0.5066 |
| E0 — inherited design, honest protocol | 649,229 | 0.5122 |
| E1a — focal loss, α power 1.0 | 649,229 | 0.5592 |
| E1a + prior correction | 649,229 | 0.6318 |
| **E1b — focal `pt` fixed, α power 0.5** | 649,229 | **0.6055** |
| Student, distilled from E0 | 121,099 | 0.5925 |
| **Student, baseline (best model)** | **121,099** | **0.6449** |

**Net: +0.14 κ over the inherited model, at 5.36× fewer parameters**, measured on 15 subjects that appear in no training or selection set.

**Teacher improvement E0 → E1b:** accuracy **+0.0926**, κ **+0.0933**, macro-F1 **+0.0485** on held-out test — the largest single improvement in the project. Full breakdown in §5.3.

**The headline negative result:** knowledge distillation *hurt*, at both teacher-quality points tested, and the harm is **significant** (p = 0.00262, worse on 13 of 15 held-out subjects). But it is not arbitrary — it scales with teacher quality:

| Teacher | Teacher test κ | Distilled student test κ | Gain vs baseline |
|---|---:|---:|---:|
| E0 | 0.5122 | 0.5925 | **−0.0523** |
| **E1b** | **0.6055** | **0.6134** | **−0.0314** |

A **+0.0933 κ** better teacher shrank the penalty by **+0.0209 κ — a 40% reduction**. Naive linear break-even sits near teacher κ ≈ 0.75. The mechanism is identified and evidenced (§6.3): the baseline's KL-to-teacher *rises* 0.196 → 5.802 as it improves, so the KD term pulls the student away from its own optimum. RQ3 is answered.

**Statistical status of the compression claim.** Paired per-subject Wilcoxon signed-rank over the 15 test subjects:

| Comparison | Student better on | Mean Δκ | p |
|---|---|---:|---:|
| Student (121K) vs **E0** (649K) | 14 / 15 | +0.1337 | **0.00043** |
| Student (121K) vs **E1b** (649K) | 10 / 15 | +0.0309 | 0.135 (n.s.) |

So the defensible claim is: **the 121K student matches the best 649K teacher at 5.36× compression, and significantly beats the faithful replication of the original design.** It should not be written as "outperforms E1b".

---

## 2. What We Started With

The repository is the shared working tree of two final-year groups. Group 48's staging work had:

- A model class that existed **only inside a Kaggle notebook** (`code/Phase2_48/temporal-spectral-fusion.ipynb`, cells 5–6)
- `code/Phase1_48/Phase1reworked/model.py` — a `.py` file that matched **no committed checkpoint** (stale)
- `processed_sleepedf/index.csv` with absolute paths from two other machines
- **No test split had ever existed** — training used an 80/20 train/val split only
- Six checkpoints named `best_model*.pt` with no metadata, several under a `module.` DataParallel prefix

---

## 3. Files Created

### 3.1 Documentation (repo root)

| File | Size | Contents |
|---|---|---|
| `CODEBASE_AUDIT.md` | 45 KB | Full audit of all 142 Python files: what each contributes, the Group 47/48 split, 9 broken imports in `code/`, four-way file duplication, and 6 code-level defects |
| `README_V2.md` | 28 KB | Group 48 orientation: teacher architectures layer-by-layer, data state, the cell-14 evaluation finding written up for your supervisor, Phase 1 and Phase 2 results |
| `update.md` | this file | Session audit |

### 3.2 Pipeline code (`distillation/`)

| File | Purpose | Why it exists |
|---|---|---|
| `fix_index_paths.py` | Rewrites `index.csv` to repo-relative paths | Both path columns pointed at other machines; data was unusable locally |
| `verify_data.py` | Loads all 197 recordings, checks shapes/dtypes/alignment/label vocabulary | Needed to know whether `N4` (R&K scoring) was present before training |
| `make_splits.py` | Subject-level, cohort-stratified 70/15/15 → `splits.json` | 97 of 100 subjects have 2 nights; a recording-level split leaks |
| `teacher_model.py` | The real teacher class, extracted from the notebook | The class lived only in a notebook; `model.py` was stale |
| `eval_teacher.py` | Phase 2 gate: teacher by exposure level | To measure how much of κ 0.6663 was leakage |
| `exposure_analysis.py` | Permanent artefact generator for the exposure gradient | The gradient is a reportable figure |
| `eval_final.py` | Held-out test evaluation, per-recording κ + entropy | The honest numbers |
| `diagnose_separability.py` | One-vs-rest **and pairwise** ROC-AUC, probability mass | To decide whether N1 is fixable by rebalancing |
| `prior_correction.py` | Validation-fitted class-prior correction | Free κ gain; fixes the N1 over-prediction |
| `kd_loss.py` | Sequence-aware Hinton KD loss | Core of the distillation objective |
| `test_kd_loss.py` | 13 unit tests | The `T²` term is a silent-failure risk |
| `student_model.py` | Student definition + compression accounting | Phase 4 |
| `cache_teacher_logits.py` | fp16 soft-target cache, resumable, verified | Avoids re-running the teacher every epoch |
| `train_student.py` | Local student trainer | Phase 5 |
| `evaluate_student.py` | Student val-reproduction + test evaluation | Correctness check plus honest numbers |
| `calibrate.py` | Temperature scaling, ECE, reliability table | Phase 7 deliverable |
| `make_figures.py` | All 8 report figures, light + `--dark` | Reads the JSON artefacts only — no hardcoded numbers, so it cannot drift from the results |

### 3.3 Kaggle-ready scripts

| File | Used for |
|---|---|
| `kaggle_train_teacher.py` | **E0** — frozen baseline, do not modify |
| `kaggle_train_improved.py` | **E1a / E1b / E4** ladder. `EXPERIMENT` constant at the top — currently `"E1b"` (complete). Paste whole, edit nothing. |
| `kaggle_train_student.py` | Student distilled + baseline. `TEACHER_TAG` currently `"E1b"`, `RUN_BASELINE = False`. Hard-fails if the attached checkpoint's own metadata disagrees with `TEACHER_TAG`. |

### 3.4 Result artefacts (`distillation/results/`)

`data_verification.json`, `splits.json`, `eval_teacher.json`, `exposure_gradient.json`, `exposure_gradient_comparison.json`, `separability_diagnostic.json`, `separability_diagnostic_E1a.json`, `separability_diagnostic_E1b_E1a.json`, `separability_verdict.json`, `prior_correction.json`, `prior_correction_E1a.json`, `prior_correction_E1b.json`, `eval_final_inherited.json`, `eval_final_retrained.json`, `eval_final_E1a.json`, `eval_final_E1b.json`, `eval_students.json`, `retrain_summary.json`, `e1a_summary.json`, `reliability_table.json`

Model checkpoints: `retrained/` (E0), `E1a/`, `E1b/` — each `teacher_best.pt` + `teacher_last.pt` + `training_metrics.jsonl`; `students/student_distilled_E0/`, `students/student_distilled_E1b/` and `students/student_baseline_E0/`. Also `teacher_logits_E0/` (fp16 soft-target cache, 137 recordings). The equivalent E1b cache was generated on Kaggle during the E1b distillation run and not downloaded — it is an intermediate artefact and regenerates in ~0.6 min on GPU via `cache_teacher_logits.py`.

Plus cached predictions/probabilities per model, and `teacher_logits_E0/` (137 recordings, 169,909 epochs, 1.6 MB fp16).

### 3.5 Figures (`distillation/figures/`, `figures_dark/`)

Eight figures, regenerated by `python distillation/make_figures.py` from the JSON artefacts. **Re-run this after any new evaluation** — the script hardcodes no numbers, so it stays correct automatically, but stale PNGs on disk will not.

`fig1_exposure_gradient` (the leakage proof) · `fig2_compression_vs_kappa` · `fig3_per_class_f1` · `fig4_pairwise_auc` · `fig5_kl_vs_performance` (**the RQ2 mechanism**) · `fig6_confusion_matrix` · `fig7_reliability_diagram` · `fig8_rq3_teacher_quality` (**the RQ3 finding**)

**E1b checkpoint verification** (run before evaluating, to catch the stale-best resume bug that bit the E0 run): `teacher_best.pt` reports epoch 51 with `best_macro_f1` 0.6976, matching the training log exactly; `experiment` field reads `E1b`; and `best`/`last` have different weight hashes, confirming `best` was not overwritten by a later worse epoch.

---

## 4. Data — Verified State

| Check | Result |
|---|---|
| Recordings | 197 (153 SC + 44 ST), 100 subjects |
| Total epochs | 237,950 |
| Load failures | 0 |
| Shape / dtype | `[T,3000]` + `[T,34]`, float32, all |
| Epoch alignment | 197/197, zero mismatches |
| Label vocabulary | exactly `{W, N1, N2, N3, REM}` — **no `N4`**, no remapping needed |

**Class distribution:** N2 37.40%, W 29.48%, REM 14.37%, N1 10.58%, N3 8.18% — 4.6:1 imbalance.

**Splits** (subject-level, cohort-stratified, seed 42; assertions on subject IDs):

| Split | Subjects | Recordings | Epochs |
|---|---|---|---|
| train | 69 | 137 | 169,909 |
| val | 16 | 31 | 34,610 |
| test | 15 | 29 | 33,431 |

Subject key is `recording_id[:5]`, validated for **both** cohorts — SC is `SC4<ss><night>E0`, ST is `ST7<ss><night>J0`.

---

## 5. What Succeeded

### 5.1 The leakage forensics

Reconstructed the inherited teacher's split exactly: `train_test_split(range(197), test_size=0.2, random_state=42)`.

**Exposure gradient** — monotonic decline with distance from training:

| Exposure | κ |
|---|---|
| same recording (trained on) | 0.7412 |
| same subject, different night | 0.6855 |
| unseen subject | 0.6151 |

**Reconstruction check:** epoch-weighting the two measured validation subgroups gives **0.6655** against the originally reported **0.6663** — agreement to **0.0008**. This proves the split reconstruction is exact and that the published figure is a blend of mostly-leaked measurement.

Only **5 subjects / 10 recordings** were ever unseen by the inherited teacher, all SC. **No ST subject was ever held out.**

### 5.2 The separability diagnostic

Pairwise ROC-AUC (your refinement — one-vs-rest alone would have hidden this):

| | E0 | E1a |
|---|---|---|
| N1 vs N3 | 0.9720 | 0.9700 |
| **N1 vs N2** (binding) | **0.7716** | **0.8142** |
| N1 average precision | 0.2699 | 0.3202 |

**Threshold sweep:** current N1 F1 0.3390; best achievable over *any* threshold 0.3476. **Headroom +0.0086.** This killed the planned E2 (class-balanced sampling) experiment with evidence rather than opinion — no decision rule can fix N1.

### 5.3 Teacher improvement — the E1 ladder, E0 → E1a → E1b

**All three exposure levels, all three teachers.** Best-macro-F1 checkpoint, same evaluation script, test split loaded exactly once per model.

| Model | Group | n epochs | Accuracy | κ | macro-F1 | entropy |
|---|---|---:|---:|---:|---:|---:|
| **E0** | train | 33,963 | 0.8065 | 0.7527 | 0.7911 | 0.540 |
| | val | 34,610 | 0.5910 | 0.4854 | 0.5862 | 0.652 |
| | **test** | 33,431 | **0.6160** | **0.5122** | **0.6063** | 0.655 |
| **E1a** | train | 33,963 | 0.7787 | 0.7182 | 0.7618 | 0.622 |
| | val | 34,610 | 0.6205 | 0.5290 | 0.6260 | 0.681 |
| | **test** | 33,431 | **0.6506** | **0.5592** | **0.6458** | 0.710 |
| **E1b** | train | 33,963 | 0.8924 | 0.8541 | 0.8538 | 0.573 |
| | val | 34,610 | 0.7347 | 0.6469 | 0.6977 | 0.651 |
| | **test** | 33,431 | **0.7086** | **0.6055** | **0.6548** | 0.650 |

**Improvement E0 → E1b on held-out test: accuracy +0.0926, κ +0.0933, macro-F1 +0.0485.** Exposure gradient monotone (train > val > test) for every model — no leakage.

**Per-class F1 by exposure level:**

| Model | Group | W | N1 | N2 | N3 | REM |
|---|---|---:|---:|---:|---:|---:|
| E0 | train | 0.924 | 0.554 | 0.763 | 0.777 | 0.938 |
| | val | 0.806 | 0.382 | 0.575 | 0.571 | 0.598 |
| | test | 0.807 | 0.339 | 0.622 | 0.651 | 0.612 |
| E1a | train | 0.892 | 0.495 | 0.746 | 0.782 | 0.894 |
| | val | 0.819 | 0.453 | 0.555 | 0.583 | 0.721 |
| | test | 0.818 | 0.386 | 0.640 | 0.674 | 0.712 |
| **E1b** | train | 0.945 | 0.615 | 0.905 | 0.882 | 0.922 |
| | val | 0.877 | 0.449 | 0.750 | 0.667 | 0.745 |
| | **test** | **0.821** | **0.353** | **0.756** | **0.686** | **0.658** |

**What E1b actually fixed.** E1a's focal loss computed `pt = exp(-ce)` on the *α-weighted* CE, making `pt = p^α` and the modulator `(1-p^α)²`. For α > 1 (N1, N3) that inflates the weight; for α < 1 (W, N2) it deflates it — so the bug **compounded** the class rebalancing. N2 is ~38% of all epochs, and suppressing it costs more than the N1 gain repays. E1b removes the compounding and softens α to sqrt-inverse-frequency. **N2 validation F1 moves 0.554 → 0.750 (+0.196)**, which is where the +0.114 validation accuracy comes from.

**The N1 result — equal detection, honestly earned:**

| | N1 F1 (val) | N1 prediction ratio (val) |
|---|---:|---:|
| E1a | 0.453 | **2.41×** |
| E1b | 0.449 | **1.06×** |

E1a was not detecting N1 better; it was guessing N1 2.4× more often than N1 occurs. It never dropped below 2.3× across all 75 epochs. E1b gets the same detection with **2.3× fewer false N1 calls**.

**E1b confusion matrix, held-out test** (rows = true, columns = predicted):

| | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| **W** | **7,438** | 891 | 223 | 50 | 295 |
| **N1** | 492 | **1,280** | 1,217 | 91 | 423 |
| **N2** | 594 | 1,116 | **9,610** | 660 | 798 |
| **N3** | 219 | 10 | 573 | **1,895** | 54 |
| **REM** | 475 | 457 | 1,028 | 75 | **3,467** |

| Stage | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| W | 0.8069 | 0.8360 | 0.8212 | 8,897 |
| N1 | 0.3410 | 0.3654 | 0.3528 | 3,503 |
| N2 | 0.7596 | 0.7521 | 0.7558 | 12,778 |
| N3 | 0.6839 | 0.6888 | 0.6863 | 2,751 |
| REM | 0.6883 | 0.6301 | 0.6579 | 5,502 |

**Generalisation gap — the remaining bottleneck:**

| | train κ | test κ | gap |
|---|---:|---:|---:|
| E0 | 0.7527 | 0.5122 | +0.2405 |
| E1a | 0.7182 | 0.5592 | **+0.1590** |
| E1b | 0.8541 | 0.6055 | +0.2486 |

E1a genuinely overfits least; its **ST cohort gap collapsed from 0.067 to 0.012**. E1b wins on absolute test performance but gets there by fitting training subjects much harder, and only part transfers. The next lever is **regularisation, not loss shaping**.

**Per-subject E1b test κ** (mean 0.5897, sd 0.1538): SC mean 0.6182 (n=12) vs **ST mean 0.4755 (n=3)** — the ST cohort remains substantially harder. SC461 at 0.2823 is an outlier across every model tested.

### 5.4 Prior correction — free κ, and its own diagnostic value

Validation-fitted `g`, applied unchanged to test:

| | fitted g | κ before | κ after | Δ |
|---|---:|---|---|---|
| E0 | — | 0.5122 | 0.5715 | +0.059 |
| E1a | 0.775 | 0.5592 | **0.6318** | **+0.073** |
| **E1b** | **0.025** | 0.6055 | 0.6055 | **+0.0000** |

Zero GPU cost. For E1a it fixed the N1 over-prediction (2.61× → ~1.0×).

**E1b's +0.0000 is a result, not a disappointment.** A fitted `g` of 0.025 is the identity transform: there is nothing left to correct because the calibration is already right in-model. This is the cleanest independent confirmation that the focal fix worked rather than merely relocating numbers.

### 5.4b Separability — N1 is representation-bound, and stayed that way

One-vs-rest ROC-AUC, held-out test, plus the decisive pairwise score:

| Model | W | N1 | N2 | N3 | REM | **N1-vs-N2** |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.9449 | 0.7658 | 0.8797 | 0.9515 | 0.8968 | — |
| E1a | 0.9622 | 0.8026 | 0.9037 | 0.9553 | 0.9233 | 0.8142 |
| E1b | 0.9617 | 0.8103 | 0.8899 | 0.9575 | 0.9020 | 0.8089 |

N1-vs-N2 pairwise AUC is **unchanged** (0.8142 → 0.8089). The loss fix moved the operating point, not the underlying separability — consistent with the E2 cancellation.

### 5.5 The student

**121,099 params, 5.36× compression, test κ 0.6449.** Full figures:

| | Split | Accuracy | κ | macro-F1 | entropy |
|---|---|---:|---:|---:|---:|
| **Distilled (α=0.5, T=3.0)** | val | 0.7022 | 0.6055 | 0.6747 | — |
| | test | 0.6926 | 0.5925 | 0.6611 | 0.628 |
| **Baseline (α=1.0)** | val | 0.7302 | 0.6389 | 0.6881 | — |
| | **test** | **0.7360** | **0.6449** | **0.6915** | 0.555 |

| Model | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| Distilled (test F1) | 0.804 | 0.390 | 0.733 | 0.703 | 0.675 |
| **Baseline (test F1)** | **0.845** | 0.389 | **0.773** | **0.737** | **0.713** |
| Baseline prediction ratio | 0.95× | 1.11× | 0.96× | 0.98× | 1.12× |

Validation reproduced **exactly** (delta 0.0000) for both students when recomputed locally from checkpoints — confirming checkpoints, model class and splits all agree.

Validation reproduced **exactly** (delta 0.0000) for both students when recomputed locally from checkpoints — confirming checkpoints, model class and splits all agree.

### 5.6 Reliability table (Phase 7 deliverable)

Temperature **T = 1.3347** fitted on validation, applied to test. **ECE 0.0454 → 0.0242** (roughly halved).

| Stage | F1 | support | reliability |
|---|---|---|---|
| W | 0.8447 | 8,897 | **high** |
| N2 | 0.7735 | 12,778 | **high** |
| N3 | 0.7370 | 2,751 | medium |
| REM | 0.7135 | 5,502 | medium |
| **N1** | **0.3886** | 3,503 | **low** |

---

## 6. What Failed, and Why

### 6.1 Student attempt 1 — no spectral branch, no class weighting

**Result:** N3 and REM F1 exactly **0.000**. Collapsed to majority classes.
**My diagnosis at the time:** class imbalance.
**Correct?** No — see 6.2.

### 6.2 Student attempt 2 — added sqrt-α class weighting

**Result:** N3 and REM **still exactly 0.000** at epoch 11, hard-CE 1.627 against a chance level of ln(5) = 1.609. Not learning at all.

**Why I was wrong:** I fixed the mechanism that *usually* causes majority collapse without checking the evidence I already had. The teacher reaches **N3 F1 = 0.406 at epoch 0** on identical data. That gap should have pointed at the *input*, not the loss.

**Actual cause:** the student had no spectral branch. **N3 is defined by delta-band power, which is directly one of the 34 precomputed spectral features.** The teacher is handed it; the student had to rediscover it from a raw 3000-sample trace through 2 conv branches and a global average pool — a strictly harder problem than its teacher solves, while being 6× smaller.

**Fix:** restore the spectral branch (`Linear(34→64)` + concat fusion, ~10.6K params). Compression 5.88× → 5.36×. N3 went from 0.000 to 0.795 **at epoch 0** in a smoke test.

**Cost of the error:** roughly 5 wasted GPU-hours and one wasted diagnosis cycle.

### 6.3 Distillation itself — a real negative result, now with a dose–response curve

Both rungs, held-out test, against the same stored baseline (κ 0.6449):

| Held-out test | distilled from E0 | distilled from E1b | baseline |
|---|---:|---:|---:|
| accuracy | 0.6926 | 0.7123 | **0.7360** |
| κ | 0.5925 | 0.6134 | **0.6449** |
| macro-F1 | 0.6611 | 0.6708 | **0.6915** |
| **gain (κ)** | **−0.0523** | **−0.0314** | — |

**The penalty scales with teacher quality.** A teacher +0.0933 κ better shrank the penalty by +0.0209 κ — a **40% reduction**. Slope ≈ 0.224 κ per κ of teacher quality; naive linear break-even at teacher κ ≈ **0.75**. Two points is directional evidence, not a fitted law.

**The penalty is significant, not noise.** Paired per-subject Wilcoxon over the 15 test subjects for the E1b rung: distilled better on **2/15**, mean Δκ **−0.0352**, **p = 0.00262**.

**Report the test figure, not validation.** Validation understates the penalty ~4× in both rungs, because validation is also the selection split:

| Rung | Validation gain | Test gain | Ratio |
|---|---:|---:|---:|
| E0 | −0.0134 | −0.0523 | 3.9× |
| E1b | −0.0072 | −0.0314 | 4.4× |

**This is not a bug.** The mechanism is visible in the baseline's KD loss, which is *computed but not optimised* when α=1.0:

```
ep  0: KL-to-teacher = 0.196
ep 20: KL-to-teacher = 3.394
ep 74: KL-to-teacher = 5.802     ← 30× divergence
```

**The student that ignores the teacher moves steadily away from it while getting steadily better.** The distilled runs drove that same term *down* — successfully matching a teacher worse than themselves. Corroborating this, E1b's soft loss sat below E0's at every epoch (3.782 → 0.428 vs 3.988 → …): the better-calibrated teacher was easier to match, and produced exactly the smaller penalty predicted.

**Why the student beats its teacher:** not distillation magic. In both rungs the 121K distilled student exceeds its own 649K teacher (E0 +0.0803, E1b +0.0079) — but the *un-distilled* baseline beats every teacher and every distilled student. The student had three advantages the teachers never did: overlap 192 (2,721 windows vs 729), sqrt-α weighting instead of full inverse-frequency focal, and lower capacity. The teachers were handicapped by the *inherited* recipe.

**Conclusion for the report:** a teacher must exceed the student's own un-distilled performance before its soft targets carry usable information. Here even the best teacher (κ 0.6055) sits below the baseline student (κ 0.6449), so KD constrains the student toward a weaker function than it finds alone.

**Per-class, held-out test:**

| | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| Distilled from E0 | 0.804 | 0.390 | 0.733 | 0.703 | 0.675 |
| Distilled from E1b | 0.826 | 0.368 | 0.761 | **0.728** | 0.671 |
| **Baseline** | **0.845** | **0.389** | **0.773** | 0.737 | **0.713** |

**Checkpoint integrity:** the E1b student's validation reproduced locally at 0.6806 vs 0.6809 logged (Δ −0.0003) before any test number was read.

### 6.4 Infrastructure failures (resolved)

| Failure | Cause | Fix |
|---|---|---|
| `NCCL Error 1` ×2 | Kaggle 2×T4 peer-to-peer | `MULTI_GPU="off"` + single-GPU chunking, verified mathematically identical (6e-07 relative gradient error) |
| `no kernel image available` | Kaggle allocated a **P100** (sm_60); PyTorch build has no sm_60 kernels | GPU capability guard at startup |
| Student resumed at epoch 41, ran 2.8 min | Stale `/kaggle/working` checkpoint, `since_best` already ≥24 | Resume guard on `(alpha, T, cw_power, use_spectral)` |
| File corruption during edits | Mangled config block, truncated training loop | Rebuilt and re-verified |

---

## 7. Bugs Found in Inherited Code

| # | Bug | Status |
|---|---|---|
| 1 | Cell 14 builds `test_dataset` with **no `file_indices`** → evaluates on all 197 recordings, 157 of which it trained on. Committed confusion matrices and ROC curves are train+val combined | Documented in `README_V2.md` §5A.1 for your supervisor |
| 2 | Recording-level split → 86% subject leakage | Fixed by `make_splits.py` |
| 3 | Focal `pt = exp(-ce)` where `ce` is already α-weighted → `pt = p^α`, not `p`. Suppresses N2 ~6× more than intended | Preserved in E0/E1a for faithful replication; `FIX_FOCAL_PT=True` in E1b |
| 4 | `train_epoch` defined twice in `train_comorbidity_classifier.py` | Documented |
| 5 | `'attention_light'` unreachable in FLOP estimator | Documented |
| 6 | κ = 0.738 in the project proposal appears in **no committed log**; observed ceiling was 0.6663 | Documented — **correct the proposal** |

---

## 8. Where We Stand Against the Plan

| Phase | Planned | Status |
|---|---|---|
| **0** Orient & document | README_V2, pick teacher | ✅ Complete |
| **1** Repair data layer | paths, verify, subject splits | ✅ Complete |
| **2** Teacher gate | evaluate, decide | ✅ Complete — verdict: retrain |
| **Retrain** | honest-protocol teacher | ✅ E0, κ 0.5122 |
| **3** Cache soft targets | fp16, resumable, verified | ✅ 137 recordings, 1.6 MB |
| **4** Define student | compact model | ✅ 121,099 params, 5.36× |
| **5** KD loss + trainer | with `T²` test | ✅ 13/13 tests pass |
| **6** Train & evaluate | Run A + baseline | ✅ Both, on held-out test |
| **7** Calibrate | temperature, ECE, reliability table | ✅ `reliability_table.json` |
| **E1a** Focal, α power 1.0 | ladder rung | ✅ κ 0.5592 / 0.6318 corrected |
| **E1b** Loss geometry (`pt` fix, α power 0.5) | ladder rung | ✅ **Complete — κ 0.6055, best teacher** |
| **E2** Class-balanced sampling | ladder rung | ❌ **Cancelled with evidence** (threshold headroom +0.009) |
| **E3** EOG channel | ladder rung | 📋 Future work, evidence-justified |
| **E4** Multi-token encoding | ladder rung | 📋 Not started |
| **RQ3** Student from better teacher | teacher-quality transfer | ✅ **Complete — E1b rung run, κ 0.6134, penalty −0.0314** |

**Research questions:**

- **RQ1 — can the inherited model be improved?** ✅ **Answered.** κ 0.5066 → 0.6449, +0.14, with mechanism. On the teacher line specifically, E0 → E1b is +0.0933 κ.
- **RQ2 — does distillation transfer?** ✅ **Answered, negatively.** −0.052 κ from a weak teacher, mechanism evidenced.
- **RQ3 — does teacher quality matter?** ✅ **Answered.** Two points: E0 (κ 0.5122) → −0.0523; E1b (κ 0.6055) → −0.0314. Distillation hurts at both, significantly (p = 0.00262), but the penalty **shrinks 40%** as the teacher improves. A teacher must exceed the student's own un-distilled performance before soft targets help; break-even extrapolates to teacher κ ≈ 0.75.

---

## 9. What Is Still Left

**Nothing is required.** Every planned phase is complete, all three research questions are answered, and the primary deliverable — a compact model small enough for Kaggle free tier — is validated on a held-out test split loaded once per model.

**Future work, evidence-justified:**
1. **Regularisation on the teacher.** E1b's remaining bottleneck is overfitting (train κ 0.8541 vs test 0.6055, gap +0.2486), not loss shaping. If a teacher could clear κ ≈ 0.75 the RQ3 trend predicts distillation would stop hurting — that is the one experiment that would extend the result rather than repeat it. New experimental ladder, outside the original scope.
2. **E3 — add EOG.** Strongly justified: N1-vs-N2 pairwise AUC caps around 0.81 on single-channel Fpz-Cz and did not move between E1a (0.8142) and E1b (0.8089); published models reaching N1 F1 0.45–0.52 use EEG+EOG. Requires re-preprocessing.
3. **E4** — multi-token epoch encoding (+129 params).
4. **A third RQ3 point.** Two points establish direction; a third would let the break-even estimate be stated with a confidence interval rather than as an extrapolation.
5. Apply prior correction to the student as well (currently teacher-only). Note this may yield little: the baseline student's prediction ratios are already within 12% of correct frequency.
6. **ST cohort.** Test κ 0.4755 (n=3) vs SC 0.6182 (n=12). Under-measured and substantially harder; worth stating as a limitation rather than fixing.

---

## 10. Key Lessons

1. **Check the evidence you already have before fixing the obvious cause.** The teacher's epoch-0 N3 score of 0.406 was in the logs the whole time and pointed straight at the input asymmetry. I reached for class weighting instead and lost a run.

2. **Pairwise beats one-vs-rest for diagnosing multi-class failure.** N1's one-vs-rest AUC (0.766) averaged an excellent N1-vs-N3 (0.972) with a poor N1-vs-N2 (0.772) and hid which one was broken.

3. **A metric can be measuring the wrong quantity.** The logit-cache verification gated on teacher *argmax* agreement and failed a recording over a single epoch with a 0.00096 logit gap — while the KD soft loss, the thing distillation actually consumes, differed by exactly 0.0.

4. **Fixing a leaky split makes overfitting visible; it doesn't reduce it.** The exposure gap narrowed only when training data increased (E1a).

5. **A negative result with a mechanism is worth more than a small positive one.** "Distillation hurts here, and here is the KL trace showing why" is stronger than a +0.01 gain would have been.

6. **Guard rails pay for themselves.** The resume guard, provenance check, and GPU capability check each caught a real failure that would otherwise have cost hours.

7. **A fixed seed makes a re-run look like a result.** E1a was accidentally re-run because the experiment selector was left as a manual edit; seed 42 reproduced it bit-for-bit, so the log looked entirely correct while answering nothing. ~5.5 GPU-hours. Both Kaggle scripts now carry their selector as a constant in the file with a warning comment, so they are paste-whole / edit-nothing. The same reasoning retired the redundant baseline re-training: with `alpha=1.0` the trainer never reads the teacher, so a second baseline would have been bit-for-bit identical for another ~2.5 GPU-hours.

8. **Verify the flag is live, not just present.** Before running E1b, 12 real training steps were executed under the E1a and E1b configs with identical seed and data. They diverge from step 0 (max |Δloss| 0.765), proving the config reaches the loss. A static check that the constant *says* `"E1b"` would not have caught a dead flag.

9. **A null result can be the confirmation.** E1b's prior correction gained exactly +0.0000 κ with a fitted g of 0.025. That is not a failed correction — it is the cleanest evidence that the calibration is already right in-model, which is precisely what the focal fix was supposed to achieve.

---

## 11. Current Best Model

```
distillation/results/students/student_baseline_E0/student_best.pt
```

- **121,099 parameters** (5.36× smaller than the 649,229 teacher)
- **Held-out test:** accuracy 0.7360 · κ 0.6449 · macro-F1 0.6915
- Evaluated on 15 subjects in no training or selection set
- Calibrated at T = 1.3347, ECE 0.0454 → 0.0242
- Prediction ratios within 12% of correct frequency for every class
- Reliability: W and N2 high, N3 and REM medium, N1 low

**How to state the result.** Paired per-subject Wilcoxon over the 15 test subjects gives p = 0.00043 against E0 (+0.1337 mean Δκ, better on 14/15) but p = 0.135 against E1b (+0.0309, better on 10/15). The supportable claim is therefore:

> A 121,099-parameter student **matches** the best 649,229-parameter teacher (no significant difference) at **5.36× compression**, and **significantly outperforms** a faithful replication of the original design under identical honest evaluation.

Do **not** write "outperforms E1b" — the subject-level test does not support it.

**Runner-up / best teacher:** `distillation/results/E1b/teacher_best.pt` — 649,229 params, test accuracy 0.7086 · κ 0.6055 · macro-F1 0.6548, epoch 51 of 75, 333.8 min on a single T4.

For reference, Sleep-EDF SOTA is roughly κ 0.79–0.81. The remaining gap is diagnosed rather than mysterious: single-channel Fpz-Cz without EOG (N1-vs-N2 pairwise AUC caps ~0.81 and did not move across the whole E1 ladder), the global-average-pool bottleneck, and an ST cohort that generalises substantially worse than SC (0.4755 vs 0.6182).
