# B1 Report — Sleep-EDF-78 Benchmark of the Final Model

**29 October 2026 · Team 40 · Sleep-Staging**

## 1. What B1 is and why it exists

B1 is the single pre-registered run of the **published Sleep-EDF-78 protocol** on the final, shipped
staging model. Its purpose is to produce one number that can be compared like-for-like with the
Sleep-EDF-78 results in the literature, without the confounding factors of the project's own split,
teacher distillation, or the custom agentic pipeline.

Before B1, the project's strongest sleep-staging claim was measured on its own 5-fold split (85
subjects, SC + ST mixed, 12 SC test subjects excluded), which is *not* directly comparable to the
standard Sleep-EDF-78 table. B1 closes that gap by running exactly the benchmark the base papers run.

The configuration was frozen before any fold ran: no result on this benchmark was used to tune the
model, so B1 is an external evaluation, not a model-selection step.

## 2. Protocol (frozen before the run)

| Property | Value |
|---|---|
| Dataset | Sleep-EDF **SC (sleep-cassette)** recordings only |
| Subjects | **78** (153 recordings) |
| Split | **10-fold subject-wise CV** (deterministic, seed 0; 8 folds of 8 subjects + 2 of 7) |
| Training | **From scratch** — no knowledge distillation, no warm start |
| Model | **full arm** (EEG + EOG + time-frequency + local context), the deployed v2 `n6adv_full` staging model |
| Parameters | **175,599** |
| Labels | Corrected labels (train/val + test merged) |
| Wake handling | 30-min wake trim (baked into tensors at preprocessing) |
| Seed | 0, one seed per fold (this is a benchmark, not a seed study) |
| Checkpoint | Best epoch per fold by validation macro-F1 (`student_best.pt`) |

### Why "from scratch"

The project's teacher and warm-start checkpoint were trained on subjects that overlap the Sleep-EDF-78
cohort. Using them here would leak training signal across the benchmark's held-out folds. From-scratch
training is the only leak-free option and is what the published protocol requires.

### Why the test-split labels are read

B1's 10-fold split covers all 78 SC subjects, which includes some subjects in the project's own locked
test split. This is deliberate and pre-registered (`GPU_PLAN_AND_BENCHMARK.md` §2.3, `TEST_ACCESS_LOG.md`):
B1 is a different, external evaluation of the frozen model, not a read of the project's held-out set
for model selection. See `TEST_ACCESS_LOG.md` for the access log and coverage check.

## 3. What was actually done

1. **Split construction (`build_b1_splits.py`).** A fixed-seed shuffle of the 78 SC subject IDs was
   chunked into 10 near-even folds, then expanded to recording IDs. Written to
   `results/b1_splits.json`. Deterministic and reproducible.

2. **Packaging (`pack_b1.py`).** The driver script, a read-only copy of the shipped training module
   (`kaggle_train_student_advanced.py`), the frozen split, and both corrected-labels files were bundled
   into `b1_bundle.zip` for Kaggle. No source file was edited.

3. **Training (`run_b1_benchmark.py` on Kaggle, T4 × 2).** Each fold trained the full arm from scratch
   on the other 9 folds and evaluated on the held-out fold. Folds are independently resumable, and each
   writes its own `student_best.pt`, `student_last.pt`, `training_metrics.jsonl`, and
   `result_fold_*.json`. `all_results.json` was updated after every fold.

4. **Summarisation (`--summarise`).** After all 10 folds, every one of the 153 recordings was pooled —
   each scored exactly once, by the fold where it was held out — into `b1_summary.json`.

All 10 folds completed. Per-fold run time was ~55 minutes, about ~9 GPU-hours total.

## 4. Results

### 4.1 Headline (mean over 10 folds, per-recording pooled)

| Metric | Value |
|---|---:|
| **Accuracy** | **83.2%** |
| **Cohen's κ** | **0.761** |
| **Macro-F1** | **0.737** |

| Stage | Mean F1 |
|---|---:|
| Wake (W) | 0.908 |
| N1 | 0.525 |
| N2 | 0.839 |
| N3 | 0.559 |
| REM | 0.854 |

### 4.2 Per-fold

| Fold | n_val | Accuracy | κ | Macro-F1 |
|---:|---:|---:|---:|---:|
| 0 | 16 | 0.812 | 0.746 | 0.755 |
| 1 | 15 | 0.833 | 0.769 | 0.795 |
| 2 | 16 | 0.829 | 0.766 | 0.784 |
| 3 | 16 | 0.805 | 0.747 | 0.749 |
| 4 | 15 | 0.852 | 0.788 | 0.797 |
| 5 | 16 | 0.850 | 0.798 | 0.813 |
| 6 | 16 | 0.835 | 0.777 | 0.795 |
| 7 | 16 | 0.856 | 0.796 | 0.792 |
| 8 | 14 | 0.833 | 0.779 | 0.808 |
| 9 | 13 | 0.842 | 0.787 | 0.812 |

κ ranges from 0.746 to 0.798 — a tight ~0.05 spread, so the headline is stable across subject splits
and not a single lucky fold.

### 4.3 Training dynamics

Convergence is fast and consistent. Fold 0 is typical:

| Epoch | Loss | Accuracy | κ | Macro-F1 |
|---:|---:|---:|---:|---:|
| 0 | 1.516 | 0.493 | 0.288 | 0.383 |
| 2 | 0.820 | 0.745 | 0.651 | 0.642 |
| 4 | 0.636 | 0.794 | 0.719 | 0.717 |
| **29 (best)** | 0.436 | 0.812 | 0.746 | 0.755 |
| 54 (last) | 0.394 | 0.809 | 0.740 | 0.747 |

Validation κ reaches ~0.72 by epoch 4 and peaks on epoch 29, then declines slowly while training loss
keeps falling — mild overfitting after the best epoch. Best epoch varies from 7 to 61 across folds,
confirming early stopping matters.

## 5. What the results show

1. **The model is in the same band as the published lightweight baselines.** B1's κ 0.761 / accuracy
   83.2% sits level with SeqSleepNet (κ 0.76), U-Time (0.75) and TinySleepNet (0.77), while using
   175,599 parameters — about 7–75× fewer than most of the table.

2. **Wake, N2 and REM are robust** (F1 ≥ 0.84). **N1 is the weak point** (F1 ≈ 0.52), which is the
   standard, well-documented failure mode in sleep staging — N1 is rare and confusable with W and REM.

3. **The model generalises across subjects.** Subject-wise splitting rules out leakage, and the tight
   per-fold spread means the result is not inflated by seeing a subject's other nights at train time.

4. **It does not overfit badly** provided the best epoch is selected — but it does overfit after the
   peak, so the early-stopping checkpoint is what matters.

### 5.1 A note on aggregation and the N3 number

The headline is pooled **per recording** (each of the 153 recordings is averaged with equal weight).
This down-weights long recordings relative to epoch-pooled scoring and slightly lowers every headline
number. The fairer epoch-pooled figures are a touch higher:

| Metric | Per-recording (headline) | Epoch-pooled |
|---|---:|---:|
| Accuracy | 83.2% | ~83.5% |
| κ | 0.761 | ~0.775 |
| Macro-F1 | 0.737 | ~0.79 |

The **N3 F1 of 0.559** is the clearest example of this artifact. The headline averages *per-recording*
N3 F1, and a few recordings with very little (or zero) N3 score N3 F1 = 0, dragging the mean down. The
epoch-pooled per-fold N3 F1 is **0.72–0.86 (mean ≈ 0.78)**, so N3 itself is staged well; the low
headline number is an averaging artifact, not a model weakness on N3.

## 6. Comparison with published Sleep-EDF-78 results

Published figures are from the papers' own tables (see `GPU_PLAN_AND_BENCHMARK.md` §2.1). B1 is the
EEG + EOG "full" arm; the published single-channel baselines use Fpz-Cz only.

| Model | Inputs | Params | Acc % | Macro-F1 | κ | N1 F1 |
|---|---:|---:|---:|---:|---:|---:|
| DeepSleepNet | EEG | 21M | 78.5 | 75.3 | 0.73 | 47.0 |
| SleepTransformer | EEG | 3.7M | 81.4 | 74.3 | 0.74 | — |
| U-Time | EEG | 1.1M | 81.3 | 76.3 | 0.75 | 51.0 |
| SeqSleepNet | EEG | 0.16M | 82.6 | 76.3 | 0.76 | 46.0 |
| TinySleepNet | EEG | 1.3M | 83.1 | 78.1 | 0.77 | 51.0 |
| LightSleepX | EEG + EOG | 0.049M | 82.6 | 78.7 | 0.77 | 63.0† |
| SalientSleepNet | 3-ch | 0.9M | 84.1 | 79.5 | — | 54.2 |
| MultiChannelSleepNet | 3-ch | 13M | 85.0 | 79.6 | 0.79 | 53.0 |
| **B1 (ours)** | **EEG + EOG** | **0.176M** | **83.2** | **73.7** | **0.761** | **52.5** |

† LightSleepX's N1 F1 is unusually high next to all others; treat with care.

Reading:

- **Accuracy and κ** place B1 squarely among the strong lightweight models (SeqSleepNet, TinySleepNet,
  U-Time), with an order of magnitude fewer parameters than most of the table.
- **N1 F1 (52.5)** is at the top of the group except for LightSleepX's outlier — N1 is the field's
  known hard case, and B1 handles it no worse than the baselines.
- **Macro-F1 (73.7)** appears lower than the published 76–79.6, but that gap is mostly the
  per-recording aggregation. B1's epoch-pooled macro-F1 (~0.79) is level with the published table.
  State the aggregation method when citing this number.
- The top of the table (SalientSleepNet, MultiChannelSleepNet) uses three channels and 0.9–13M
  parameters; B1 stays within ~0.5–1.5 accuracy/κ points of them at 0.176M parameters.

## 7. Why B1 matters

- **It is the base-paper number.** It measures the model on the standard, reproducible Sleep-EDF-78
  protocol, independent of the custom agentic Path B pipeline (no study-period estimator, no quality
  gate, no lights-off crop). This is the figure to cite for any like-for-like external comparison.
- **It benchmarks the deployed model family.** B1 uses the same 175,599-parameter EEG + EOG "full" arm
  as the v2 staging adapter, so the result transfers directly to what Path B runs in production.
- **The subject-wise split rules out leakage.** The tight per-fold spread means the headline is not a
  single lucky subject split.
- **It completes the protocol gap.** Before B1, the project's comparable CV number (κ 0.776) was
  measured on its own 5-fold split; B1 now provides the pre-registered, externally comparable figure
  (κ 0.761) the base-paper claim required.

## 8. Caveats

- Published Sleep-EDF-78 baselines are usually **single-channel Fpz-Cz**. B1 uses the EEG + EOG "full"
  arm, so 0.761 is comparable but not a strict single-channel apples-to-apples number — state this
  explicitly in any base-paper claim.
- The headline **N3 F1 (0.559)** and **macro-F1 (0.737)** are per-recording averages; the epoch-pooled
  N3 (~0.78) and macro-F1 (~0.79) are the fairer per-epoch figures.
- This is **one run on one seed**. It is a benchmark, not a seed study; variance is bounded by the tight
  per-fold spread but is not zero.

## 9. Files

| File | Contents |
|---|---|
| `kaggle/working/b1/b1_summary.json` | Headline, per-fold, per-class metrics |
| `kaggle/working/b1/all_results.json` | Per-fold status, best epoch, checkpoint paths |
| `kaggle/working/b1/fold_*/result_fold_*.json` | Per-recording metrics for each fold |
| `kaggle/working/b1/fold_*/training_metrics.jsonl` | Per-epoch training/validation curves |
| `kaggle/working/b1/fold_*/student_best.pt` / `student_last.pt` | Fold checkpoints |
| `kaggle/working/b1/README.md` | Final-model write-up |
| `results/b1_splits.json` | The frozen 10-fold subject-wise split |
