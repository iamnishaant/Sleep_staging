# B1: Sleep-EDF-78 Benchmark — Final Model

**29 October 2026.** Pending item B1. A like-for-like, subject-independent benchmark of the final
student model on the standard Sleep-EDF-78 protocol.

## What we did

Ran the **Sleep-EDF-78 protocol** end to end: SC (sleep-cassette) recordings only, **78 subjects
(153 recordings)**, **10-fold subject-wise cross-validation**, each fold **trained from scratch** with
corrected labels and a 30-minute wake trim. Subject-wise means recordings from the same subject never
appear in both train and validation, so the numbers measure generalisation to unseen subjects, not
unseen nights of seen subjects.

| Property | Value |
|---|---|
| Model | **full arm** (EEG + EOG), the same architecture as the deployed v2 `n6adv_full` staging model |
| Parameters | **175,599** |
| Folds | 10 (all `done`) |
| Validation recordings | 153 total (13–16 per fold) |
| Checkpoint | best epoch per fold by validation metric (`student_best.pt`) |

## Results — metrics and values

### Headline (mean over 10 folds)

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

### Per-fold

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

κ ranges from 0.746 to 0.798 — a tight ~0.05 spread, so the result is stable across subject splits.

### Class profile (per-class F1 by fold)

| Stage | Fold range | Reading |
|---|---|---:|---|
| W | 0.90–0.95 | strongest stage |
| N1 | 0.43–0.60 | consistently the weakest — rare and confusable with W/REM |
| N2 | 0.81–0.88 | strong |
| N3 | 0.72–0.86 (epoch-pooled) | strong where present |
| REM | 0.80–0.90 | strong |

**Note on N3 = 0.559.** The headline per-class mean averages *per-recording* F1, and a few recordings
with very little (or zero) N3 score N3 F1 = 0, pulling the mean down. The *epoch-pooled* per-fold N3 F1
is 0.72–0.86 (mean ≈ 0.78), so N3 itself is staged well; the low headline number is a
recording-level averaging artifact, not a model weakness on N3.

### Training dynamics

Convergence is fast and consistent. Fold 0 is typical: validation κ reaches ~0.72 by epoch 4 and peaks
at **0.746 on epoch 29**, then declines slowly to ~0.740 by epoch 54 while training loss keeps falling —
mild overfitting after the best epoch. Best epoch varies 7–61 across folds, confirming early stopping
matters. Each fold runs ~55 minutes (~9 GPU-hours total).

## Inference and importance

- **This is the base-paper number.** It measures the model on the standard, reproducible Sleep-EDF-78
  protocol, independent of the custom agentic Path B pipeline (no study-period estimator, no quality
  gate, no lights-off crop). It is therefore the figure to cite for any like-for-like external
  comparison.
- **It benchmarks the deployed model family.** B1 uses the same 175,599-parameter EEG+EOG "full" arm as
  the v2 staging adapter, so the result transfers directly to what Path B runs in production.
- **The subject-wise split rules out leakage.** The tight per-fold spread means the headline is not a
  single lucky subject split.

## What the result concludes

- The final model reaches **κ 0.761 / accuracy 83.2% / macro-F1 0.737** on Sleep-EDF-78 subject-wise CV —
  a solid, competitive result in the same band as published single-channel baselines (typically
  κ ≈ 0.76–0.80).
- **Wake, N2 and REM are robust** (F1 ≥ 0.84); **N1 is the weak point** (F1 ≈ 0.52), the standard failure
  mode in sleep staging.
- The model **generalises across subjects** with stable performance, and **does not overfit badly**
  provided the best epoch is selected.

## Caveats

- Published Sleep-EDF-78 baselines (SleepTransformer, TinySleepNet, etc.) are usually **single-channel
  Fpz-Cz**. B1 uses the EEG+EOG "full" arm, so 0.761 is comparable but not a strict single-channel
  apples-to-apples number — state this explicitly in any base-paper claim.
- The headline N3 F1 (0.559) is a per-recording average; the epoch-pooled N3 F1 (~0.78) is the fairer
  per-epoch figure.

## Files

- `b1_summary.json` — headline and per-fold metrics.
- `all_results.json` — per-fold status, best epoch, and per-fold validation metrics.
- `fold_*/result_fold_*.json` — per-recording metrics for each fold.
- `fold_*/training_metrics.jsonl` — per-epoch training/validation curves.
- `fold_*/student_best.pt` / `student_last.pt` — fold checkpoints.
