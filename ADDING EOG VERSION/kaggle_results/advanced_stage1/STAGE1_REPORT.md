# Advanced student, Stage 1 — results (seed 0, 5 folds × {base, full_eeg, full})

**Analysed 12 October 2026.** Trained and validated on **corrected labels**;
the test split was never loaded.

## 1. Formal verdict (`summarise_cv.py`, five fold means)

| Arm | Params | κ | Δκ vs base | 95% CI | N1 | REM | Verdict |
|---|---:|---:|---:|---|---:|---:|---|
| base | 139,606 | 0.7584 | — | — | — | — | — |
| full_eeg | 160,511 | 0.7653 | +0.0069 | [+0.0030, +0.0108] | +2.9 | +1.4 | not shown |
| **full** | **175,599** | **0.7763** | **+0.0179** | **[+0.0095, +0.0263]** | **+4.2** | **+3.4** | **HELPS** |

`full` beats `base` in all five folds (Δκ +0.022, +0.024, +0.022, +0.014, +0.008).

## 2. Person by person (`analyse_stage1.py`, 85 subjects)

All 15 checkpoints were re-run on CPU. They reproduce their own result files to
within κ 0.0038. Never-scored epochs are excluded here; the trainer's `accuracy`
counts them as errors (0.46% of epochs), so its accuracies are about 0.4 pts lower.

| Pooled, corrected labels | Acc | κ | Macro F1 | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 81.97% | 0.7578 | 77.89 | 91.5 | 50.3 | 84.1 | 80.7 | 82.9 |
| full_eeg | 82.64% | 0.7666 | 78.66 | 91.8 | 53.7 | 84.7 | 78.8 | 84.3 |
| **full** | **83.47%** | **0.7775** | **79.64** | 92.4 | **55.0** | 85.0 | 79.4 | **86.4** |

| Paired comparison | Δκ (pooled) | 95% CI (subject bootstrap) | Subjects better | Wilcoxon p (per-subject κ) |
|---|---:|---|---:|---:|
| full vs base | **+0.0197** | [+0.0142, +0.0262] | 65 / 85 | 1.8e-08 |
| full_eeg vs base | +0.0088 | [+0.0034, +0.0147] | 56 / 85 | 0.011 |
| full vs full_eeg (EOG's share) | +0.0109 | [+0.0061, +0.0158] | 59 / 85 | 5.6e-05 |

By cohort, `full` vs `base`:
- **SC:** κ 0.756 → 0.776, N1 49.6 → 54.1, REM 81.7 → 85.8.
- **ST:** κ 0.742 → 0.763, N1 54.0 → 59.9, REM 86.7 → 88.3, Wake 79.1 → 81.4.

## 3. Reading it honestly

- **The gain is real and consistent:** Δκ ≈ +0.02, with the interval well above
  zero on both the formal and the person-level test. Both cohorts improve.
- **N1 +4.7 and REM +3.5 points** (pooled), the two stages that were targeted.
- **N3 is the one cost:** 80.7 → 79.4 (SC 81.0 → 79.0). The frequency/local part
  (`full_eeg`) is where it appears (78.8). Watch it in Stage 3.
- **EOG is the larger share of the gain** (+0.011 κ on top of `full_eeg`); the
  frequency and local parts add +0.009 on their own.
- **The person-level test was chosen after the formal one** (as for Stage A). The formal
  verdict stands as the pre-registered answer; this is strong supporting evidence.
- **These are cross-validation numbers on corrected labels** over 85 subjects, not
  test numbers, and the original validation split is a known pessimistic draw.
- **One seed.** Stage 3 runs three.
