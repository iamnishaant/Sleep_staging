# Stage 3 — production candidate (full, 175,599 parameters) and its one test evaluation

**13 October 2026.** `full` trained on the original split with the ENSmc teacher (α 0.5),
a warm start from N4kd and corrected labels; 3 seeds, 75 epochs each (none early-stopped;
best epochs 68–72, so training was still slowly improving at the end).

## Validation (original 16 subjects, corrected labels) — chosen before any test read

| Seed | κ | Macro F1 | N1 | REM | N3 | EOG-off κ | TF-off κ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7425 | 0.7732 | 0.560 | 0.836 | 0.756 | 0.7269 | 0.7401 |
| **1 (chosen: highest macro-F1)** | 0.7427 | **0.7747** | 0.566 | 0.840 | 0.752 | 0.7292 | 0.7403 |
| 2 | 0.7416 | 0.7734 | 0.563 | 0.834 | 0.757 | 0.7277 | 0.7396 |

Pre-set bar (N4kd re-scored on corrected labels: κ 0.7218, N1 0.538, REM 0.802): κ ≥ 0.732,
N1 ≥ 0.538, REM ≥ 0.802. **All three seeds pass.** The seeds agree within 0.001 κ, so the choice
among them barely matters. (The logs' "vs N4kd 0.681" line uses stored-label figures; ignore it.)

## Test (one evaluation, corrected test labels, never-scored epochs excluded)

Checkpoint sha256 `61e94bfc8aaf547bb0ca97e7ec8e48edefda1a594f6555912314a87961cd2cb0`, recorded
before test data was read. N4kd's official test figures were reproduced first.

| | Acc | κ | Macro F1 | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N4kd, decoded as shipped | 79.00% | 0.7169 | 75.23 | 90.7 | 45.5 | 81.9 | 79.9 | 78.2 |
| N4kd, argmax | 79.06% | 0.7180 | 75.25 | 90.7 | 45.4 | 81.9 | 79.9 | 78.4 |
| **full (seed 1), argmax** | **81.96%** | **0.7557** | **78.20** | 91.4 | **50.7** | 84.3 | 80.9 | **83.7** |
| full + N4kd's decoder (not refit) | 81.96% | 0.7556 | 78.25 | 91.4 | 50.8 | 84.4 | 80.9 | 83.8 |
| full, EOG off | 81.10% | 0.7442 | 77.24 | 90.7 | 48.7 | 84.0 | 81.4 | 81.4 |
| full, frequency branch off | 80.85% | 0.7416 | 77.21 | 90.9 | 48.9 | 83.2 | 80.5 | 82.5 |

- **vs N4kd (argmax):** Δκ **+0.038**, 95% CI [+0.019, +0.057] (subject bootstrap),
  better for **13 of 15** test subjects, Wilcoxon p = 0.0009.
- **Gains:** N1 +5.3, REM +5.3, N2 +2.4, W +0.7, N3 +1.0 points. No stage got worse.
- **By cohort:** SC κ 0.754 (82.1%), ST κ 0.746 (81.4%).
- **Without EOG** the model still beats N4kd (κ 0.744 vs 0.718), so a missing EOG channel degrades
  it gracefully; REM drops 2.3 and N1 2.0 points.
- **N4kd's decoder adds nothing** here (κ 0.7556 vs 0.7557); no decoder is needed.

## Caveats
- One test evaluation, one seed chosen on validation; the other seeds were not scored on test.
- Test labels are the corrected ones (alignment fix); the figures are not comparable with the
  earlier stored-label 77.7%. The like-for-like comparison is the N4kd row above (79.0%).
- Gain sources are not separated on test (only EOG-off and TF-off of the same model).
- Not yet done: teacher/student report-pipeline integration, reliability tables refit on the new
  model's outputs, and re-validation of the SLM claims, all of which depend on the staging model.
