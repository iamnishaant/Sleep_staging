# Stage A — does EOG help? (seed 0, 5 folds × {eog, eegonly})

**Analysed 11 October 2026.** These were the Kaggle runs of
`kaggle_train_student_kd_eog.py`:
- hard labels, no teacher, no warm start;
- **trained and validated on the stored labels**, which carry the
  preprocessing alignment bug;
- 10 runs completed, with no duplicated epochs across the resumed sessions;
- the test split was never loaded.

## 1. The pre-registered verdict (`summarise_eog_cv.py`, five fold means)

| | EEG-only | EEG + EOG | Δ | 95% CI | EOG better |
|---|---:|---:|---:|---|---:|
| κ | 0.7133 | 0.7241 | **+0.0109** | [−0.0041, +0.0258] | 4 / 5 |
| Macro F1 | 0.7408 | 0.7513 | +0.0105 | [+0.0003, +0.0208] | 4 / 5 |
| Accuracy | 78.68% | 79.48% | +0.80 pts | [−0.42, +2.03] | 4 / 5 |
| N1 F1 | 47.2 | 49.2 | **+2.0** | [+0.4, +3.6] | **5 / 5** |
| REM F1 | 78.2 | 80.1 | +1.9 | [−0.3, +4.2] | **5 / 5** |
| N2 / N3 F1 | 80.8 / 74.7 | 81.1 / 74.9 | +0.3 / +0.1 | include 0 | — |

**Bar** (fixed before any run): REM ≥ +2.0, κ ≥ +0.01, κ CI excluding 0.

**Verdict: "benefit not shown."** κ's CI includes zero, driven by fold 4, and
REM is +1.94, just under the bar.

## 2. Supplementary: person by person (`analyse_stageA.py`)

The five validation folds partition the 85 non-test subjects. Each subject is
scored once by the EOG model and once by the EEG-only model of its own fold.
All 10 checkpoints were re-run on CPU; they reproduce their own result files to
within κ 0.0043.

| Pooled over 85 subjects | Accuracy | κ | Macro F1 | N1 | REM | W | N2 | N3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stored labels:** EEG-only | 78.75% | 0.7146 | 74.14 | 47.4 | 78.5 | 89.7 | 80.8 | 74.4 |
| **Stored labels:** EEG + EOG | 79.46% | 0.7243 | 75.21 | 49.4 | 80.4 | 90.4 | 81.0 | 74.8 |
| **Corrected labels:** EEG-only | 81.08% | 0.7459 | 76.78 | 49.5 | 81.8 | 91.0 | 83.2 | 78.3 |
| **Corrected labels:** EEG + EOG | **81.95%** | **0.7577** | **78.02** | **51.6** | **84.2** | 91.8 | 83.6 | 78.9 |

| Paired test | Stored labels | Corrected labels |
|---|---|---|
| Pooled Δκ, 95% CI (subject bootstrap) | +0.0097 [**+0.0028, +0.0168**] | +0.0117 [**+0.0038, +0.0199**] |
| Subjects where EOG is better | 52 / 85 | 56 / 85 |
| Per-subject Δκ, Wilcoxon | +0.0089, p = 0.034 | +0.0112, **p = 0.004** |
| Per-subject Δ accuracy, Wilcoxon | +0.61 pts, p = 0.027 | +0.76 pts, **p = 0.005** |

**By cohort** (corrected labels):
- **SC:** κ 0.752 → 0.763, REM 81.3 → **84.3**, N1 49.5 → 51.5.
- **ST:** κ 0.691 → 0.709, REM 83.5 → 83.6, N1 49.7 → **52.3**.

## 3. Reading it honestly

- **Formally,** the pre-registered test did **not** pass. Five fold means can't
  resolve a ~0.01 κ effect. That limitation was known in advance, and the plan
  called for more seeds in a borderline case.
- **Person by person,** the same predictions show a **small, consistent, real
  benefit:**
  - κ +0.010 to +0.012, with the confidence interval above zero on both label
    sets;
  - REM about +2, N1 about +2;
  - N2 and N3 no longer damaged.

  This is supplementary evidence, chosen after the formal test. Treat it as
  strong support, not as a replacement verdict.
- **Where EOG helps:** REM (on SC nights) and N1 (on both cohorts), plus Wake —
  the stages eye movements define. Removing EOG from a trained EOG model costs
  κ 0.016–0.021 on every fold.
- **The label fix is worth more than EOG.** The same EEG-only predictions go
  from 78.75% to 81.08% when scored against corrected labels (+2.3 pts), against
  +0.7–0.9 for EOG. Both together: 81.95%.
- **These are cross-validation numbers over 85 subjects, not test numbers.**
  The original validation split is a known pessimistic draw, so these aren't
  comparable to the 79.0% corrected-label test figure.

## 4. Decision

- **Keep EOG.** It goes forward as the EOG branch inside the advanced trainer's
  `full` arm.
- **Next runs use `--corrected-labels`**, since the label fix is the largest
  measured effect.
- **The advanced trainer's Stage 1** (`base`, `full_eeg`, `full`, 5 folds,
  corrected labels) re-tests EOG on correct labels. It also measures the
  frequency branch, local context and training recipe.
