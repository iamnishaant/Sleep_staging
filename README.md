# Compact Sleep Staging by Knowledge Distillation

A leakage-corrected re-evaluation of a 5-class sleep-staging model, and a compact
replacement **5.36× smaller** with no measurable loss of accuracy.

**Team 40** · Project 48 — Explainable Deep Learning for Sleep Disorder Detection
Dataset: Sleep-EDFx (PhysioNet) — 197 recordings, 100 subjects, 237,950 epochs

---

## Result

| Model | Parameters | Held-out test κ |
|---|---:|---:|
| Inherited model, *as published* | 649,229 | 0.6663 ← **not a generalisation figure** |
| Inherited model, on a subject it never saw | 649,229 | 0.5066 |
| Best rebuilt teacher (E1b) | 649,229 | 0.6055 |
| Student distilled from E1b | 121,099 | 0.6134 |
| **Delivered model — student, no distillation** | **121,099** | **0.6449** |

Measured on 15 subjects appearing in no training or model-selection set. The test
split was loaded exactly once per model.

### Two findings

**1. The published score was measuring memorisation.** Sleep-EDFx records two nights
per subject. The inherited model split the data by *recording*, so 86% of its
validation subjects also appeared in training. Re-measured on a genuinely unseen
subject, κ fell from 0.6663 to 0.5066 — with F1 on deep sleep (N3) at exactly
**0.000**. Confirmed by reconstructing the published figure as a leakage-weighted
blend, agreeing to 0.0008.

**2. Knowledge distillation made the model worse.** Distillation was implemented and
run from two teachers of different quality. Both distilled students were beaten by
the identical architecture trained on hard labels alone (−0.0523 and −0.0314 κ,
paired Wilcoxon p = 0.0026). The penalty shrinks as the teacher improves; the cause
is that the student outgrows its teacher during training, so pulling it back toward
that teacher degrades it. **The delivered model is therefore un-distilled.**

---

## Branches

| Branch | What it is |
|---|---|
| **`team-40`** | This work. Everything below. |
| **`past`** | The inherited Group 47/48 codebase, as received — the comparison baseline. |

```bash
git diff past team-40        # exactly what Team 40 contributed
```

---

## Documentation

| File | Read it for |
|---|---|
| [`ABSTRACT.md`](ABSTRACT.md) | One page — start here |
| [`PROJECT_REPORT.md`](PROJECT_REPORT.md) | Full walkthrough: what was done, why, and every result |
| [`README_V2.md`](README_V2.md) | Technical reference — architectures, file inventory, per-phase results |
| [`CODEBASE_AUDIT.md`](CODEBASE_AUDIT.md) | Audit of the inherited repository |
| [`update.md`](update.md) | Session-by-session engineering log |
| [`distillation/PREREGISTRATION_gate3a.md`](distillation/PREREGISTRATION_gate3a.md) | Pre-registered predictions for the attribution gate |

---

## Repository layout

```
distillation/
├── results/                    evidence artefacts (JSON) + trained checkpoints
│   ├── students/               the 3 student models, incl. the delivered one
│   ├── retrained/ E1a/ E1b/    the 3 teacher variants
│   └── *.json                  every number in the report traces to these
├── figures/                    8 report figures (+ figures_dark/)
├── make_splits.py              subject-level splits, with assertions
├── kd_loss.py                  distillation loss (T² correction)
├── test_kd_loss.py             13 tests — the T² term fails silently without them
├── night_confidence.py         per-recording confidence, no ground truth needed
└── kaggle_train_*.py           training scripts (paste-whole, edit-nothing)
```

## Reproducing

```bash
python distillation/verify_data.py       # data integrity + label vocabulary
python distillation/make_splits.py       # subject-level splits
python distillation/test_kd_loss.py      # 13/13 must pass
python distillation/evaluate_student.py \
    --students student_distilled_E0 student_distilled_E1b student_baseline_E0
python distillation/make_figures.py      # regenerates all 8 figures
```

Every figure and table is computed from `distillation/results/*.json`. Nothing is
transcribed by hand, so re-running after any new experiment keeps the documents
consistent.

## Data

Raw recordings and preprocessed tensors are **not** in this repository — they are
~2 GB and freely available from [PhysioNet](https://physionet.org/content/sleep-edfx/).
`processed_sleepedf/index.csv` (included) describes the expected layout.

## Limitations

- N1 is representation-bound, not tuning-limited: no decision threshold improves it
  by more than +0.0086, and its separability from N2 did not move across any teacher
  variant. Single-channel EEG without EOG cannot resolve it.
- The best teacher overfits (train κ 0.8541 vs test 0.6055); further teacher work
  should target regularisation.
- The Sleep Telemetry cohort generalises worse than Sleep Cassette (κ 0.4755 vs
  0.6182) and is under-measured at 3 test subjects.

---

MIT licensed. See [`LICENSE`](LICENSE).
