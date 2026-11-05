> **Banner added 2 Nov 2026.** The current release is **v2** (`CURRENT_RELEASE.md`). Where this file calls `student_N4kd` the delivered model, or says the shipped model takes one electrode, it describes release v1. Corrected figures and what changed: `output_v2/README.md`. **F19's four TinySleepNet per-class values other than N1 had no source and were removed.** Nothing below this banner was changed.

# FIGURES.md — Visualization Spec for the Report / Journal

One spec per figure: what it shows, the chart type, the exact data to plot, where the data lives, and
the caption to use. **Priority** column: ★ = must-have (carries the argument), ◦ = supporting.

Numbering matches `S7_Report_Draft.md` where possible. Values marked "in the report" are quoted from
`PROJECT_REPORT.md` / `README.md` / `b1_summary.json` and can be hardcoded if regenerating is heavy.

---

## Quick index

| ID | Title | Type | Priority | Source |
|---|---|---|---|---|
| F1 | Subject-level split composition | stacked bar | ★ | `make_splits` output |
| F2 | Recording-level vs subject-level split (leak schematic) | illustration | ★ | none (diagram) |
| F3 | Exposure gradient (leakage proof) | grouped bar | ★ | `exposure_analysis.json` |
| F4 | Model architecture | illustration | ★ | none (diagram) |
| F5 | Receptive field vs. stage-defining events | timeline/bar | ★ | `student_encoder.py` + §6b |
| F6 | Teacher ladder (E0/E1a/E1b) | grouped bar | ★ | §4 / §7 tables |
| F7 | Encoder ablation (before/after) | paired bar | ★ | `multiscale_encoder_ablation.json` |
| F8 | Channel ablation (EOG+Pz-Oz) | two diverging bars | ★ | §6c |
| F9 | Cohort transfer, per-stage F1 | grouped bar | ◦ | `cohort_transfer.json` |
| F10 | Prior & amplitude negatives | two mini-panels | ◦ | `prior_adaptation.json`, `cohort_transfer_pernorm.json` |
| F11 | Distillation mechanism (KL vs performance) | two-panel line | ★ | §6 RQ2 table |
| F12 | Teacher quality vs. KD penalty | line/scatter | ★ | distillation results |
| F13 | Calibration + REM latency, before/after | paired bar | ★ | `kd_result.json` + §6d |
| F14 | Parameters vs. κ | scatter | ★ | §7 table |
| F15 | Confusion matrix (delivered model) | heatmap | ★ | model predictions |
| F16 | Per-class F1 across models | radar / grouped bar | ★ | §7 + README |
| F17 | B1 per-fold κ | line/bar | ★ | `b1_summary.json` |
| F18 | B1 vs published (acc vs params) | scatter | ★ | `b1_summary.json` + §2.1 |
| F19 | B1 per-class F1 vs published | grouped bar | ◦ | `b1_summary.json` + §2.1 |
| F20 | Verifier ablation (A/B/C) | grouped bar | ★ | `ablation.json` |
| F21 | Evidence-packet → verifier schematic | illustration | ◦ | none (diagram) |
| F22 | N1 confidence vs accuracy | line | ★ | `n1_flag.json` + §6k |
| F23 | N1 threshold sweep (headroom) | line | ◦ | §8.4 |
| F24 | N1-vs-N2 pairwise AUC | bar | ◦ | `diagnose_separability.py` |

---

## Data & evaluation

### F1 — Subject-level split composition ★
- **Type:** 100% stacked horizontal bar, three bars (train / validation / test).
- **What it shows:** each bar is split by cohort (SC vs ST), with subject counts labelled.
- **Data:** train 69 (54 SC / 15 ST), val 16 (12 SC / 4 ST), test 15 (12 SC / 3 ST). Recording counts: 137 / 31 / 29. Source: the split built by `distillation/make_splits.py` (subject-level, cohort-stratified, test touched once).
- **Axes:** x = count (subjects or recordings), y = split; segments coloured SC/ST.
- **Caption:** *Subject-level, cohort-stratified split of Sleep-EDFx. No subject appears in more than one split; the 15 test subjects are absent from all training and model-selection folds.*

### F2 — Recording-level vs subject-level split ★
- **Type:** schematic / diagram (no data).
- **What it shows:** two panels. Left: a recording-level split puts Subject A's night 1 in train and night 2 in validation. Right: a subject-level split keeps both of Subject A's nights on the same side. Shade the "leaked" night on the left.
- **Caption:** *Why a recording-level split leaks. Sleep-EDFx records two nights per subject; splitting by recording lets the model see night 1 and be scored on night 2 of the same person, measuring recognition rather than generalisation.*

### F3 — Exposure gradient (the leakage proof) ★
- **Type:** grouped bar (or dot plot with CI).
- **What it shows:** κ (and optionally N3/REM F1) at three exposure levels, with the published 0.6663 as a horizontal dashed line.
- **Data:** same recording 0.8058, same subject/different night 0.6797, genuinely unseen subject 0.5066. N3 F1: 0.889 / 0.837 / **0.000**. REM F1: 0.876 / 0.718 / **0.184**. Source: `distillation/results/exposure_analysis.json` (§3).
- **Axes:** x = exposure level; y = κ. A second panel for N3 and REM F1.
- **Caption:** *The inherited model measured at three exposure levels. On a genuinely unseen subject, κ collapses to 0.5066 and N3 F1 to 0.000 — the published κ 0.6663 (dashed) was memorisation, not generalisation.*

---

## Model & architecture

### F4 — Model architecture ★
- **Type:** block diagram.
- **What it shows:** raw EEG → multi-scale temporal encoder (0.5 s + 2 s kernels) and 34 spectral features → spectral encoder → fusion → cross-epoch transformer → per-epoch softmax. Annotate parameter counts on the student vs teacher.
- **Caption:** *The staging model: a multi-scale waveform branch (short kernel for spindles/K-complexes, long kernel for slow waves) fused with a spectral branch over 34 band-power features, then a cross-epoch transformer.*

### F5 — Receptive field vs. stage-defining events ★
- **Type:** horizontal timeline / bar.
- **What it shows:** the old encoder's receptive field (0.25 s), the teacher's (0.49 s), the replacement's (8.75 s), aligned against the durations of sleep spindles (0.5–2 s), K-complexes (0.5–2 s), slow waves (0.5–2 s) and sawtooth waves (1–3 s).
- **Data:** §6b; field lengths computed by backpropagation in `distillation/student_encoder.py`.
- **Axes:** x = time (log or linear scale); one row per event/encoder.
- **Caption:** *Measured receptive fields versus the durations of the events that define each stage. The original encoder saw only 0.25 s, shorter than any stage-defining event, which is why the raw EEG contributed nothing until the encoder was rebuilt.*

---

## Ablation studies

### F6 — Teacher ladder (E0 / E1a / E1b) ★
- **Type:** grouped bar — one group per teacher, bars for κ and per-class F1.
- **What it shows:** the loss-function fix moved N2 (+0.196) and κ (+0.093) while N1 stayed flat; E1a inflated N1 predictions 2.41×, E1b at 1.06×.
- **Data:** §4: E0 κ 0.5122, E1a 0.5592, E1b 0.6055; per-class F1 table in §4 ("Where the gain came from").
- **Caption:** *The teacher ladder under identical architecture, changing only the objective. Fixing the focal-loss defect recovered N2 (+0.196 F1) and κ (+0.093) with no N1 gain — evidence the earlier N1 improvement was inflated by over-prediction.*

### F7 — Encoder ablation (before/after) ★
- **Type:** paired before/after bar (two grouped bars per metric).
- **What it shows:** κ, macro-F1, N1 F1, REM F1, and a "raw EEG drives X% of predictions" callout.
- **Data:** §6b: val macro-F1 0.6821 → 0.7236, N1 0.431 → 0.523, REM 0.699 → 0.763, raw-EEG drive 0% → 71%, params 121,099 → 139,606. Source: `distillation/results/multiscale_encoder_ablation.json`.
- **Caption:** *Replacing the atrous encoder with a multi-scale encoder lifted every class and made the raw waveform drive 71% of predictions (previously 0%).*

### F8 — Channel ablation (EOG + Pz-Oz) ★
- **Type:** two small diverging bars (validation vs test Δκ), or a dot with CI spanning zero.
- **What it shows:** the change is +0.0028 κ on test but −0.0122 on validation; paired p = 0.865.
- **Data:** §6c table.
- **Caption:** *Adding EOG and a second EEG derivation (Pz-Oz) gained on test but lost on validation; the paired test is p = 0.865 — a mechanism that reverses across splits is noise, not signal.*

### F9 — Cohort transfer, per-stage F1 ◦
- **Type:** grouped bar — SC (CV) vs ST (held-out) per stage.
- **What it shows:** the −0.187 κ gap is almost entirely Wake; W F1 0.898 → 0.622 with precision 54.6%.
- **Data:** `distillation/results/cohort_transfer.json` (§6h / README finding 6).
- **Caption:** *Crossing from the Sleep Cassette to the Sleep Telemetry cohort. Sleep-stage discrimination largely survives; the collapse is a Wake precision failure, not a general detection failure.*

### F10 — Prior & amplitude negatives ◦
- **Type:** two mini-panels side by side.
- **What it shows:** (a) prior correction — uncorrected 0.5516 vs oracle 0.5539 (recovers 1.2% of the gap) vs EM 0.4757; (b) per-recording normalisation — 0.5516 → 0.5341 (paired CI excludes zero, negative).
- **Data:** `distillation/results/prior_adaptation.json`, `distillation/results/cohort_transfer_pernorm.json` (§6h/§6i).
- **Caption:** *Two cheap explanations for the cohort gap are measured and excluded: perfect prior knowledge recovers 1.2% of the gap, and amplitude normalisation makes transfer slightly worse.*

---

## Distillation mechanism

### F11 — Distillation mechanism: KL vs. performance ★
- **Type:** two-panel line (shared x = epoch).
- **What it shows:** top panel: the un-distilled baseline's KL to the teacher rises 0.196 → 5.802; bottom panel: its validation macro-F1 rises 0.3312 → 0.6825. The divergence is the point.
- **Data:** §6 RQ2 table (epochs 0/20/40/74).
- **Caption:** *The student that ignores the teacher moves steadily away from it while getting steadily better (30× KL divergence). Distilling from a weaker teacher therefore pulls the student away from its own optimum.*

### F12 — Teacher quality vs. KD penalty ★
- **Type:** line or scatter with three points, y=0 reference line.
- **What it shows:** the penalty shrinks with teacher quality: E0 −0.0523, E1b −0.0314, ensemble +0.0008.
- **Data:** §6 RQ3 + §6d.
- **Caption:** *The distillation penalty is a function of the teacher–student gap: it shrinks as the teacher improves and vanishes once the teacher (an ensemble) exceeds the student.*

### F13 — Calibration + REM latency, before/after ★
- **Type:** paired bar (two metrics).
- **What it shows:** ECE test 0.0359 → 0.0241; REM-latency error 33.7 → 18.3 min. Add "accuracy ties (p = 0.24)" as a note.
- **Data:** `distillation/results/kd_result.json` + §6d.
- **Caption:** *Ensemble distillation does not move accuracy; it transfers calibration — ECE falls by a third and REM-latency error nearly halves on both splits.*

---

## Main results

### F14 — Parameters vs. κ ★
- **Type:** scatter (log x-axis for parameters).
- **What it shows:** every model from §7 as a point; the delivered 139,606-parameter student highlighted above the 649K teachers.
- **Data:** §7 complete-results table.
- **Caption:** *Test κ versus parameter count. The 139,606-parameter delivered student beats every 649,229-parameter teacher — a 4.65× compression with no accuracy loss.*

### F15 — Confusion matrix (delivered model) ★
- **Type:** row-normalised heatmap, 5×5.
- **What it shows:** W/N2/REM are diagonally strong; N1 leaks into N2 and REM.
- **Data:** regenerate from the delivered model's test predictions (per-class F1 in README: W 0.897 · N1 0.429 · N2 0.805 · N3 0.779 · REM 0.778).
- **Caption:** *Row-normalised confusion matrix of the delivered student on held-out test. N1 is the weak point, split between N2 and REM — the standard, human-scorer-level ambiguity.*

### F16 — Per-class F1 across models ★
- **Type:** radar (5 axes) or grouped bar.
- **What it shows:** the delivered model's per-class profile vs the best teacher, with N1 clearly the floor.
- **Data:** §7 per-class F1 table + README headline per-class F1.
- **Caption:** *Per-class F1 on held-out test. Wake, N2, N3 and REM are strong; N1 is the consistent weak point across every model.*

---

## Benchmark (B1)

### F17 — B1 per-fold κ ★
- **Type:** bar or dot plot, 10 points, with the mean and the 0.746–0.798 range.
- **What it shows:** tight across-fold spread — stable, not a lucky split.
- **Data:** `kaggle/working/b1/b1_summary.json` (`per_fold[*].val.kappa`).
- **Caption:** *Sleep-EDF-78 subject-wise cross-validation, per-fold κ. The tight 0.746–0.798 spread shows the headline is stable across subject splits.*

### F18 — B1 vs published: accuracy vs. parameters ★
- **Type:** scatter (log x = params), your point highlighted.
- **What it shows:** your 0.176M / 83.2% point sits in the same accuracy band as TinySleepNet/SeqSleepNet with far fewer parameters than most.
- **Data:** `kaggle/working/b1/b1_summary.json` + the published table in `GPU_PLAN_AND_BENCHMARK.md` §2.1.
- **Caption:** *Sleep-EDF-78: accuracy versus parameter count. This work reaches 83.2% accuracy / κ 0.761 at 0.176M parameters, competitive with the published single-channel baselines at a fraction of their size.*

### F19 — B1 per-class F1 vs published ◦
- **Type:** grouped bar, one group per model, bars = N1 and macro-F1 (or all five stages).
- **What it shows:** N1 F1 52.5 vs the published range; macro-F1 (epoch-pooled ≈ 0.79) level with the table.
- **Data:** `b1_summary.json` per-class + §2.1 table.
- **Caption:** *Per-class F1 on Sleep-EDF-78 versus published baselines. N1 is at the top of the published group (except LightSleepX's outlier), and the epoch-pooled macro-F1 is level with the table.*

---

## Trustworthiness / verifier

### F20 — Verifier ablation (arm A/B/C) ★
- **Type:** grouped bar, one group per arm, bars = "reports released with a flaw".
- **What it shows:** trained model: A 0, B 1, C 2 flawed; untrained base model: A 0, B 31, C 31. Add the "0 wrong numbers in every arm" note.
- **Data:** `ROADMAP R2 - VERIFIER ABLATION/results/ablation.json`.
- **Caption:** *The deterministic verifier releases nothing incorrect. For the trained model the gate blocks 1–2 hedging/completeness errors; for the untrained base model it blocks all 31 — correctness is independent of model quality.*

### F21 — Evidence-packet → verifier schematic ◦
- **Type:** flow diagram.
- **What it shows:** staging probabilities → evidence packet (per-value error bounds, tiers) → report generator (structured claims) → deterministic verifier → VERIFIED / fallback.
- **Caption:** *The report layer: derived values travel with measured error bounds, and a 30-rule verifier gates every claim so unsupported statements become fallbacks rather than errors.*

---

## N1 and confidence

### F22 — N1 confidence vs accuracy ★
- **Type:** line or bar.
- **What it shows:** monotone accuracy rise with confidence: 0.0–0.4 → 12%, 0.4–0.6 → 39%, 0.6–0.8 → 59%, 0.8–1.0 → 80%.
- **Data:** `distillation/results/n1_flag.json` + §6k.
- **Caption:** *The model's own confidence tracks its N1 accuracy monotonically, so the packet flags low-confidence N1 epochs for review rather than asserting them as fact.*

### F23 — N1 threshold sweep ◦
- **Type:** line (threshold vs F1), with current operating point marked.
- **What it shows:** headroom of only +0.0086 over any threshold — N1 is representation-bound, not threshold-bound.
- **Data:** §8.4 threshold-sweep result.
- **Caption:** *Sweeping the N1 decision threshold yields only +0.0086 κ of headroom, so rebalancing cannot fix N1; the correct response is abstention, not tuning.*

### F24 — N1-vs-N2 pairwise AUC ◦
- **Type:** bar (N1-vs-N2 ≈ 0.81 vs N1-vs-N3 ≈ 0.97).
- **What it shows:** the one-vs-rest AUC hides which specific pair is broken.
- **Data:** `distillation/diagnose_separability.py` (§8.4).
- **Caption:** *N1 separates well from N3 (AUC 0.97) but poorly from N2 (≈ 0.81); the one-vs-rest AUC conceals this, which is why N1 error concentrates at the N1/N2 boundary.*

---

## Notes for generating

- **Consistent stage colours** (W, N1, N2, N3, REM) across every per-class figure.
- **Subject-clustered CIs** on any κ point estimate; the delivered model's is [0.6432, 0.7514].
- **Tables vs figures:** the complete-results table, distillation table and B1 comparison table stay as
  tables — don't duplicate them as plots.
- **Regenerate, don't hardcode**, where the JSON is cheap to read (`b1_summary.json`, `ablation.json`,
  the `distillation/results/*.json` files). For the historical §3/§4/§6 numbers, the report text is the
  source of truth.
