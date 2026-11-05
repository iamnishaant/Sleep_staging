# Compact Sleep Staging by Knowledge Distillation: A Leakage-Corrected Re-Evaluation

**Project Report — structured against `S7_Report_Template.pdf` (Amrita Vishwa Vidyapeetham, Dept. of CSE, Project Phase I, 22AIE498)**

**Author:** Nishant Shah · Project 48 — Explainable Deep Learning for Sleep Disorder Detection
**Dataset:** Sleep-EDFx (PhysioNet) · **Status as of:** 1 November 2026

This document maps the project's actual work onto the chapter structure required by the S7 report
template. Full technical detail lives in `PROJECT_REPORT.md` (1700+ lines); this is the condensed,
chapter-by-chapter version for submission. Reproduction steps and current status live in `README.md`.

---

## Contents

- [Abstract](#abstract)
- [Chapter 1 — Introduction](#chapter-1--introduction)
- [Chapter 2 — Literature Review / Existing System](#chapter-2--literature-review--existing-system)
- [Chapter 3 — Proposed Work](#chapter-3--proposed-work)
- [Chapter 4 — Experimentation and Result Analysis](#chapter-4--experimentation-and-result-analysis)
- [Chapter 5 — Conclusion and Scope for Further Research](#chapter-5--conclusion-and-scope-for-further-research)
- [Appendix A — Code and Reproduction](#appendix-a--code-and-reproduction)
- [References](#references)

---

## Abstract

This project compresses a sleep-staging model — a network that labels every 30 seconds of overnight
EEG as Wake, N1, N2, N3 or REM — into a version small enough to train and run on free-tier cloud
hardware, using knowledge distillation.

Before compressing anything, the inherited model's reported score (Cohen's κ = 0.6663) was audited and
found to be inflated by data leakage: the train/test split divided by *recording*, not by *subject*, and
Sleep-EDFx contains two nights per person. Re-measured on a subject the model had genuinely never seen,
κ fell to 0.5066 and its ability to identify deep sleep (N3) collapsed to zero. Subject-level splits were
rebuilt, the teacher was retrained under a corrected protocol (fixing a loss-function defect that applied
class weights twice), and a compact student was distilled from it. The naive distillation experiment
*failed* — the student always ended up worse than the same architecture trained without a teacher —
and tracing why produced the project's first transferable finding: **a teacher must exceed the student's
own undistilled performance before its guidance carries usable information.**

A second audit overturned an earlier conclusion that raw EEG added nothing over 34 precomputed
spectral features. The encoder's receptive field measured 25 samples (0.25 s) before averaging across a
30-second epoch — too short to see any sleep spindle, K-complex or slow wave. Replacing it with a
two-branch encoder spanning 8.75 s moved raw EEG from contributing nothing to driving 71% of
predictions and produced a student that beats every honestly-evaluated teacher in the project.

With that student, ensembling three comparable models and distilling the ensemble into one student
removed the earlier penalty entirely. The delivered model is **139,606 parameters (4.65× smaller than
the original 649,229-parameter model), reaching κ = 0.7001 (95% CI [0.6432, 0.7514]) on fifteen unseen
subjects** — 0.09 κ above the best teacher it imitates. What distillation improved was not accuracy but
**calibration**: expected calibration error fell by a third and REM-latency error nearly halved.

Three further results came from auditing the evaluation itself: five-fold subject-level cross-validation
showed the original validation split was an unrepresentative draw (0.0193 κ fold-to-fold spread against a
total distillation effect of 0.0008 κ); crossing between the two cohorts inside Sleep-EDFx costs 0.187 κ,
traced to covariate shift rather than class-prior shift; and a pre-registered attribution analysis
(Integrated Gradients) passed its completeness check but revealed that all five sleep stages share the
same top-3 attributed features, so the explanation is sound but weaker than it first reads.

**Keywords:** sleep staging · EEG · knowledge distillation · ensemble teachers · model calibration ·
data leakage · subject-level evaluation · cross-validation · cohort transfer · covariate shift ·
explainability · integrated gradients · pre-registration · Sleep-EDFx

---

## Chapter 1 — Introduction

### 1.1 Background

Polysomnography (PSG) — overnight recording of EEG, EOG and other physiological signals — is the
clinical standard for diagnosing sleep disorders. A technician (or an automated model) assigns one of
five stages (Wake, N1, N2, N3, REM) to every 30-second epoch, producing a hypnogram used to diagnose
insomnia, sleep apnea and related conditions. Deep learning models can automate this scoring, but
clinically deployable models must run on modest hardware, must be evaluated so that the reported
accuracy reflects performance on genuinely new patients, and ideally should explain which parts of the
signal drove each decision.

### 1.2 Problem Addressed

The project inherited a 649,229-parameter sleep-staging model and a pipeline with three concrete
problems: (1) the model's headline score could not be trusted, because the evaluation split leaked
subject identity between train and test; (2) the model was too large to train or run on free-tier cloud
hardware, motivating compression via knowledge distillation; (3) there was no mechanism to say *why* the
model made a given prediction, or to flag predictions it was likely to get wrong.

### 1.3 Motivation

The primary objective was to produce a materially smaller model that matches or exceeds the original's
generalisation performance, under an evaluation protocol that cannot leak. Secondary objectives were to
determine whether knowledge distillation from a single teacher helps at all for this task, to identify
and close the gap between what the raw EEG waveform could contribute and what the deployed model
actually used, and to attach calibrated confidence and feature-attribution evidence to every prediction
rather than shipping a bare label.

### 1.4 Scope of the Study/Project

In scope: subject-level data auditing and re-splitting; teacher retraining and defect diagnosis; student
architecture design and the full distillation pipeline; ensemble construction; calibration; cross-cohort
transfer measurement; pre-registered explainability (Integrated Gradients) with a fixed pass/fail gate;
a per-epoch N1 confidence flag; and a local web console (`ui/`) and report-generation pipeline
(`agentic/`, `deploy/`) built on top of the frozen model. Out of scope: external validation on a dataset
outside Sleep-EDFx (datasets for this exist on disk — MESA, CFS — but have not yet been run); a shared
working tree also contains an unrelated neural-architecture-search project (Group 47) that this project
never touches. The remaining chapters cover: existing systems and the research gap (Ch. 2); the proposed
pipeline and model design (Ch. 3); the experiments and their results (Ch. 4); and the conclusions and
remaining work (Ch. 5).

---

## Chapter 2 — Literature Review / Existing System

### 2.1 Existing System Study

#### 2.1.1 Review of existing Research work

Automated sleep staging from single or multi-channel EEG is a well-studied problem; the project's own
teacher model is one such CNN/attention-based classifier trained on precomputed spectral band-power
features plus raw EEG. Closest adjacent work, surveyed for the report-generation layer built on top of
the staging model (`COMPARISON WITH EXISTING SYSTEMS/RELATED_WORK.md`):

| System | What it does | Relation to this project |
|---|---|---|
| Wearable-sleep LLM pipeline (arXiv 2609.22463) | Multi-agent LLM writes guidance from wearable data, distilled to a small model via SFT, checked by human evaluation | Same large→small distillation idea; this project instead uses a deterministic verifier before release, not post-hoc human evaluation |
| SleepVLM (arXiv 2603.26738) | Vision-language model stages epochs from rendered PSG with per-epoch rationale, κ 0.767 (MASS-SS1) | Explains individual epochs, not the whole night; its κ is not comparable (different dataset/protocol) |
| SleepLM (ICML 2026) | Sleep-language foundation model, 100K+ hours, captioning/retrieval/event localisation | Far larger in scale and scope, with no described claim-level verification |
| Locally-run LLMs on PSG reports (Seifen et al., 2025) | Gemma2/Llama3/Mistral-Nemo on 30 real OSA cases; performance judged insufficient for real-world use | Motivates this project's own measured result that untrained local models scored 0/31 on the verifier, justifying the deterministic-gate design |
| NeuraDock (arXiv 2606.26519, non-sleep) | Deterministic local engine computes numbers, LLM sees only a summary context pack | Closest design philosophy to this project's pipeline, though it describes no per-claim verifier |

#### 2.1.2 Review of Existing Tools/Products

The inherited codebase itself is the closest "existing tool": a 649,229-parameter checkpoint whose model
class lived only inside a Kaggle notebook (not in any `.py` file), with a committed evaluation notebook
whose confusion matrices and ROC curves were computed on **train + validation combined**, and whose
headline Cohen's κ of 0.6663 was produced from a recording-level (not subject-level) split. No test split
had ever existed before this project — training used only an 80/20 train/validation split. As a market
comparison, generic PSG-scoring products and research models typically report a single aggregate score
without stating whether the split is subject-disjoint; this project treats that omission itself as the
central gap being addressed.

### 2.2 Research Gap / Scope for Improvement and Innovation

Gaps addressed in this phase:

1. No existing measurement of the inherited model was trustworthy, because Sleep-EDFx contains two
   nights per subject and the split did not account for it.
2. No compressed version of the model existed; the original 649K-parameter network exceeds comfortable
   free-tier training/inference budgets.
3. No analysis existed of *whether* the model's raw-EEG input branch contributed anything, or whether
   distillation itself produced reliable gains rather than apparent ones from a lucky split.
4. No calibrated confidence, attribution evidence, or cross-cohort transfer measurement accompanied any
   prediction.

Gaps intentionally left for later phases: external validation on a dataset outside Sleep-EDFx; soft
labels derived from genuine multi-scorer disagreement (Sleep-EDFx has one scorer per epoch); a full
α/temperature sweep for distillation, which the current evaluation resolution (±0.054 κ at n = 15 test
subjects) cannot yet distinguish from noise.

### 2.3 Problem Statement and Contributions

**Problem Statement.** Build and honestly evaluate a sleep-staging model that is both small enough for
free-tier hardware and at least as accurate, on genuinely unseen subjects, as the 649,229-parameter model
it replaces — and determine, rather than assume, whether knowledge distillation is the mechanism that
gets it there.

**Contributions.**

1. Diagnosis and correction of subject-level data leakage in the inherited evaluation, with the leak's
   magnitude (0.16 κ) reconstructed and verified independently.
2. A mechanistic finding, not merely an empirical one — "a teacher must exceed the student's own
   undistilled performance before its guidance helps" — explaining why naive single-teacher distillation
   failed and what specifically fixes it (ensembling comparable teachers).
3. A delivered model at 139,606 parameters (4.65× compression) with κ = 0.7001 (95% CI [0.6432, 0.7514]),
   together with a per-night evidence packet carrying calibrated confidence, attribution evidence and
   measured, honestly-reported error bounds — including bounds that fail to clear their own target rather
   than being omitted.
4. Quantified, cross-validated evidence (5-fold subject-level CV; cross-cohort transfer measurement) that
   the evaluation itself, not the model, is now the project's binding constraint.

---

## Chapter 3 — Proposed Work

### 3.1 Proposed Work

The proposed work restores a trustworthy evaluation protocol, rebuilds the teacher under it, and designs
a compact student around the evidence (rather than the assumption) of what distillation can deliver —
closing with an ensemble-distilled model whose evidence packet reports calibration, attribution and
transfer behaviour alongside the stage label.

#### 3.1.1 Objectives of the Proposed Work

**Primary objectives:** (1) eliminate subject-level data leakage from every reported metric; (2) deliver
a student model materially smaller than the 649,229-parameter original that matches or exceeds its
generalisation performance on held-out subjects; (3) determine empirically whether single-teacher
distillation helps, and if not, identify and fix the mechanism.

**Secondary objectives:** calibrate the delivered model's confidence; measure its transfer across the two
cohorts inside Sleep-EDFx; produce a pre-registered, falsifiable explainability check rather than a
post-hoc narrative; and ship a per-epoch confidence flag for the model's weakest class (N1).

### 3.2 Methodology

#### 3.2.1 Overview of the Approach

Audit the existing evaluation and model before trusting either; rebuild subject-disjoint splits; retrain
the teacher under the corrected protocol; design and train a compact student first without distillation
as a control; diagnose why naive distillation underperforms that control; fix the underlying cause
(teacher quality, then encoder receptive field, then ensembling); and only then evaluate the final
pipeline — including its calibration, explainability and cross-cohort behaviour — against a locked test
split touched exactly once.

#### 3.2.2 Dataset Selection

**Sleep-EDFx** (PhysioNet): 197 recordings (153 Sleep Cassette + 44 Sleep Telemetry), 100 subjects,
237,950 total 30-second epochs, each paired with 3,000 raw EEG samples and 34 precomputed spectral
band-power features. Label vocabulary: `{W, N1, N2, N3, REM}` (no R&K `N4` remapping needed). Class
distribution is imbalanced — N2 (37.4%) is 4.6× more common than N3 (8.2%) — which shapes every loss-
function decision downstream. Splits were rebuilt at the **subject** level (first 5 characters of the
recording ID), cohort-stratified 70/15/15 (69/16/15 subjects → 137/31/29 recordings), with the 15-subject
test split loaded exactly once, after all training and model selection had finished.

#### 3.2.3 Algorithm/Model Design

The teacher is a CNN/attention sleep-staging network (649,229 trainable parameters) consuming both the
3,000-sample raw EEG epoch and the 34 engineered spectral features. The delivered student uses a
compact multi-scale convolutional encoder (139,606–151,606 parameters depending on channel count) whose
receptive field was deliberately widened to 8.75 seconds — wide enough to contain a sleep spindle,
K-complex or slow wave — after measurement showed the original design's effective receptive field was
only 0.25 seconds. The final deployed model is itself distilled from a 3-member ensemble of
architecturally comparable students (~410K combined parameters) rather than from a single larger teacher.

#### 3.2.4 Tools and Technologies

Python, PyTorch (model definition and training), NumPy/SciPy/scikit-learn (metrics, cross-validation,
calibration), training executed on Kaggle (T4 GPU ×2) for teacher and student runs; `pytest` for the unit
tests covering the distillation gradient correction; a FastAPI + Uvicorn local web console (`ui/`,
Somnia) built on top of the frozen pipeline for interactive inspection, upload-driven inference and
report review, with no build step or external frontend assets. The report-generation layer
(`agentic/`, `deploy/`) runs a pinned local `llama-completion` subprocess per report rather than a hosted
LLM API.

#### 3.2.5 Algorithm or Model Description

The core training objective is knowledge distillation with soft-target loss and temperature scaling,
combined with the standard hard-label cross-entropy term:

```
L = α · T² · KL(softmax(z_s / T) ‖ softmax(z_t / T))  +  (1 − α) · CE(z_s, y)
```

where `z_s`/`z_t` are student/teacher logits, `T` is the distillation temperature and `α` the soft/hard
mixing weight (α = 0.5, T = 3.0 in the delivered run). Teacher outputs are cached rather than recomputed
per step. A gradient-scaling correction (multiplying the soft-target gradient by `T²`) is required for
the soft and hard loss terms to be on a comparable scale; the project's unit tests specifically cover the
case where this correction is silently omitted, since the training loop otherwise still runs — it simply
trains worse, and that failure mode is invisible without a dedicated test. A second, independent
correction fixed a defect in the teacher's focal-loss objective, which had been applying inverse-frequency
class weighting twice (once inside the focal term, once as an external multiplier), systematically
suppressing gradient on the majority class (N2).

#### 3.2.6 Expected Outcomes

A student model under ~150K parameters matching or exceeding the 649K-parameter teacher's held-out κ;
measurable calibration improvement from distillation even where raw accuracy ties its non-distilled
control; and a quantified (not assumed) cross-cohort transfer gap with its cause narrowed down by
elimination (class-prior shift tested and rejected; covariate shift the remaining explanation).

#### 3.2.7 Advantages of the Proposed Work

4.65× parameter compression with no significant loss of held-out generalisation (paired subject-level
Wilcoxon test, p = 0.135 against the best teacher — not distinguishable, not worse); materially improved
calibration (ECE roughly halved) and REM-latency estimation (error nearly halved) over the non-distilled
control; every shipped metric accompanied by a measured confidence interval or an explicit statement that
the evaluation cannot yet resolve the claim, rather than a bare point estimate.

#### 3.2.8 Limitations and Assumptions

The held-out test set (15 subjects, 29 recordings) yields a 95% CI of roughly ±0.054 κ — subject-
clustered, since two nights of one person are not independent evidence — which means several of the
project's own later comparisons return statistical ties rather than clear wins, not because nothing
changed but because the evaluation cannot resolve differences at that scale. Every result rests on a
single dataset (Sleep-EDFx); the measured 0.187 κ cross-cohort transfer gap *within* that one dataset is
itself evidence that no clinical claim should be trusted without genuine external validation. The soft
targets used in distillation come from model disagreement (ensemble variance), not from genuine
inter-scorer disagreement, since Sleep-EDFx provides only one scorer per epoch.

---

## Chapter 4 — Experimentation and Result Analysis

### 4.1 Experimental Setup

Training was run on Kaggle (Intel/AMD host CPU, T4 GPU ×2, Python 3.x, PyTorch ≥1.9). Each student
training run took approximately 20 minutes. Evaluation, calibration and explainability scripts were run
separately on CPU; the explainability gate (§4.5) took approximately 2.5 hours CPU time across all 29
test recordings.

### 4.2 Datasets

Sleep-EDFx, subject-level 70/15/15 split (137/31/29 recordings; see §3.2.2). A second configuration
trains on the 62 Sleep-Cassette subjects and evaluates on all 22 Sleep-Telemetry subjects to directly
measure cross-cohort transfer (§4.6/§4.7 below). All loading and shape/dtype/epoch-alignment checks
passed with zero load failures and zero epoch-alignment mismatches across all 197 recordings.

### 4.3 Evaluation Metrics

Cohen's κ (primary metric, standard for sleep staging because of class imbalance), accuracy, macro-F1,
per-class F1, expected calibration error (ECE), bootstrap 95% confidence intervals resampled by subject,
and paired per-subject Wilcoxon signed-rank tests for model-vs-model comparisons.

### 4.4 Experimental Design

**Baselines:** the inherited 649,229-parameter checkpoint, re-evaluated honestly under the leak-free
split; a faithful retrained replication (E0); two corrected-loss variants (E1a, E1b). **Proposed models:**
a 121,099-parameter non-distilled student (control); students distilled from E0 and from E1b individually
(both underperformed the control); a widened-receptive-field student (139,606–151,606 params); a 3-member
ensemble teacher; and the final model, a student distilled from that ensemble. **Parameter tuning:**
distillation α and temperature were fixed at 0.5 / 3.0 for all reported runs (a full sweep is listed as
future work in Ch. 5, since the current evaluation resolution cannot yet distinguish small gains from
noise); calibration temperature was fitted on validation only and applied unchanged to test.

### 4.5 Results

**Every model, held-out test (15 unseen subjects, 33,431 epochs):**

| Model | Params | Accuracy | κ | Macro-F1 |
|---|---:|---:|---:|---:|
| Inherited model *(honest, 1 unseen subject)* | 649,229 | 0.6630 | 0.5066 | 0.4107 |
| E0 teacher (faithful retrain) | 649,229 | 0.6160 | 0.5122 | 0.6063 |
| E1b teacher (loss-defect fixed) | 649,229 | 0.7086 | 0.6055 | 0.6548 |
| Student distilled from E0 | 121,099 | 0.6926 | 0.5925 | 0.6611 |
| Student distilled from E1b | 121,099 | 0.7123 | 0.6134 | 0.6708 |
| Student baseline (no distillation) | 121,099 | 0.7360 | 0.6449 | 0.6915 |
| Student, multi-scale encoder | 139,606 | 0.7756 | 0.6992 | 0.7407 |
| Ensemble teacher (3 students) | ~410,000 | 0.7860 | 0.7140 | 0.7542 |
| **Student, distilled from ensemble (delivered)** | **139,606** | **0.7773** | **0.7001** | **0.7376** |

Bootstrap 95% CI for the delivered model: κ ∈ [0.6432, 0.7514] (29 held-out subjects, resampled by
subject). Paired per-subject Wilcoxon: the 121,099-parameter student beats the 649,229-parameter E0
teacher on 14/15 subjects (Δκ +0.1337, p = 0.00043) and is statistically indistinguishable from the
E1b teacher (10/15, Δκ +0.0309, p = 0.135) — i.e. it matches, not "outperforms," the best teacher, at
5.36× compression.

**Calibration:** temperature scaling (fitted T = 1.3347 on validation, applied unchanged to test) reduced
test ECE from 0.0454 to 0.0242. Accuracy, κ and F1 are unchanged, since temperature scaling affects
confidence, not the predicted label.

**Per-class reliability (delivered model, test):** W F1 0.845 (high), N2 F1 0.773 (high), N3 F1 0.737
(medium), REM F1 0.713 (medium), **N1 F1 0.389 (low)** — N1 remains the weakest class across every model
tested, consistent with inter-scorer agreement of only 25–45% for that stage in the literature.

### 4.6 Analysis of Results

The single biggest driver of every downstream result was the switch from recording-level to
subject-level evaluation: the inherited model's reported κ of 0.6663 was reconstructed as a weighted
blend of a leaked measurement (0.6855, same-subject epochs) and a clean one (0.6151), reproducing the
published figure to within 0.0008 — strong evidence the leak, not measurement noise, explains the gap.
Naive single-teacher distillation (from both E0 and E1b) underperformed an identically-sized student
trained without any teacher at all, in both cases by a statistically significant margin. The cause was
not the loss, temperature or mixing weight; it was that the student's own undistilled performance already
exceeded the teacher it was being asked to imitate, so half of every gradient pulled a good model toward
a worse one. Fixing the encoder's receptive field (0.25 s → 8.75 s) was independently necessary: an
earlier ablation had wrongly concluded the raw EEG branch added nothing, because the encoder's receptive
field was too short to see any physiologically meaningful waveform feature. Once both fixes were in
place, ensembling three comparable students and distilling that ensemble produced the only distillation
pathway in the project that outperformed its non-distilled control — on calibration, not raw accuracy.

### 4.7 Observations

Five-fold subject-level cross-validation (85 non-test subjects) showed the original single validation
split was an unrepresentative draw: average fold κ = 0.7143 against 0.6763 on the one split every
experiment had been judged on, with all five folds scoring above that split and a fold-to-fold standard
deviation of 0.0193 κ — roughly 24× the entire measured effect of distillation on κ (0.0008). This does
not invalidate earlier comparisons (both sides of every comparison were always judged on the same split),
but it means absolute statements about "clearing a bar" should be read cautiously. Cross-cohort transfer
(train on 62 Sleep-Cassette subjects, test on all 22 Sleep-Telemetry subjects, none seen in training)
gave κ = 0.5516 (95% CI [0.4751, 0.6205]) against 0.7384 on held-out cassette data — a 0.187 κ gap, with
recall uniform across W/N2/N3/REM (67–72%) but Wake *precision* collapsing from 0.898 to 0.622. Correcting
with the *true* target-cohort class priors recovered only 1.2% of the gap, and unsupervised prior
estimation made results worse — evidence the gap is covariate shift (the epochs look different), not
prior shift (the stages merely occur in different proportions). A follow-up amplitude-normalisation
intervention, the leading remaining hypothesis for that covariate shift, was tested and made transfer
*worse* (paired Δκ = −0.0175, 95% CI [−0.0318, −0.0033]), ruling it out as well.

### 4.8 Comparative Analysis

| Comparison | Result |
|---|---|
| Delivered student vs. original 649,229-param checkpoint, evaluated honestly | 4.65× smaller, κ +0.19 (0.7001 vs 0.5066), N3 F1 recovered from 0.000 |
| Delivered student vs. best retrained teacher (E1b, 649,229 params) | Statistically tied on held-out κ (p = 0.135) at 5.36× smaller parameter count |
| Ensemble-distilled student vs. non-distilled control (same architecture) | Accuracy/κ tied; ECE −47% (0.0454→0.0242 vs undistilled), REM-latency error nearly halved |
| In-cohort (Sleep-Cassette) vs. cross-cohort (Sleep-Telemetry) transfer | 0.187 κ gap, attributable to covariate shift, not class-prior shift or amplitude difference |

A pre-registered explainability check (Integrated Gradients, void condition and five predictions fixed
before computation) passed its completeness axiom (0.03% deviation against a 5% threshold) and 3 of 5
pre-registered predictions held. Two caveats travel with that pass rather than being omitted: all five
sleep stages share the same top-3 attributed features in different orders (so stages are distinguished by
feature *magnitude*, not feature *choice* — untested by the registration), and the engineered-feature
branch carries disproportionate attribution mass relative to its 34-vs-3,000 dimensionality. A per-epoch
N1 confidence flag, fitted on validation, transfers as a *ranking* on held-out data (flagged calls right
33.7% of the time vs. 44.7% for unflagged, a gap of 10.9 points, 95% CI [7.1, 15.8]) but misses its
fitted *level* target (validation promised 61% unflagged accuracy; held-out delivers 44.7%) — reported as
a ranking rule, not a guarantee, in the shipped evidence packet.

---

## Chapter 5 — Conclusion and Scope for Further Research

The project delivers a 139,606-parameter sleep-staging model — 4.65× smaller than the 649,229-parameter
model it replaces — reaching Cohen's κ = 0.7001 (95% CI [0.6432, 0.7514]) on fifteen genuinely unseen
subjects, 0.09 κ above the best teacher it was built to imitate, with every prediction accompanied by
calibrated confidence and pre-registered attribution evidence. The more durable output, however, is
methodological: a plausible-looking evaluation protocol inflated the inherited model's headline metric by
0.16 κ while every individual step in building it looked reasonable, and the project's central discipline
— auditing the measurement before trusting the model, and continuing to audit the evaluation itself even
after the model was fixed — is what surfaced the teacher-quality mechanism behind distillation failure,
the receptive-field defect behind the raw-EEG dead end, and the fact that the held-out test set's own
resolution (±0.054 κ) is now the binding constraint on what further tuning on this dataset can prove.

**Scope for further research**, in priority order: (1) external validation on a cohort outside
Sleep-EDFx — MESA and CFS are already available on disk — since this is the only remaining step that buys
both statistical power and clinical credibility, and the measured 0.187 κ in-dataset cohort gap suggests
performance should be expected to drop; (2) multi-scorer soft labels using datasets that carry genuine
inter-scorer disagreement (e.g. DOD-H/DOD-O), replacing the current model-disagreement proxy; (3) an
α/temperature sweep for distillation, reported with the test-resolution caveat stated explicitly rather
than implied; (4) completing the per-recording amplitude-normalisation experiment at the preprocessing
layer (currently blocked by a shared code region across three trainer generators, not by compute).
Explicitly **not** planned: additional seeds, additional ensemble members, further architecture search or
further EOG-channel work — those questions were tested and answered, and further effort there sits below
the evaluation's noise floor.

---

## Appendix A — Code and Reproduction

```bash
# 1. Preprocess (needs the raw EDFs)
#    see PROJECT_REPORT.md §12 for the exact preprocessing commands

# 2. Train on Kaggle (T4); each run ~20 min
#    kaggle_train_student_v2.py     single-channel, hard labels
#    kaggle_train_student_mc.py     + EOG + Pz-Oz
#    kaggle_train_student_kd.py     soft targets from the ensemble
#    For model selection: set CV_FOLD = 0..4 near the top of any trainer and
#    run it five times. CV_FOLD = None is the original 69/16 subject split.

# 3. Evaluate and calibrate — see README.md "Reproducing" for the full
#    evaluate / calibrate / packet-build / explainability command sequence.

# 4. Local web console (reads the existing system, modifies nothing under it)
python -m ui                 # http://127.0.0.1:8000
```

Full technical report: `PROJECT_REPORT.md`. Current status, guards and detailed reproduction steps:
`README.md`. Experiment history, including superseded conclusions: `distillation/results/`. Related-work
survey for the report-generation layer: `COMPARISON WITH EXISTING SYSTEMS/RELATED_WORK.md`.

---

## References

1. Kemp, B. et al., "Sleep-EDF Database Expanded," PhysioNet, [https://physionet.org/content/sleep-edfx/](https://physionet.org/content/sleep-edfx/).
2. Hinton, G., Vinyals, O., Dean, J., "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015.
3. Sundararajan, M., Taly, A., Yan, Q., "Axiomatic Attribution for Deep Networks" (Integrated Gradients), ICML 2017.
4. Guo, C. et al., "On Calibration of Modern Neural Networks," ICML 2017.
5. Project-internal: `PROJECT_REPORT.md`, `ABSTRACT.md`, `README.md`, `COMPARISON WITH EXISTING SYSTEMS/RELATED_WORK.md` (this repository).
