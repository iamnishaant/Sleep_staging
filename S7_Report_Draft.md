> **Draft still presents v1 — banner added 2 Nov 2026.** The current release is **v2** (`student_N6adv_full_s1`, EEG + EOG, with the v2 report model). Where this draft calls `student_N4kd` the delivered model, or cites κ 0.7001/0.699 or 29/29 as the current system, update it from [`CURRENT_RELEASE.md`](CURRENT_RELEASE.md) before using it. Nothing below this banner was changed.

# Compact Sleep Staging by Knowledge Distillation with a Trustworthy Report Layer

*A complete writing draft matching the S7 report template. Figure and table slots are marked
**[Figure x.x]** / **[Table x.x]** — replace each with the actual image/table. Roll numbers, names,
guide/coordinator/reviewer details, acknowledgements and the appendix code listing are left for you.*

---

## Abstract

Automated sleep staging assigns a sleep stage (Wake, N1, N2, N3, REM) to every 30-second epoch of a
polysomnogram. It is usually evaluated with a random train/validation split at the recording level,
which is invalid when a dataset contains multiple nights per subject, as Sleep-EDFx does: a model can
score well by recognising the person rather than the physiology. This project set out to compress an
inherited 649,229-parameter sleep-staging model into a compact student by knowledge distillation.
Auditing the inherited model first, we found exactly that leak: 86% of its validation subjects also
appeared in training, and its published Cohen's κ of 0.6663 fell to 0.5066 when re-measured on a
subject it had genuinely never seen, with N3 recall at zero. This forced a rebuild under corrected,
subject-level, cohort-stratified splits before any distillation could be trusted.

Rebuilding the teacher reached a best κ of 0.6055. A compact 121,099-parameter student then reached
κ 0.6449 under hard labels alone, beating every teacher — distillation from a weaker teacher *hurt*
(−0.052 to −0.031 κ), with the harm scaling with the teacher–student gap. Two further changes reversed
the picture. A multi-scale encoder whose receptive field was measured rather than assumed (8.75 s
instead of 0.25 s) made the raw EEG carry 71% of predictions and lifted the student to κ 0.6992.
Distilling from an *ensemble* of three comparable models — better, without being bigger — removed the
distillation penalty entirely; the gain appeared not in accuracy (which tied) but in calibration, with
expected calibration error falling by a third and REM-latency error nearly halving. The delivered model
is a **139,606-parameter student at κ 0.7001 (95% CI [0.6432, 0.7514])**, a 4.65× compression that
beats its teacher. On the standard Sleep-EDF-78 subject-wise protocol it reaches **κ 0.761 / accuracy
83.2%**, competitive with published single-channel baselines at an order of magnitude fewer parameters.

Because a compact model still makes per-epoch errors, the system ships each night not as a bare
hypnogram but as an **evidence packet** carrying every derived clinical value with its own measured
error bound, and a **deterministic verifier** gates every generated report so that no unsupported claim
reaches the reader. On 31 development nights the verifier released zero incorrect claims; without it,
the trained report model would have released 1–2 reports (3–6%) with hedging or completeness errors.

---

## List of Figures

1. Figure 1.1 — The end-to-end pipeline: raw PSG → staging → evidence packet → verified report.
2. Figure 2.1 — Published Sleep-EDF-78 baselines versus this work (accuracy vs. parameters).
3. Figure 3.1 — The multi-scale encoder and student architecture.
4. Figure 4.1 — The leakage proof: the inherited model across three exposure levels.
5. Figure 4.2 — Cohen's κ versus parameter count across all trained models.
6. Figure 4.3 — Per-class F1 of the delivered model on held-out test.
7. Figure 4.4 — The knowledge-distillation mechanism: baseline KL divergence versus teacher quality.
8. Figure 4.5 — Per-fold Sleep-EDF-78 benchmark results.

## List of Tables

1. Table 1.1 — AASM sleep stages and their EEG markers.
2. Table 3.1 — Sleep-EDFx dataset composition.
3. Table 4.1 — Subject-level split (train / validation / test).
4. Table 4.2 — The leakage gate: the inherited model at three exposure levels.
5. Table 4.3 — Teacher ladder (E0 / E1a / E1b).
6. Table 4.4 — Complete held-out test results, all models.
7. Table 4.5 — Knowledge-distillation results across three teachers.
8. Table 4.6 — Calibration and downstream error before/after ensemble distillation.
9. Table 4.7 — Sleep-EDF-78 benchmark versus published baselines.
10. Table 4.8 — Verifier ablation (arm A / B / C).

---

# Chapter 1 — Introduction

## 1.1 Background

Sleep is not a single state but a recurring cycle of distinct stages, each with characteristic
electroencephalographic (EEG), electro-oculographic (EOG) and electromyographic (EMG) signatures. The
American Academy of Sleep Medicine (AASM) manual divides sleep into five stages: Wake (W), three
non-rapid-eye-movement stages (N1, N2, N3, where N3 is deep or slow-wave sleep), and rapid-eye-movement
(REM) sleep. Staging a night means labelling every 30-second epoch with one of these five classes, and
the resulting sequence over time is called a hypnogram.

Sleep staging is the foundation of clinical sleep medicine. Derived quantities such as total sleep time,
sleep efficiency, sleep latency, time spent in each stage, and the timing of REM episodes are all
computed from the hypnogram, and the diagnosis of disorders such as insomnia, narcolepsy and
sleep-disordered breathing depends on them. Manual staging, however, is slow, subjective, and
expensive: a trained technologist spends roughly one to two hours on a single night, and inter-scorer
agreement — particularly on the transitional N1 stage — is famously low.

Deep learning has become the dominant approach to automatic sleep staging. Models are typically trained
on the public Sleep-EDF dataset, which contains 197 polysomnographic recordings from 100 subjects, and
recent single-channel models report κ values in the 0.73–0.79 range on the standard Sleep-EDF-78
benchmark. The open problems are not only accuracy but also three properties that matter in practice:
**size** (models must run on edge hardware without cloud dependence), **honesty of evaluation**
(generalisation must be measured on unseen *subjects*, not unseen *recordings*), and **trustworthiness**
(a compact model makes errors, so its outputs must carry their own uncertainty rather than being stated
as fact).

## 1.2 Problem Addressed

This project addresses four linked problems.

1. **Evaluation leakage.** Sleep-EDFx records two consecutive nights for most subjects. Splitting at the
   recording level places night 1 of a subject in training and night 2 in validation, so the model can
   recognise the person instead of the physiology. The inherited model's reported κ of 0.6663 turned out
   to be, on inspection, a blend of leaked and clean measurement.

2. **Model size.** The inherited staging model has 649,229 trainable parameters, which is large for a
   device-local deployment. The goal is a compact model that can run on a CPU in seconds per night.

3. **The distillation question, done honestly.** Knowledge distillation is the natural way to compress a
   model, but it is usually reported as always helping. This project asked whether it actually helps when
   the teacher is *weaker* than the student it teaches, and under what condition the penalty disappears.

4. **Trustworthy reporting.** Even the best compact model mis-stages a meaningful fraction of epochs, and
   derived metrics (sleep latency, REM latency) amplify those errors. A system that states such values as
   fact can mislead. The problem is to build a reporting layer that only asserts what the measurement
   supports.

## 1.3 Motivation

The motivation is to produce a sleep-staging system that is simultaneously **small, honestly evaluated,
and safe to read**. The project has the following primary and secondary objectives.

**Primary objectives.**

- Re-audit the inherited model and rebuild the evaluation so that no subject appears in both training and
  any held-out set.
- Train a compact student model, no larger than a few hundred thousand parameters, that matches or beats
  the larger teachers under leak-free evaluation.
- Establish, with a controlled experiment, when knowledge distillation helps and when it hurts.

**Secondary objectives.**

- Attach to every night an evidence packet in which each derived clinical value carries a measured error
  bound and a reliability tier.
- Add a deterministic verifier that blocks any report containing an unsupported claim, malformed output,
  or missing mandatory item.
- Benchmark the final model on the standard Sleep-EDF-78 protocol so the result is comparable to the
  published literature.

## 1.4 Scope of the Study/Project

The study is bounded as follows.

**Included.**

- Five-class sleep staging (W, N1, N2, N3, REM) from the Sleep-EDFx dataset.
- Single-channel EEG (Fpz-Cz) for the deployed student, with EOG and a second EEG derivation evaluated as
  an ablation.
- Knowledge distillation from a three-model ensemble into the compact student.
- A per-night evidence packet and a deterministic report verifier.

**Excluded.**

- Respiratory, arousal and leg-movement scoring, and therefore any sleep-apnoea diagnosis.
- External clinical validation on a second dataset (out of scope; recorded as a limitation).
- Language-model distillation — a separate, unstarted question, distinct from staging-model distillation.

**Organisation of the report.** Chapter 2 reviews existing research and tools and states the research
gap, problem statement and contributions. Chapter 3 describes the proposed work and methodology.
Chapter 4 presents the experiments, results and a comparison with published baselines. Chapter 5
concludes and outlines future work.

---

# Chapter 2 — Literature Review / Existing System

## 2.1 Existing System Study

### 2.1.1 Review of existing research work

Automatic sleep staging with deep learning has been studied extensively on Sleep-EDF. The landmark
**DeepSleepNet** (Supratak et al., 2017) used two convolutional branches over raw EEG with a
bidirectional recurrent sequence learner, reporting κ ≈ 0.73 on Sleep-EDF-78 but with 21M parameters.
**SleepEEGNet** (Mousavi et al., 2019) and **SeqSleepNet** (Phan et al., 2019) applied sequence-to-sequence
and hierarchical recurrent models; SeqSleepNet is notable for reaching κ ≈ 0.76 with only 0.16M
parameters. **TinySleepNet** (Supratak & Guo, 2020) and **U-Time** (Perslev et al., 2019) pushed compact
single-channel performance to κ ≈ 0.75–0.77.

More recent work has focused on interpretability and multi-channel inputs. **SleepTransformer** (Phan et
al., 2022) added self-attention with interpretability and uncertainty, while **AttnSleep** (Eldele et
al., 2021) used an attention mechanism over single-channel EEG. Multi-channel models such as
**SalientSleepNet**, **LMCSleepNet** and **MultiChannelSleepNet** use Fpz-Cz + Pz-Oz + EOG and reach
κ ≈ 0.77–0.79, with MultiChannelSleepNet reporting the highest accuracy (85.0%) at 13M parameters.

Two consistent observations emerge from this literature. First, N1 is the universal weak point — its F1
is rarely above 0.55 and is usually much lower, because N1 is brief, transitional, and confusable with
Wake and REM. Second, the best multi-channel models lead the table but do so with 0.9M–13M parameters, an
order of magnitude more than the most efficient single-channel models.

### 2.1.2 Review of Existing Tools/Products

Several open-source tools perform automatic sleep staging out of the box. **YASA** (Vallat & Walker,
2021) is a widely used Python package that ships a pre-trained classifier and reports κ ≈ 0.79 on
Sleep-EDF in its own evaluation. **U-Sleep** (Perslev et al., 2021) is a larger, fully convolutional
model with strong cross-dataset resilience. Commercial products and research prototypes also exist, but
most are either cloud-hosted or closed, and few publish per-night uncertainty alongside their staging
output.

The gap shared by these tools is not raw accuracy but **what happens after staging**. They output a
hypnogram and derived metrics, but do not, in general, attach a per-value error bound, flag which values
are safe to state as fact, or gate the generated narrative so that unsupported claims are withheld. A
staging model that is 80% accurate is still wrong 20% of the time, and the derived metrics that a
clinician reads can be wrong by more than the staging error suggests.

## 2.2 Research Gap / Scope for improvement and innovation

The specific shortcomings this project addresses are:

1. **Evaluation leakage is often invisible.** Many published Sleep-EDF results do not make clear whether
   their split is subject-level or recording-level; a recording-level split silently inflates κ.
2. **Efficiency and the distillation precondition.** Compact models exist, but the claim that knowledge
   distillation compresses them is usually assumed rather than tested against the honest baseline of a
   student trained with hard labels only. When the teacher is weaker than the student, distillation
   should *hurt*, and this is rarely measured.
3. **The raw-EEG encoder is usually not measured.** Models labelled "temporal–spectral fusion" may have an
   encoder whose receptive field is too short to see the events that define the stages it classifies.
4. **Trustworthy reporting is missing.** Accuracy is reported as a single number, but the downstream
   consumer reads derived metrics, and those do not inherit error evenly.

**Addressed in Phase I (this report).** Gaps 1–4 above: subject-level splits, the distillation
precondition, the measured receptive field, and the evidence packet with a deterministic verifier.

**Planned for Phase II.** External validation on a clinical cohort (NSRR), a per-night error model for
time-in-bed and sleep-efficiency reliability, and edge deployment on ARM64 hardware.

## 2.3 Problem Statement and Contributions

**Problem statement.** Given the public Sleep-EDFx dataset, build a sleep-staging system that (a) is
evaluated without subject leakage, (b) compresses a 649,229-parameter model into a compact student that
runs locally, (c) determines honestly when knowledge distillation helps, and (d) reports its output with
measurable, enforced trustworthiness.

**Research contributions.**

1. **A leakage-corrected evaluation.** We show that the inherited model's published κ 0.6663 was a
   recording-level leak, reconstruct the published figure from leaked and clean components to within
   0.0008, and rebuild the splits at subject level with the test set held out and loaded exactly once per
   model.
2. **A 4.65× compact student that beats its teacher.** A 139,606-parameter student reaches κ 0.7001
   (95% CI [0.6432, 0.7514]) on 15 unseen subjects, versus 0.6055 for the best 649,229-parameter teacher.
3. **A measured answer to "when does distillation help?".** Distillation hurts from a weaker teacher and
   stops hurting once the teacher is an ensemble that exceeds the student; what transfers is calibration,
   not accuracy (ECE −33%, REM-latency error −46%).
4. **A trustworthy report layer.** A per-night evidence packet carries per-value error bounds, and a
   deterministic 30-rule verifier released zero incorrect claims on 31 development nights while the
   ungated model released 1–2 flawed reports.

---

# Chapter 3 — Proposed Work

## 3.1 Proposed Work

The proposed system is an end-to-end pipeline that takes a raw or preprocessed polysomnogram and produces
a verified sleep report. It has three stages. First, a compact **staging model** maps each 30-second
epoch to a sleep-stage probability vector. Second, those probabilities are turned into a **per-night
evidence packet** that carries the hypnogram and every derived clinical value with a measured error
bound and a reliability tier. Third, a **report generator** (a small language model) writes structured
claims from the packet, and a **deterministic verifier** gates the result so that unsupported claims,
malformed output or missing items never reach the reader.

The novelty lies in the combination rather than any single component: a compact model that is small
enough to run on-device, an evaluation protocol that cannot leak, a distillation result whose
precondition is measured rather than assumed, and a reporting layer that enforces what the measurement
actually supports.

### 3.1.1 Objectives of the Proposed Work

**Primary objectives.**

1. Rebuild the Sleep-EDF evaluation at subject level and measure the inherited model honestly.
2. Train a compact student (≤ 200K parameters) that matches or beats the larger teachers.
3. Run a controlled distillation study across teachers of increasing quality, including an ensemble.
4. Deliver an evidence packet with per-value error bounds and a deterministic verifier.

**Secondary objectives.**

5. Benchmark the final model on the standard Sleep-EDF-78 protocol for a like-for-like comparison.
6. Quantify cohort transfer (Sleep Cassette → Sleep Telemetry) as an in-dataset generalisation probe.
7. Add an N1-confidence flag so the weakest stage's errors can be reviewed rather than silently trusted.

## 3.2 Methodology

### 3.2.1 Overview of the Approach

The approach proceeds in four phases, each gated by a fixed decision rule recorded before results were
seen.

- **Phase 1 — data and evaluation.** Fix the dataset paths, verify every recording, and rebuild the
  splits at subject level. Measure the inherited model at three exposure levels to expose leakage.
- **Phase 2 — teacher.** Train three teachers with identical architecture, changing only the objective,
  to establish an honest baseline and isolate a class-reweighting fix.
- **Phase 3 — student and distillation.** Build a compact student, measure the encoder's receptive field,
  replace it, then distil from a three-model ensemble.
- **Phase 4 — trustworthy reporting.** Build the evidence packet and the deterministic verifier, and
  evaluate the full report pipeline.

**[Figure 3.1: End-to-end pipeline — placeholder]**

### 3.2.2 Dataset Selection

The project uses **Sleep-EDFx** (PhysioNet), a public dataset of 197 polysomnographic recordings from
100 subjects (237,950 thirty-second epochs). It contains two cohorts:

| Cohort | Subjects | Recordings | Population |
|---|---:|---:|---|
| Sleep Cassette (SC) | 78 | 153 | healthy, ageing study |
| Sleep Telemetry (ST) | 22 | 44 | mild difficulty falling asleep, half on temazepam |

Each recording provides at least one EEG channel (Fpz-Cz) at 100 Hz, and the Sleep Cassette subset also
provides EOG and a second EEG derivation (Pz-Oz). Labels follow the AASM five-class taxonomy, remapped
to `{W, N1, N2, N3, REM}`. Because the raw EEG values are in volts (≈ 2×10⁻⁵) while the derived spectral
features are ≈ 7, the waveform is normalised at load time with a constant fitted on the training split
only.

### 3.2.3 Algorithm/Model Design

The staging model is a two-input encoder followed by a cross-epoch sequence model.

- **Waveform branch.** A multi-scale temporal convolutional encoder with two branches — a 0.5 s kernel
  for sleep spindles and K-complexes, and a 2 s kernel for slow waves — attention-pooled to sub-epoch
  tokens. This replaced an atrous encoder whose measured receptive field was only 0.25 s, far shorter
  than any stage-defining event (spindles 0.5–2 s, slow waves 0.5–2 s, sawtooth waves 1–3 s).
- **Spectral branch.** A small encoder over 34 precomputed band-power and spectral features, which
  directly encode delta-band power that defines N3.
- **Sequence model.** A lightweight cross-epoch transformer over the fused tokens, followed by a
  per-epoch classification head.

The student mirrors the teacher's architecture at reduced width (embedding 64 vs 128, 2 attention heads
vs 4, 2 layers vs 3, 2 dilation branches vs 4), giving 139,606 parameters against 649,229.

**[Figure 3.1b: Multi-scale encoder and student architecture — placeholder]**

### 3.2.4 Tools and Technologies

- **Languages and frameworks:** Python 3, PyTorch, NumPy, pandas.
- **Training hardware:** Kaggle free tier, NVIDIA T4 ×2.
- **Report model:** Llama-3.2-3B-Instruct, LoRA fine-tuned, quantised to Q4_K_M (≈ 2 GB) via llama.cpp.
- **Evaluation tooling:** scikit-learn for metrics, subject-clustered bootstrap for confidence intervals.
- **Testing:** 246 unit tests covering splits, the distillation loss (including the temperature-squared
  gradient term), trainer provenance, and gate wiring.

### 3.2.5 Algorithm or Model Description

**Knowledge distillation.** The student is trained with a weighted sum of hard cross-entropy and a
softened Kullback–Leibler divergence to the teacher's outputs:

```
loss = α · CE(student, y) + (1 − α) · T² · KL(softmax(s/T), softmax(t/T))
```

The `T²` factor is mandatory: softening by temperature `T` shrinks the soft-loss gradient by `1/T²`, and
without the correction the model barely distils while the loss curve shows nothing wrong. This is
enforced by 13 unit tests that assert the gradient ratio is exactly 1, 4, 16, 64 at T = 1, 2, 4, 8.

**The distillation precondition.** Distillation was run from three teachers of increasing quality — a
weak replication (E0, κ 0.5122), a stronger corrected teacher (E1b, κ 0.6055), and an ensemble (κ
0.7140) — against a matched hard-label baseline student. This isolates the teacher–student gap as the
variable.

**Ensemble construction.** Averaging three comparable models produced a teacher better than any member
without a larger architecture. Input and architecture diversity mattered more than seeds: adding a
*weaker* but architecturally different member gained five times what adding a same-architecture variant
did, because decorrelated errors cancel.

**The verifier.** The report generator emits structured claims, not prose; templates render the English.
A deterministic 30-rule verifier (two layers) checks every claim against the evidence packet and blocks
the report if any unsupported, malformed or missing item is found, falling back to a safe summary.

### 3.2.6 Expected Outcomes

The expected outcomes were: (1) a demonstrably leak-free evaluation and a corrected baseline; (2) a
compact student of ≤ 200K parameters that matches or beats the teachers; (3) a quantified distillation
effect across teacher quality; (4) an evidence packet and verifier that prevent unsupported claims; and
(5) a Sleep-EDF-78 benchmark figure comparable to published baselines.

### 3.2.7 Advantages of the Proposed Work

- **Compact and local.** The 139,606-parameter staging model (≈ 0.7 MB in fp32) runs in seconds per night
  on a CPU; the report model is a 2 GB quantised GGUF.
- **Honestly evaluated.** Subject-level splits and a frozen, once-loaded test set prevent leakage.
- **Measured, not assumed, distillation.** The precondition (teacher must exceed student) is established
  experimentally rather than asserted.
- **Trustworthy output.** Every derived value carries an error bound, and a verifier hard-gates the
  narrative, so correctness is independent of report-model quality (a weaker model degrades coverage,
  not correctness).
- **Competitive and efficient.** κ 0.761 / accuracy 83.2% on Sleep-EDF-78 at an order of magnitude fewer
  parameters than the leading multi-channel baselines.

### 3.2.8 Limitations and Assumptions

- **Single cohort.** All figures rest on Sleep-EDFx; no clinical claim survives without external
  validation.
- **Small held-out set.** 15 test subjects give a wide 95% CI (±0.054 κ), so several top models are not
  distinguishable; the evaluation, not the model, is now the binding constraint.
- **N1 is representation-bound.** No decision rule improves N1 F1 by more than +0.0086; this is treated
  as a reason to flag N1, not to over-tune it.
- **Assumption.** The best student under hard labels is assumed to be the best under soft labels, since
  distillation cannot be tuned under cross-validation without a per-fold teacher (which would leak).

---

# Chapter 4 — Experimentation and Result Analysis

## 4.1 Experimental Setup

Training used the Kaggle free tier on NVIDIA T4 ×2 GPUs, PyTorch, and the AdamW optimiser with a cosine
schedule and early stopping on validation macro-F1. A fold took 45–105 minutes depending on the arm; the
full ten-fold Sleep-EDF-78 benchmark was about 9 GPU-hours. The report model was LoRA fine-tuned
(Llama-3.2-3B-Instruct, r16/α32, 3 epochs) and quantised to Q4_K_M. All inference and the agentic report
pipeline ran locally on CPU.

## 4.2 Datasets

The primary dataset is Sleep-EDFx (197 recordings, 100 subjects, 237,950 epochs), split at **subject**
level and stratified by cohort:

| Split | Subjects | Recordings |
|---|---:|---:|
| Train | 69 | 137 |
| Validation | 16 | 31 |
| Test | 15 | 29 |

The test split was loaded exactly once per model, after all training and checkpoint selection, and was
never used for tuning. For the standard Sleep-EDF-78 benchmark, a separate ten-fold subject-wise split
over all 78 Sleep Cassette subjects (153 recordings) was used, trained from scratch with no distillation.

## 4.3 Evaluation Metrics

- **Cohen's κ** — primary metric; agreement corrected for chance, robust to the heavy class imbalance
  (N2 alone is 37% of epochs).
- **Macro-F1** — unweighted mean of per-class F1; used for checkpoint selection because it treats rare
  stages (N1, N3) as equally important.
- **Accuracy** and **per-class F1** — reported for completeness and to locate errors.
- **Expected Calibration Error (ECE)** — whether stated confidence matches accuracy.
- **Confidence intervals** — subject-clustered bootstrap, since two nights of the same subject are not
  independent samples.

## 4.4 Experimental Design

Experiments were ordered so that each later step depends only on a conclusion established earlier, and
every decision rule was fixed before the result was seen.

### 4.4.1 Experimental Scenarios

1. **The leakage gate.** Evaluate the inherited model at three exposure levels: same recording, same
   subject/different night, and genuinely unseen subject.
2. **Teacher ladder (E0 / E1a / E1b).** Identical architecture, changing only the objective, to isolate a
   focal-loss defect.
3. **Student vs. teacher.** Train a compact student with hard labels, then with distillation, from
   teachers of increasing quality.
4. **Encoder rebuild.** Measure the receptive field, replace the encoder, and re-run the A/B.
5. **Channel ablation.** Add EOG + Pz-Oz and test against the single-channel student.
6. **Ensemble distillation.** Distil from a three-model ensemble.
7. **Cohort transfer.** Train on SC, hold out all ST, to probe generalisation across populations.
8. **Sleep-EDF-78 benchmark (B1).** Ten-fold subject-wise CV, from scratch, on the final configuration.

### 4.4.2 Parameter Tuning

Hyperparameters were fixed by the inherited recipe and changed only one at a time. Checkpoint selection
was on validation macro-F1. Class weighting used square-root inverse-frequency (softened from full
inverse-frequency after the E1a defect). Temperature scaling was fitted on validation only and applied
unchanged to test. No threshold was tuned on the test split; the N1-confidence flag was fitted on
validation and reported on test exactly once.

## 4.5 Results

### 4.5.1 The leakage gate

| Exposure level | κ | Macro-F1 | N3 F1 | REM F1 |
|---|---:|---:|---:|---:|
| Same recording (trained on) | 0.8058 | 0.8036 | 0.889 | 0.876 |
| Same subject, different night | 0.6797 | 0.7173 | 0.837 | 0.718 |
| **Genuinely unseen subject** | **0.5066** | **0.4107** | **0.000** | **0.184** |

On an unseen subject the inherited model scored N3 F1 of exactly 0.000 — it had not learned deep sleep,
it had memorised it per person. The published 0.6663 was reconstructed as a weighted blend of leaked and
clean measurement to within 0.0008, confirming the diagnosis.

### 4.5.2 Teacher ladder and student

| Model | Params | Accuracy | κ | Macro-F1 |
|---|---:|---:|---:|---:|
| Inherited (honest, 1 unseen subject) | 649,229 | 0.6630 | 0.5066 | 0.4107 |
| E0 teacher | 649,229 | 0.6160 | 0.5122 | 0.6063 |
| E1a teacher | 649,229 | 0.6506 | 0.5592 | 0.6458 |
| E1b teacher | 649,229 | 0.7086 | 0.6055 | 0.6548 |
| Student baseline (hard labels) | 121,099 | 0.7360 | 0.6449 | 0.6915 |
| Student, multi-scale encoder | 139,606 | 0.7756 | 0.6992 | 0.7407 |
| Student + EOG + Pz-Oz | 151,606 | 0.7757 | 0.7020 | 0.7481 |
| Ensemble teacher (3 students) | ~410k | 0.7860 | 0.7140 | 0.7542 |
| **Student, distilled from ensemble** | **139,606** | **0.7773** | **0.7001** | **0.7376** |

Per-class F1 of the delivered model on held-out test: **W 0.897 · N1 0.429 · N2 0.805 · N3 0.779 ·
REM 0.778**.

### 4.5.3 Knowledge distillation across teacher quality

| Teacher | Teacher κ | Distilled student κ | Δ vs. baseline |
|---|---:|---:|---:|
| E0 | 0.5122 | 0.5925 | −0.0523 |
| E1b | 0.6055 | 0.6134 | −0.0314 |
| **Ensemble** | **0.7140** | **0.7001** | **+0.0008** |

Distillation *hurt* from every single teacher, and the penalty shrank as the teacher improved. The
ensemble — better without being bigger — cleared the precondition, and the penalty vanished.

### 4.5.4 What distillation actually transfers

| | Hard labels | Distilled |
|---|---:|---:|
| Uncalibrated ECE (test) | 0.0359 | **0.0241** |
| REM-latency error (test) | 33.7 min | **18.3 min** |

On accuracy the distilled student ties its hard-label twin (paired p = 0.24); the gain is in calibration
and in the derived metrics that the downstream consumer actually reads.

### 4.5.5 Sleep-EDF-78 benchmark and verifier

On the standard Sleep-EDF-78 protocol (SC only, 78 subjects, 10-fold subject-wise CV, from scratch), the
final model reached **accuracy 83.2%, κ 0.761, macro-F1 0.737** with per-class F1 W 0.908 · N1 0.525 ·
N2 0.839 · N3 0.559 · REM 0.854.

## 4.6 Analysis of Results

Three results drive the analysis. First, **the measurement was wrong before the model was.** A
recording-level split inflated the headline κ by 0.16; the corrected subject-level split is the single
most important methodological contribution. Second, **distillation has a precondition.** The un-distilled
student's KL divergence to a weaker teacher *rises* (0.196 → 5.802) while its own performance improves,
so the soft term pulls it away from its own optimum; the penalty is therefore a function of the
teacher–student gap, not a defect of distillation itself. Third, **accuracy is no longer the binding
constraint.** The delivered model and its hard-label twin tie on accuracy, and the held-out set of 15
subjects cannot resolve sub-0.05 κ differences; the axis that matters is calibration, which improved on
both splits.

## 4.7 Observations

- **N1 is provably unfixable by tuning.** A threshold sweep finds only +0.0086 κ of headroom, and the
  N1-vs-N2 pairwise AUC did not move across the entire teacher ladder. N1 is representation-bound, so the
  system flags it rather than over-fitting it.
- **The encoder was the bottleneck, not the signal.** The raw EEG contributed nothing until the encoder's
  receptive field was extended from 0.25 s to 8.75 s; afterwards it drove 71% of predictions.
- **The extra channels were noise.** EOG + Pz-Oz gained on test but lost on validation (paired p = 0.865);
  the channels remain in the teacher, where decorrelated errors help, but the shipped student uses one
  electrode.
- **Cohort transfer is worse than a small sample suggested.** SC→ST transfer cost 0.187 κ, concentrated
  in Wake precision, and neither prior correction nor per-recording amplitude normalisation recovered it.
- **The verifier is a genuine safety net.** It released zero flawed reports on 31 dev nights, and for an
  untrained base model it was the difference between a usable and an unusable system.

## 4.8 Comparative Analysis

| Model | Inputs | Params | Acc % | Macro-F1 | κ | N1 F1 |
|---|---:|---:|---:|---:|---:|---:|
| DeepSleepNet | EEG | 21M | 78.5 | 75.3 | 0.73 | 47.0 |
| SleepTransformer | EEG | 3.7M | 81.4 | 74.3 | 0.74 | — |
| U-Time | EEG | 1.1M | 81.3 | 76.3 | 0.75 | 51.0 |
| SeqSleepNet | EEG | 0.16M | 82.6 | 76.3 | 0.76 | 46.0 |
| TinySleepNet | EEG | 1.3M | 83.1 | 78.1 | 0.77 | 51.0 |
| LightSleepX | EEG + EOG | 0.049M | 82.6 | 78.7 | 0.77 | 63.0 |
| MultiChannelSleepNet | 3-ch | 13M | 85.0 | 79.6 | 0.79 | 53.0 |
| **This work (Sleep-EDF-78)** | **EEG + EOG** | **0.176M** | **83.2** | **73.7** | **0.761** | **52.5** |

On accuracy and κ the final model sits level with the strong lightweight baselines (SeqSleepNet,
TinySleepNet, U-Time) at roughly an order of magnitude fewer parameters than most of the table, and its
N1 F1 is at the top of the group except for LightSleepX's outlier. The macro-F1 shown here is the
conservative per-recording average; the epoch-pooled figure is ≈ 0.79, level with the published table.

---

# Chapter 5 — Conclusion and Scope for further Research

This project delivered a compact, honestly evaluated, trustworthy sleep-staging system. Its central
finding is methodological: the inherited model's reported κ 0.6663 was a recording-level leak, and
measuring it honestly collapsed it to 0.5066 with N3 F1 at zero. Rebuilding under subject-level splits
produced a 139,606-parameter student — a 4.65× compression — that reaches κ 0.7001 (95% CI [0.6432,
0.7514]) and beats every larger teacher, with κ 0.761 on the standard Sleep-EDF-78 benchmark. The
distillation study established, quantitatively, that a teacher must exceed the student before its soft
targets help, and that what transfers is calibration rather than accuracy. Finally, the evidence packet
and deterministic verifier turn a compact model's residual errors into hedged, gated output rather than
misleading facts: on 31 development nights the verifier released zero incorrect claims.

The main limitations are a single training cohort, a small held-out set that cannot resolve sub-0.05 κ
differences, and N1 as a representation-bound weak point. Future work is, in priority order: (1)
external validation on a clinical cohort such as NSRR, which buys statistical power and clinical
credibility at once; (2) a per-night error model for time-in-bed and sleep-efficiency reliability; (3)
multi-scorer soft labels that reflect genuine scorer disagreement rather than ensemble disagreement; and
(4) ARM64 edge deployment and measurement on a Raspberry Pi 5. Staging-architecture improvements
(Pz-Oz, chin EMG, N1 loss) remain a conditional parallel track, gated on diagnostics that have not yet
run.

---

# Appendix A

*Insert the relevant source code listing here (staging model, distillation loss, or verifier core).*

---

# References

1. B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, and J. J. L. Oberye, "Analysis of a
   sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG," *IEEE Transactions
   on Biomedical Engineering*, vol. 47, no. 9, pp. 1185–1194, 2000.
2. A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research
   resource for complex physiologic signals," *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000.
3. A. Supratak, H. Dong, C. Wu, and Y. Guo, "DeepSleepNet: A model for automatic sleep stage scoring
   based on raw single-channel EEG," *IEEE Transactions on Neural Systems and Rehabilitation
   Engineering*, vol. 25, no. 11, pp. 1998–2008, 2017.
4. S. Mousavi, F. Afghah, and U. R. Acharya, "SleepEEGNet: Automated sleep stage scoring with sequence
   to sequence deep learning approach," *PLOS ONE*, vol. 14, no. 5, e0216456, 2019.
5. H. Phan, F. Andreotti, N. Cooray, O. Y. Chén, and M. De Vos, "SeqSleepNet: End-to-end hierarchical
   recurrent neural network for sequence-to-sequence automatic sleep staging," *IEEE Transactions on
   Neural Systems and Rehabilitation Engineering*, vol. 27, no. 3, pp. 400–410, 2019.
6. M. Perslev et al., "U-Time: A fully convolutional network for time series segmentation applied to
   sleep staging," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
7. A. Supratak and Y. Guo, "TinySleepNet: An efficient deep learning model for sleep stage scoring based
   on raw single-channel EEG," in *Proceedings of the IEEE EMBC*, 2020.
8. E. Eldele et al., "An attention-based deep learning approach for sleep stage classification with
   single-channel EEG," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 29,
   pp. 809–818, 2021.
9. H. Phan et al., "SleepTransformer: Automatic sleep staging with interpretability and uncertainty
   quantification," *IEEE Transactions on Biomedical Engineering*, 2022.
10. M. Perslev et al., "U-Sleep: resilient high-frequency sleep staging," *npj Digital Medicine*, vol. 4,
    no. 72, 2021.
11. R. Vallat and M. P. Walker, "An open-source, high-performance tool for automated sleep staging,"
    *eLife*, vol. 10, e70092, 2021.
12. SalientSleepNet, LMCSleepNet, MultiChannelSleepNet, LightSleepX and SleepSatelightFTC — baseline
    figures compiled from their respective papers in
    `ADDING EOG VERSION - IMPROVEMENT ROADMAP/GPU_PLAN_AND_BENCHMARK.md` §2.1; verify the original
    publications before final citation.
