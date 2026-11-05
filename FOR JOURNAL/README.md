# For Journal: Validity Review and Comparison with Similar Work

**Sleep-Staging · Team 40 · 1 November 2026 · updated 2 November: release v2 is the system (`CURRENT_RELEASE.md`), and its report model now has a test result (P8)**

This is a pre-submission review for the paper. It covers three things:

- **§2:** the results we can report, each traced to a file in this repository.
- **§3:** every methodological problem found, ranked by how much it threatens a claim.
- **§4:** comparison with similar work, with metric values.

The goal is that no reviewer finds something we haven't already found and disclosed.

**Labels on numbers:**
- **[M]** measured by us, from a results file in this repository;
- **[P]** published by the paper's authors (sources in §8);
- **[C]** computed for this review from our raw result files (method stated next to it).

---

## 1. The system in one paragraph

The system is an offline pipeline that turns one night of EEG + EOG into a written sleep report.

1. **Staging.** A 175,599-parameter student network stages each 30-second epoch. It was distilled from
   a 412,311-parameter teacher ensemble.
2. **Evidence packet.** Calibrated probabilities, reliability tiers and a night-confidence tier go into a
   structured packet.
3. **Report.** A fine-tuned Llama-3.2-3B, run locally under a GBNF grammar, writes claims. Every claim
   must cite a packet field.
4. **Release gate.** A frozen deterministic verifier (30 violation codes) checks the claims. A report is
   released only if it passes. Otherwise the reader gets a deterministic fallback summary or the night
   goes to human review.
5. **Agent.** An agentic layer runs the steps: quality control, study-period detection, routing,
   checkpoint/resume and an audit trace. It decides workflow only, never clinical content.

---

## 2. Results we can report

### 2.1 Staging

| Evaluation | Protocol | Acc | κ | Macro-F1 | N1 F1 | Source |
|---|---|---:|---:|---:|---:|---|
| **B1, standard benchmark** | Sleep-EDF-78, SC only, 78 subjects / 153 nights, 10-fold subject-wise CV, trained from scratch | 83.5% | 0.775 ± 0.018 | 79.0 | 53.1 | [M] `kaggle/working/b1/`, mean of 10 fold-level scores (± is fold SD) |
| B1, conservative (last epoch, see §3 F1) | same | 83.0% | 0.769 | 78.4 | 51.6 | [C] `fold_*/training_metrics.jsonl` |
| B1, per-recording mean | same | 83.2% | 0.761 | 73.7 | 52.5 | [M] `b1_summary.json` top level |
| 5-fold CV, SC subjects | Project CV, 85 subjects (SC + ST), corrected labels | 83.5% | 0.776 | 79.3 | 54.1 | [M] `COMPARISON_WITH_EXISTING_SYSTEMS.md` §2 |
| Held-out test | 15 subjects / 29 nights, one evaluation | 81.96% | 0.756 | 78.2 | 50.7 | [M] `STAGE3_REPORT.md` |
| Held-out test, EOG off | same model, EOG channel removed | 81.10% | 0.744 | 77.2 | 48.7 | [M] same |
| Raw EDF → system (Path B) | 31 dev nights, device-like crop, estimated study period | 79.5% | 0.719 | 76.9 | 55.1 | [M] `agentic/reports/P2_STATUS.md` |

**Two numbers agree independently.** The 5-fold project CV (83.5%, κ 0.776) and the standard B1 protocol
(83.5%, κ 0.775) used different folds, different subjects and different training starts. They land
within 0.001 κ of each other.

### 2.2 Report generation

| Evaluation | Result | Source |
|---|---|---|
| v1 report model, held-out test (29 nights) | 29/29 at 5/5 mandatory items, 0 violations, numeric fidelity 1.000. 95% lower bound: **≥ 88.1%** [C] | [M] `student/slm_models/SLM_FINDINGS_REPORT.md` |
| v2 report model, dev (31 nights) | 31/31 verified, 31/31 at 5/5, 0 violations, fidelity 527/527. 95% lower bound: **≥ 88.8%** [C] | [M] `ROADMAP R3 - V2 REPORT MODEL/results/dev_summary.json` |
| **v2 system, held-out test (29 nights), the pre-registered final run (P8)** | **28/29 verified, 0 flawed reports released.** The 1 other night was blocked by the verifier and got a fallback summary: on a night with almost no REM sleep, the model invented a REM-latency value. 28/29 at 5/5, fidelity 492/493. 95% lower bound: **≥ 82.2%** [C] | [M] `ROADMAP P8 - V2 TEST EVALUATION/RESULTS.md` |
| Base Llama-3.2-3B, same task | 2/31 | [M] `COMPARISON_WITH_EXISTING_SYSTEMS.md` §5 |
| Path B, end to end, v2 (31 dev nights) | 30 verified + 1 routed to review (line noise), 0 fallbacks, 0 rejected | [M] `P2_STATUS.md` |
| Reproducibility (v2, 2 runs × 31 nights) | 31/31 byte-identical report pages | [M] `ROADMAP V4 - REPRODUCIBILITY/` |

The lower bounds are two-sided 95% Clopper–Pearson: for n/n successes, the bound is 0.025^(1/n).

v1 (29/29) and v2 (28/29) were scored on different packets (N4kd vs N6adv staging), so report them side by side and never as one pooled figure. A one-night difference is not meaningful (Fisher's exact p = 1.0).

### 2.3 Verifier ablation (31 dev nights, definitions pre-registered)

| Report model | Gate on: released with a flaw | Gate off: released with a flaw | Gate + grammar off |
|---|---:|---:|---:|
| **Llama-3.2-3B-Student v2 (release v2, v2 packets)** | **0/31** | **0/31** | **0/31** |
| Llama-3.2-3B-Student v1 (release v1, v1 packets) | 0/31 | 1/31 | 2/31 |
| Llama-3.2-3B base, untrained (v1 packets) | **0/31** (all fall back) | 31/31 | 31/31 |

**Sources:** `ROADMAP R2b - V2 VERIFIER ABLATION/RESULTS.md` (v2, 2 Nov) and
`ROADMAP R2 - VERIFIER ABLATION/RESULTS.md` (v1 and base). The base model was not regenerated on v2
packets; this was pre-registered.

**What the v2 row shows:**
- With the grammar removed, v2's output was **byte-identical** to its grammar-constrained output on all
  31 nights. These were fresh generations, checked to differ from release v2's invocation only by the
  grammar flag. At temperature 0, the grammar never changed a token.
- So on dev the fine-tuned v2 model needed neither the grammar nor the verifier; there was nothing for
  them to stop.
- No arm of any model produced a wrong number. v1's ungated errors were about hedging and completeness.

**The verifier's role, by model:**
- **v2: a safety net.** On held-out test (P8) it blocked the one flawed v2 generation (1 of 29), an
  invented REM-latency value.
- **The untrained model: the decisive control.** Without the gate, 31 of 31 reports carried a flaw.

---

## 3. Fundamental problems and threats to validity

**Severity key:**
- **Must fix:** a reviewer can reject a headline claim on this. Fix it before submission.
- **Must disclose:** the claim survives, but only if the limitation is stated in the paper.
- **Minor:** worth a sentence in the limitations section.

### F1. B1 picks its best epoch on the fold it reports. Must fix

- **What happens:** in every fold, `run_b1_benchmark.py` evaluates the validation fold after each epoch,
  keeps the epoch with the best macro-F1, then reports that same fold's score (lines 250–275).
- **Why it matters:** model selection on the reported data is optimistically biased.
- **How large [C]:** comparing the selected epoch with the final epoch of each fold:

  | | Selected epoch (reported) | Final epoch | Difference |
  |---|---:|---:|---:|
  | κ, mean of 10 folds | 0.775 | 0.769 | 0.006 |
  | Macro-F1, mean of 10 folds | 79.0 | 78.4 | 0.6 |

  The worst fold is fold 3: κ 0.747 selected at epoch 7 vs 0.727 at its final epoch.
- **The final epoch is still not perfectly clean,** because early stopping (patience 25) also used the
  validation fold. It is a conservative estimate, not an unbiased one.
- **Fix options:**
  1. **Best:** re-run B1 with an inner split. Hold out about 10% of each fold's *training* subjects for
     early stopping. Cost: about 9 GPU-hours on Kaggle, same bundle with a split change.
  2. **Cheap:** report a fixed epoch count chosen in advance, and present the last-epoch figures
     (κ 0.769, macro-F1 78.4) as the headline.
- **Context:** published Sleep-EDF papers differ on how they early-stop. We should state ours exactly,
  whatever we choose.

### F2. B1's "pooled" number is a mean of fold scores. Must fix

- **What happens:** we have two aggregations:
  - per-recording mean (κ 0.761, macro-F1 73.7);
  - mean of 10 fold-level pooled scores (κ 0.775, macro-F1 79.0).
- **The usual convention is different again:** one confusion matrix over every epoch from all folds.
- **Why we can't compute it now:** the result files keep no per-epoch predictions or per-class support.
- **Fix:** the 10 checkpoints are saved, so this needs **inference only**, on CPU. Save the per-epoch
  predictions, build one confusion matrix, and report the result. Combine it with F1's fix if B1 is
  re-run anyway.
- **The 5.3-point macro-F1 gap between aggregations** comes from recordings with little or no N3, where
  a per-recording F1 of 0 drags the mean. State the convention next to every number in the paper.

### F3. "Student beats teacher" mixes three effects. Must fix (the wording)

- **What happens:**
  - the ENSmc teacher and the shipped N4kd were trained on the **original, misaligned labels**;
  - the Stage 3 student was trained on **corrected labels**;
  - all three were scored on corrected test labels.
- **So the test gains** (student κ 0.756 vs teacher 0.732; vs N4kd +0.038, CI [+0.019, +0.057]) combine:
  1. EOG;
  2. the architecture and training recipe;
  3. training on correct labels.
- **The same-label comparison exists, and it is the one to cite.** 5-fold CV, corrected labels for every
  arm (`ADDING EOG VERSION - FINAL SUMMARY/README.md` §3.3):

  | Arm | Params | κ | Δκ vs base, 95% CI |
  |---|---:|---:|---|
  | `base` (N4kd architecture) | 139,606 | 0.7584 | — |
  | `full_eeg` (+ TF branch, local context, recipe) | 160,511 | 0.7653 | +0.0069 [+0.003, +0.011] |
  | `full` (+ EOG) | 175,599 | **0.7763** | **+0.0179 [+0.0095, +0.026]** |

- **Fix:** describe the test comparison as "the combined effect of label correction, EOG and training
  recipe". Attribute architecture and EOG gains only through the CV ablation.
- **For "student beats teacher":** either retrain the teacher on corrected labels (about 3 GPU-hours,
  3 models), or drop the claim. Without one of these, a reviewer will see the confound.

### F4. Our labels differ from other papers' labels. Must disclose

- **What happens:** we found a label-alignment bug in a drop-then-trim preprocessing pattern
  (`ADDING EOG VERSION - ST NIGHTS/README.md`). It shifted 5.65% of training labels and 5.17% of test
  labels. All our current numbers, B1 included, use corrected labels rebuilt from the raw hypnograms.
- **Why it matters:** published baselines use their own pipelines. We don't know whether any of them has
  the same bug. So our labels may differ from theirs.
- **What to do:**
  - state it in the methods;
  - release `verify_hypnogram_alignment.py`;
  - report its validation result: it reproduces the stored labels on 197/197 recordings, and its offsets
    match an independent offset scan.
- **The bug is also a contribution in its own right**, worth a short section: it was worth 3 accuracy
  points on validation (76.1% → 79.2%), and 0.48 → 0.65 κ on ST nights.

### F5. "Verified" means consistent with the packet, not clinically correct. Must disclose

- **What the verifier checks:** our own 30 rules. Each claim must cite packet evidence, numbers must match
  exactly, and unreliable values must be hedged.
- **What it does not check:** whether the report is clinically useful or correct as a clinical
  interpretation.
- **No clinician has rated a report** (R6 not run).
- **Fix the wording.** Write "verifier-consistent" or "released without a detected violation". Do not
  write "accurate reports" or "clinically valid". The 5/5 score counts mandatory items present; it is
  not a quality rating.
- **Strongest fix:** a small blinded clinician rating (R6) before submission, even 2–3 raters on 20
  reports.

### F6. Small evaluation sets, reused. Must disclose

- **Dev reuse:** the 31 dev nights were used at least 3 times to develop the study-period estimator
  (`PENDING.md` X2). They are a development acceptance set, not independent evidence.
- **Test reuse:** the test set (29 nights) has 7 logged reads across this model line (P8, the final run, is the 6th; the 7th redrew v2's confusion matrix and reproduces a logged result exactly). The logs are in
  6 `TEST_ACCESS_LOG.md` files (including P8's and `figures/`), and 1 more under B1, which read IDs only. No read was used for selection,
  but the set is no longer pristine.
- **v2 was run on test once (P8, 2 Nov).** The run was pre-registered as the single final evaluation, so the set cannot be reused to evaluate any later model.
- **What to do:** report the lower bounds from §2.2, not only "100%". B1 is the cleanest staging evidence
  we have, because it is subject-wise CV over all 78 SC subjects.

### F7. B1 has one seed per fold. Must disclose

- **What's there:** the fold-to-fold SD is ±0.018 κ [C], but seed variance is unmeasured.
- **Fair context:** Stage 3's three seeds agreed within 0.001 κ on validation, so seed noise is likely
  small for this architecture.

### F8. Comparison tables come from secondary sources. Must fix (the citations)

- **What's wrong:** most published rows in §4.1 were taken from comparison tables in LMCSleepNet and
  LightSleepX, not from the original papers.
- **Known problems with those rows:**
  - the tables disagree on some entries (AttnSleep's size);
  - LightSleepX's N1 F1 of 63.0 is far above every other model;
  - protocols differ: 10- vs 20-fold, SC-20 vs SC-78, trimming rules, channels.
- **Fix:** check every row against its original paper before submission. Add a column for protocol, or
  drop rows whose protocol we can't confirm.

### F9. B1 uses EEG + EOG. Must disclose

- **The issue:** the strongest small baselines (SeqSleepNet, TinySleepNet, SleepSatelightFTC) are
  EEG-only.
- **Fair comparisons:** against EEG-only models, our 5-fold `full_eeg` arm (κ 0.765) or the test EOG-off
  figure (κ 0.744). Against multi-channel models (LightSleepX, LMCSleepNet, SalientSleepNet,
  MultiChannelSleepNet), the main B1 numbers.

### F10. The YASA comparison is in-domain vs out-of-domain. Must disclose

- **What happens:** our model was trained on Sleep-EDF; YASA wasn't. Both were scored on Sleep-EDF dev
  nights, on our estimated spans.
- **How to word it:** "v2 outperforms an off-the-shelf offline baseline on the same nights" (κ 0.701 vs
  0.635). Don't present it as a general superiority claim.

### F11. Path B uses the dataset's lights-off times. Minor

- **What happens:** the device-like crop starts at the lights-off time from `SC-subjects.xls` /
  `ST-subjects.xls`. A real device starts when the user presses start, which is close to lights-off but
  not identical.
- **Related limitations to state alongside it:**
  - study-period end-point p90 is 49 epochs against a bar of 30 (`PENDING.md` X1);
  - 12–16% of nights have time-in-bed more than 30 min off (`SYSTEM_ACCEPTANCE.md` B6, not met).

### F12. Novelty comes from a targeted search, not a systematic review. Minor

Word every novelty claim as "to our knowledge". The searches were on 14 and 24 October 2026
(`RELATED_WORK.md`).

### F13. No external cohort and no edge measurements. Must disclose

- **External cohort:** none yet (B2/R7 need NSRR access).
- **Edge:** Raspberry Pi 5 numbers are estimates (6–9 min per night). Measured in an offline Docker container limited to 4 CPUs / 16 GB (3 Nov): network off, no secrets, 31/31 dev reports verified and byte-identical to the reference PC, median 151 s and 5.2 GB peak per report on an x86 desktop, and staging parity under ARM64 emulation on 8 Sleep Cassette nights (max |Δp| 7.7e-7). Not a Pi measurement: no thermal, power or Pi timing, and no ARM report generation.
- **What to do:** either remove edge claims from the abstract, or present them as design and estimates
  only.

### F14. Research-packet "night" windows include evening naps on some nights. Must disclose

Found 2 Nov (`PLAIN LANGUAGE REPORT/README.md`).
- **How it happens:** preprocessing starts each Path A packet 30 min before the first expert-scored
  sleep anywhere in the ~20 h home recording. On some SC nights that first sleep is an evening nap.
- **How common:** with the v2 model, 29 of 168 dev+train nights (3 of 31 dev; 30 with v1) have the model's first sleep more than
  30 min before the recorded lights-off. On 31 SC nights the expert's first sleep is over an hour
  before lights-off.
- **Example:** SC4172E0 (dev) has a nap with deep sleep at 18:28, then 4.5 h awake, then lights off at
  00:37. Its v2 packet reports WASO 288 min (v1: 286 min) and efficiency of about 63%.
- **What it does not affect:** staging accuracy, or the model-vs-expert metric errors. Both sides use
  the same window.
- **What it does affect:** reading WASO, time in bed, efficiency or sleep-onset latency as clinical
  values, or comparing them with norms, on those nights.
- **Path B is not affected.** It crops at the estimated lights-off.
- **What to do:** state the window definition in Methods. Don't compare Path A clinical metrics with
  norms without the lights-off check.

### What is not a problem

These were checked and hold:
- **B1 has no subject leakage:** 78/78 subjects and 153/153 recordings are each in exactly one fold.
- **B1 trained from scratch:** α = 1.0 and no warm start, confirmed in every checkpoint.
- **Numbers in the B1 write-up match the raw files.**
- **The test set was never used to select or tune.**
- **Every released report's numbers are exact,** because the verifier compares values exactly, not
  approximately. The one inexact generation seen (v2 test, 492/493) was blocked, not released.
- **Determinism:**
  - the date-dependent chat template was found and fixed (release v1.1);
  - repeated runs are byte-identical;
  - `llama-server` does not reproduce the evaluated outputs (20–23/31 vs 30/31), so the system uses
    `llama-completion`. This is worth reporting as a reproducibility finding.

---

## 4. Comparison with similar work

### 4.1 Staging on Sleep-EDF-78 (SC, 153 nights, 78 subjects)

Published rows are [P] and must be re-checked against the originals (F8).

| Model | Inputs | Params | Acc % | Macro-F1 | κ | N1 F1 |
|---|---|---:|---:|---:|---:|---:|
| DeepSleepNet | EEG | 21M | 78.5 | 75.3 | 0.73 | 47.0 |
| SleepEEGNet | EEG | 2.1M | 82.8 | 77.0 | 0.73 | 44.6 |
| SleepTransformer | EEG | 3.7M | 81.4 | 74.3 | 0.74 | — |
| AttnSleep | EEG | 0.5–5.2M | 81.3 | 75.1 | 0.74 | 42.0 |
| U-Time | EEG | 1.1M | 81.3 | 76.3 | 0.75 | 51.0 |
| SeqSleepNet | EEG | 0.16M | 82.6 | 76.3 | 0.76 | 46.0 |
| TinySleepNet | EEG | 1.3M | 83.1 | 78.1 | 0.77 | 51.0 |
| LightSleepX | EEG + EOG | 0.049M | 82.6 | 78.7 | 0.77 | 63.0 (anomalous) |
| LMCSleepNet | 2 EEG + EOG | 1.49M | 84.1 | 77.7 | 0.77 | 48.5 |
| SalientSleepNet | 2 EEG + EOG | 0.9M | 84.1 | 79.5 | — | 54.2 |
| SleepSatelightFTC | EEG | 0.47M | 84.8 | — | 0.787 | — |
| MultiChannelSleepNet | 2 EEG + EOG | 13M | **85.0** | **79.6** | **0.79** | 53.0 |
| **Ours, B1 (selected epoch)** | EEG + EOG | **0.176M** | 83.5 | 79.0 | 0.775 | 53.1 |
| **Ours, B1 (final epoch, conservative)** | EEG + EOG | **0.176M** | 83.0 | 78.4 | 0.769 | 51.6 |

**Where we stand, conservatively:**
- **κ:** level with the 0.77 group (TinySleepNet, LightSleepX, LMCSleepNet). Below SleepSatelightFTC
  (0.787) and MultiChannelSleepNet (0.79).
- **Macro-F1:** above TinySleepNet and LMCSleepNet, level with LightSleepX, below SalientSleepNet and
  MultiChannelSleepNet.
- **Size:** smaller than every model above us except LightSleepX.
- **Safe claim:** "competitive with models 5–74× larger, at 0.18M parameters". **Unsafe claim:**
  "state of the art".

### 4.2 Per-class F1

| Model | Params | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|---:|
| DeepSleepNet [P] | 21M | 91.0 | 47.0 | 81.0 | 69.0 | 79.0 |
| SleepEEGNet [P] | 2.1M | 90.3 | 44.6 | 85.7 | 81.6 | 82.9 |
| TinySleepNet [P] | 1.3M | 92.8 | 51.0 | 85.3 | 81.1 | 80.3 |
| SalientSleepNet [P] | 0.9M | 93.3 | **54.2** | 85.8 | 78.3 | **85.8** |
| LMCSleepNet [P] | 1.49M | **94.2** | 48.5 | 85.8 | 77.6 | 82.7 |
| MultiChannelSleepNet [P] | 13M | 94.0 | 53.0 | **86.9** | **81.8** | 82.6 |
| **Ours, B1 [C]** (mean of fold-pooled F1) | 0.176M | 92.9 | 53.1 | 84.7 | 78.3 | 85.9 |
| Ours, held-out test [M] | 0.176M | 91.4 | 50.7 | 84.3 | 80.9 | 83.7 |

- **Strengths:** N1 and REM, the stages EOG is known to help.
- **Weaknesses:** N2 (about 2 points below the best) and N3 (about 3.5 points below on B1).

### 4.3 Same-night baseline (31 dev nights, identical epochs and labels) [M]

| | YASA 0.8.1 (pretrained, offline) | Ours (v2) | Δ |
|---|---:|---:|---:|
| Accuracy | 0.744 | **0.790** | +0.046 |
| κ | 0.635 | **0.701** | +0.066 |
| Macro-F1 | 0.642 | **0.717** | +0.076 |
| N1 F1 | 0.302 | **0.545** | +0.243 |

U-Sleep was excluded: it is an online service, and the project is offline by design. Caveat: F10.

### 4.4 Cleared clinical products [P]

These are not comparable: different datasets, populations and protocols. Cite them as context only.

| Product | Clearance | Reported staging agreement |
|---|---|---|
| Somnolyzer / Sleepware G3 (Philips) | FDA 2013, AASM-certified | κ 0.78 vs manual (children, 2025) |
| Cerebra Sleep System | FDA K213007 | κ 0.72, 138 studies |
| EnsoSleep (EnsoData) | FDA 2017, AASM pilot-certified | No peer-reviewed κ found |
| Dreem 3S (Beacon) | FDA 2023 | Vendor claim only |
| U-Sleep / YASA (insomnia benchmark, 904 patients) | Research tools | Macro-F1 0.62 / 0.52 |

The products have regulatory clearance and score respiratory events, arousals and leg movements. We do
neither.

### 4.5 LLM-based sleep and EEG reporting [P]

None of these report the same metric as ours. The table compares the design and the kind of check.

| System | What it does | How output is checked | Reported result |
|---|---|---|---|
| ChatGPT-4o on PSG (2025) | Interprets PSG reports | Compared with physicians afterwards | Diagnosis matched 7/10; therapy 22/50 |
| Locally run LLMs on OSA (Seifen et al., 2025) | Gemma2, Llama3, Mistral Nemo on 30 cases | Compared with physicians afterwards | "Needs significant improvement" |
| SleepVLM (2026) | Vision-language model: stage + rationale per epoch | AASM-feature audit + experts, during evaluation | κ 0.767 MASS-SS1, 0.743 external |
| SleepLM (ICML 2026) | Sleep-language foundation model, 10,000+ people | No claim-level check described | Task breadth, not comparable |
| Wearable sleep-guidance SLMs (2026) | LLM reasoning distilled into small models | Human evaluation afterwards | Human ratings |
| Hybrid EEG report system (2024) | 512 EEG reports | Three LLMs by majority vote | LLM-checked |
| NeuraDock (2026, not sleep) | Deterministic EEG engine + LLM with a context pack | Grounded input, deterministic runs | No per-claim verifier |
| **Ours** | Local 3B model writes evidence-cited claims from a packet | **Deterministic verifier before release** | Test, v2: 28/29 verified, 0 flawed released (1 blocked). v1: 29/29. Dev, v2: 0/31 flawed even with grammar and gate removed. Base model: 0/31 gated vs 31/31 ungated |

**What differs:** every other system checks output after generation, by humans or by another LLM. Ours
blocks a report before release with a deterministic check. That is the claim the evidence supports.
Clinical usefulness (F5) remains unshown.

### 4.6 Model size

| Model | Params | fp32 size |
|---|---:|---:|
| LightSleepX | 49K | 0.2 MB |
| Micro SleepNet (phone) | 48K | ~0.1 MB |
| SeqSleepNet | 164K | 0.66 MB |
| **Ours, staging student** | **175,599** | **0.70 MB** |
| Our teacher (ENSmc) | 412,311 | 1.65 MB |
| TinySleepNet | 1.3M | 5.2 MB |
| MultiChannelSleepNet | 13M | 52 MB |
| DeepSleepNet | 21M | 84 MB |
| **Ours, report model** (Llama-3.2-3B, Q4_K_M) | 3B | **2.02 GB on disk**, ~4.7 GB peak RAM |

**State this plainly:** the staging model is tiny, the full system is not. The report model is 99.9% of
the footprint.

---

## 5. Claims the evidence supports (safe to write)

1. A 0.18M-parameter EEG + EOG staging model reaches κ 0.769–0.775 and macro-F1 78.4–79.0 on the standard
   Sleep-EDF-78 protocol. That is competitive with models 5–74× larger. Give the range until F1 and F2
   are fixed.
2. Two independent protocols agree: project 5-fold CV and B1, both κ ≈ 0.775.
3. On the same CV folds and labels, EOG adds Δκ +0.011 to +0.018, with CIs above 0.
4. The model degrades gracefully without EOG: test κ 0.744 vs the EEG-only baseline's 0.718.
5. The fine-tuned v2 report model released no flawed report on 31 dev nights even with the grammar and
   the verifier removed. The deterministic verifier is the safety net: on held-out test it blocked the
   one flawed v2 generation (1 of 29). For an untrained model it is decisive: 31/31 flawed without the
   gate, 0/31 with it (everything falls back).
6. Numeric fidelity is exact on every evaluation except one test night: v2 scored 492/493, the night
   whose report the verifier blocked. No released report contained a wrong number.
7. The pipeline is deterministic: 31/31 byte-identical reruns. One source of nondeterminism, a
   date-dependent chat template, was identified and fixed.
8. A common preprocessing pattern misaligns 5–7% of Sleep-EDF labels. Correcting it changes reported
   accuracy by about 3 points.

## 6. Claims to avoid (or qualify)

| Avoid | Why | Write instead |
|---|---|---|
| "State-of-the-art staging" | Below SleepSatelightFTC and MultiChannelSleepNet | "Competitive at a fraction of the size" |
| "Student outperforms its teacher" (unqualified) | Label confound, F3 | Qualify it, or retrain the teacher on corrected labels |
| "Clinically accurate reports" | No clinician rating, F5 | "Verifier-consistent reports" |
| "100% report accuracy" | n = 29–31, F6; v2 is 28/29 | "v2: 28/29 verified, 0 flawed released (95% lower bound 82%)" |
| "Runs on a Raspberry Pi" | Not measured, F13 | "Designed for; estimated 6–9 min per night" |
| "Outperforms YASA" (general) | In-domain advantage, F10 | "On the same nights, against an off-the-shelf offline baseline" |
| "The verifier prevents v2's errors" (on dev) | v2 made none on dev, even ungated (R2b) | "Fine-tuning made v2 correct on dev; the verifier is a safety net (blocked 1 of 29 on test)" |

## 7. Pre-submission checklist

| # | Action | Fixes | Cost |
|---|---|---|---|
| 1 | Re-run B1 with an inner early-stopping split, and save per-epoch predictions | F1, F2 | ~9 GPU-h (Kaggle) |
| 2 | Or, if no re-run: inference-only pooled confusion matrix from the 10 saved checkpoints, plus last-epoch headline | F2 (F1 partly) | ~1–2 h CPU |
| 3 | Retrain the ENSmc teacher on corrected labels, or drop "beats teacher" | F3 | ~3 GPU-h, or a wording change |
| 4 | Verify every published row against its original paper; add a protocol column | F8 | ~1 day |
| 5 | Small blinded clinician rating of reports | F5 | 3–4 days + raters |
| 6 | Add the Clopper–Pearson bounds, fold SDs and conventions to every table | F6, F7, F2 | ~1 h |
| 7 | Limitations section covering F4, F9–F13 | — | ~2 h |
| 8 | Release the label-audit script with the paper | F4 | — |

---

## 8. Sources

**Our results:**
- `kaggle/working/b1/` and `ROADMAP B1 - BENCHMARK/VERIFICATION.md` (B1)
- `ADDING EOG VERSION - ADVANCED STUDENT/STAGE3_REPORT.md` (test)
- `ADDING EOG VERSION - FINAL SUMMARY/README.md` (CV ablation, teacher)
- `ADDING EOG VERSION - ST NIGHTS/README.md` (label bug)
- `student/slm_models/SLM_FINDINGS_REPORT.md` (v1 report model)
- `ROADMAP R2 - VERIFIER ABLATION/RESULTS.md` (v1 and base model)
- `ROADMAP R2b - V2 VERIFIER ABLATION/RESULTS.md` (v2)
- `ROADMAP P8 - V2 TEST EVALUATION/RESULTS.md` (v2, held-out test)
- `ROADMAP R1 - YASA COMPARISON/RESULTS.md`
- `agentic/reports/P2_STATUS.md`
- `ROADMAP V4 - REPRODUCIBILITY/`
- `COMPARISON WITH EXISTING SYSTEMS/` (tables, related work, acceptance)

**Published work** (from `COMPARISON WITH EXISTING SYSTEMS/`; verify against originals before citing):
- LMCSleepNet: https://pmc.ncbi.nlm.nih.gov/articles/PMC12527093/
- LightSleepX: https://arxiv.org/html/2609.19062
- SleepSatelightFTC: https://www.biorxiv.org/content/10.1101/2024.08.02.606301v2.full
- Micro SleepNet: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10416229/
- SleepTransformer: https://arxiv.org/abs/2105.11043
- SleepVLM: https://arxiv.org/abs/2603.26738
- SleepLM: https://arxiv.org/abs/2602.23605
- Wearable sleep-guidance SLMs: https://arxiv.org/abs/2609.22463
- ChatGPT-4o on PSG: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11890353/
- Locally run LLMs for OSA: https://pmc.ncbi.nlm.nih.gov/articles/PMC12257169/
- NeuraDock: https://arxiv.org/abs/2606.26519
- Hybrid EEG report system: https://arxiv.org/abs/2411.09874
- Insomnia staging benchmark (U-Sleep, YASA): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12592834/
- Somnolyzer in children: https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1617530/pdf
- Cerebra K213007: https://fda.innolitics.com/device/K213007
