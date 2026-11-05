> **Historical document — banner added 2 Nov 2026.** This describes release v1 (staging model `student_N4kd`, EEG only) or an earlier state of the project. The current release is **v2** (`student_N6adv_full_s1`, EEG + EOG, with the v2 report model). Current numbers and status are in [`CURRENT_RELEASE.md`](../CURRENT_RELEASE.md). Nothing below this banner was changed.

# Comparison with existing systems

**14 October 2026.** This compares three things:

1. **Existing systems:** published sleep-staging models, open/clinical staging tools, and published
   LLM-based PSG interpretation.
2. **Our system today:** a staging student plus a verified local report pipeline. There are two staging
   versions: N4kd (shipped) and the Stage 3 EEG + EOG student.
3. **Our proposed system:** today's system plus an agentic layer and Raspberry Pi 5 edge simulation
   (`AGENTIC AND SIMULATION/`).

Labels on every number:
- **[M]** measured by us, from a results file in this repository;
- **[P]** published by the paper's authors (sources at the end);
- **[E]** an estimate or target, not yet measured.

> **Read this first.** The published numbers use their own protocols (mostly 10- or 20-fold CV on
> Sleep-EDF-78, SC only). Ours use a 5-fold, 85-subject CV over SC + ST with corrected labels, plus a fixed
> held-out test set. The comparison shows *where we stand*, not a ranking. A like-for-like number needs one
> run of the standard protocol (`ADDING EOG VERSION - IMPROVEMENT ROADMAP/GPU_PLAN_AND_BENCHMARK.md`, B1).

---

## 1. The three systems at a glance

| Capability | Existing systems | Our system today | Our proposed system |
|---|---|---|---|
| **Sleep staging** | Many deep models. Sleep-EDF-78: 78.5–85.0% accuracy, κ 0.73–0.79 [P] | EEG + EOG student, 175,599 parameters. CV (SC): 83.5%, κ 0.776. Test: 82.0%, κ 0.756 [M] | Same model [M]; EEG-only routing when EOG is missing (κ 0.744 on test with EOG off [M]) |
| **Model size (staging)** | 0.049M–21M parameters [P] | **0.176M** (0.70 MB fp32) [M] | Same; ONNX export [E] |
| **Teacher → student** | KD is rare in the published staging models compared here | 412K-parameter ensemble → 176K student that **beats its teacher** (κ 0.756 vs 0.732 on test) [M] | Same |
| **Confidence** | Most papers report accuracy only; some report uncertainty (SleepTransformer) [P] | Calibrated probabilities (T = 1.002), per-stage reliability tiers, night-confidence tier, N1 flag [M] | Plus signal-quality checks before staging [E] |
| **Written report** | Studies of general LLMs reading PSG reports (ChatGPT-4o; locally run LLMs for OSA), judged not ready for clinical use [P]; research agent tools that summarise sleep architecture [P] | Local 3B model, grammar-constrained, writing **evidence-cited claims only** [M] | Same, plus fallback summary and routing [E] |
| **Verification of the report** | No deterministic verification gate found in the work we searched | **Frozen two-layer verifier** (30 violation codes) as a hard gate; numeric fidelity 1.000 [M] | Same gate, plus agent-level invariants (I1–I8) [E] |
| **Report quality** | — | Shipped: test 29/29 at 5/5, 0 violations [M]. With the Stage 3 staging model: test 27/29, 1 blocked [M] | Target ≥ 29/29 after the report model is retrained [E] |
| **Runs offline** | Mostly cloud LLMs or GPU research code | Yes, x86 CPU [M] | Yes, Raspberry Pi 5 [E] |
| **Edge device** | Phones and microcontrollers, staging only (Micro SleepNet: 48K parameters, 2.8 ms/epoch on Android) [P] | x86 host only | Pi 5 16 GB: staging + report ≈ 6–9 min per night [E] |
| **Autonomy / workflow** | Human-run tools | Script pipeline (`pipeline.py`) | **Agent:** QC → study period → routing → staging → report → fallback → audit, with 15 fault scenarios [E] |
| **Audit and safety** | Rarely described | Hash-pinned components, audit log, human-review routing [M] | Plus a per-step decision trace with reason codes, checkpoint/resume, and a test-data guard [E] |

---

## 2. Staging metrics against published models (Sleep-EDF-78)

| Model | Inputs | Parameters | Accuracy % | Macro-F1 | κ | Source |
|---|---|---:|---:|---:|---:|---|
| DeepSleepNet | EEG | 21M | 78.5 | 75.3 | 0.73 | [P] LMCSleepNet T5 |
| SleepEEGNet | EEG | 2.1M | 82.8 | 77.0 | 0.73 | [P] LMCSleepNet T5 |
| SleepTransformer | EEG | 3.7M | 81.4 | 74.3 | 0.74 | [P] LightSleepX T2 |
| U-Time | EEG | 1.1M | 81.3 | 76.3 | 0.75 | [P] LightSleepX T2 |
| SeqSleepNet | EEG | 0.164M | 82.6 | 76.3 | 0.76 | [P] LightSleepX T2 |
| TinySleepNet | EEG | 1.3M | 83.1 | 78.1 | 0.77 | [P] |
| LightSleepX | EEG + EOG | 0.049M | 82.6 | 78.7 | 0.77 | [P] |
| LMCSleepNet | 2 EEG + EOG | 1.49M | 84.1 | 77.7 | 0.77 | [P] |
| SalientSleepNet | 2 EEG + EOG | 0.9M | 84.1 | 79.5 | — | [P] |
| SleepSatelightFTC | EEG | 0.47M | 84.8 | — | 0.787 | [P] |
| MultiChannelSleepNet | 2 EEG + EOG | 13M | **85.0** | **79.6** | **0.79** | [P] |
| **Ours: shipped N4kd architecture** (CV, SC subjects) | EEG | 0.140M | 82.0 | 77.6 | 0.756 | [M] |
| **Ours: EEG + EOG student** (CV, SC subjects) | EEG + EOG | **0.176M** | **83.5** | **79.3** | **0.776** | [M] |
| Ours: EEG + EOG student (held-out test, SC subjects) | EEG + EOG | 0.176M | 82.1 | 77.3 | 0.754 | [M] |

**Where we stand:**
- **Upper-middle of the table** on accuracy and κ, and **second-best on macro-F1** (79.3 vs 79.6).
- **Among the smallest models:** only SeqSleepNet and LightSleepX are smaller.
- **The best model** (MultiChannelSleepNet) is +1.5 accuracy and +0.014 κ with **74× more parameters** and an
  extra EEG channel.

## 3. Per-class F1 (Sleep-EDF-78)

| Model | Parameters | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|---:|
| DeepSleepNet [P] | 21M | 91.0 | 47.0 | 81.0 | 69.0 | 79.0 |
| SleepEEGNet [P] | 2.1M | 90.3 | 44.6 | 85.7 | 81.6 | 82.9 |
| TinySleepNet [P] | 1.3M | 92.8 | 51.0 | 85.3 | 81.1 | 80.3 |
| SalientSleepNet [P] | 0.9M | 93.3 | **54.2** | 85.8 | 78.3 | **85.8** |
| LMCSleepNet [P] | 1.49M | **94.2** | 48.5 | 85.8 | 77.6 | 82.7 |
| MultiChannelSleepNet [P] | 13M | 94.0 | 53.0 | **86.9** | **81.8** | 82.6 |
| **Ours, EEG + EOG student, CV SC [M]** | **0.176M** | 93.1 | **54.1** | 84.3 | 79.0 | **85.8** |
| Ours, held-out test, all 29 nights [M] | 0.176M | 91.4 | 50.7 | 84.3 | 80.9 | 83.7 |
| Ours, shipped N4kd, held-out test [M] | 0.140M | 90.7 | 45.5 | 81.9 | 79.9 | 78.2 |

**Reading by stage:**
- **N1 and REM:** joint best, the stages EOG is known to help.
- **W:** in the pack.
- **N2:** about 1–2.5 points below the best.
- **N3:** about 2–3 points below the best. These two are our relative weaknesses.

### Per-class precision and recall, ours on the held-out test (corrected labels) [M]

| Stage | N4kd P / R / F1 | EEG + EOG student P / R / F1 | Δ F1 |
|---|---|---|---:|
| W | 96.8 / 85.3 / 90.7 | 95.9 / 87.2 / 91.4 | +0.7 |
| N1 | 40.8 / 51.6 / 45.5 | 47.6 / 54.3 / 50.7 | **+5.3** |
| N2 | 81.1 / 82.7 / 81.9 | 82.7 / 86.1 / 84.3 | +2.4 |
| N3 | 78.3 / 81.5 / 79.9 | 79.4 / 82.4 / 80.9 | +1.0 |
| REM | 79.9 / 76.5 / 78.2 | 86.2 / 81.3 / 83.7 | **+5.3** |
| **Overall** | acc 79.00%, κ 0.717, MF1 75.2 | acc **81.96%**, κ **0.756**, MF1 **78.2** | |

## 4. Model size

| Model | Parameters | fp32 size | Relative to ours |
|---|---:|---:|---:|
| LightSleepX | 49K | 0.2 MB | 0.28× |
| Micro SleepNet (phone) | 48K | ~0.1 MB on device | 0.27× |
| SeqSleepNet | 164K | 0.66 MB | 0.93× |
| **Ours, shipped N4kd** | **139,606** | **0.56 MB** | 0.80× |
| **Ours, EEG + EOG student** | **175,599** | **0.70 MB** | **1×** |
| **Our teacher (ENSmc)** | 412,311 | 1.65 MB | 2.3× |
| SleepSatelightFTC | 0.47M | 1.9 MB | 2.7× |
| SalientSleepNet | 0.9M | 3.6 MB | 5.1× |
| TinySleepNet | 1.3M | 5.2 MB | 7.4× |
| LMCSleepNet | 1.49M | 6.0 MB | 8.5× |
| SleepTransformer | 3.7M | 14.8 MB | 21× |
| MultiChannelSleepNet | 13M | 52 MB | 74× |
| DeepSleepNet | 21M | 84 MB | 120× |

**The whole system isn't small.** The report model (Llama-3.2-3B-Student, Q4_K_M) is **2.0 GB** on
disk and peaks at about **4.7 GB** of RAM on x86 [M]. On an edge device, the staging model is negligible
and the report model is the cost.

**Teacher → student.** Our student has 43% of the teacher's parameters, and it is better on every
test metric [M]:

| | Accuracy | κ | Macro-F1 |
|---|---:|---:|---:|
| Student | 81.96% | 0.756 | 78.2 |
| Teacher | 80.02% | 0.732 | 76.9 |

The usual KD outcome is a student that comes close to its teacher; this one exceeds it. That's because
it adds EOG and the training recipe on top of distillation.

---

## 5. Report generation compared

| | Existing (published) | Ours today [M] | Proposed |
|---|---|---|---|
| Model | General LLMs: ChatGPT-4o (cloud), locally run open LLMs [P] | Llama-3.2-3B fine-tuned on 123 oracle examples, Q4_K_M, local | Same; retrained on the new packets [E] |
| Input | PSG report PDFs / text [P] | A structured evidence packet: metrics, reliability, confidence | Plus QC and routing facts [E] |
| Output control | Free text | GBNF grammar: claims only, each citing packet evidence | Same |
| Checking | Comparison with physicians after the fact [P] | **Every report verified before release:** 30 codes, numeric fidelity checked exactly | Same, plus a fallback summary built only from packet values [E] |
| Reported quality | Locally run LLMs on OSA: "needs significant improvement prior to real-world implementation" [P] | Test 29/29 at 5/5 mandatory, 0 violations, fidelity 1.000 (shipped). Dev 30/31 on the new packets. Base Llama-3.2-3B: 2/31 | ≥ 29/29 target [E] |
| Speed | — | ~70–115 s per report (x86 CPU) | Pi 5: ~5–8 min per report [E] |

---

## 6. What the proposed agentic + simulation layer adds

| | Today [M] | Proposed [E] |
|---|---|---|
| Missing-EOG handling | Manual | Agent routes to the EOG-off mode (measured: κ 0.744, still above N4kd) |
| Unusable signal | Goes straight to the model | QC step (flat/stuck/clipping/line noise): REVIEW_ONLY with a reason code |
| Unlabelled night | The study period comes from the hypnogram, or the full recording | Training-rule study-period detection (two-pass) |
| Sampling rate / montage | Not checked | Resample to 100 Hz; montage policy |
| LLM failure | Human-review file | Restart once → (optional repair) → fallback summary → review |
| LLM runtime | `llama-server` path: 20–23/31 at 5/5 on dev, vs 30/31 on the evaluated path [M] | The evaluated `llama-completion` path |
| Device | x86 PC | Raspberry Pi 5 16 GB: ~6–8 GB peak, ~6–9 min per night, active cooling, NVMe |
| Validation | Dev / test sets | Simulation ladder: Docker x86 → ARM64 emulation → overnight replay with 15 fault scenarios → real Pi |
| Audit | Per-report entries | Per-step trace with reason codes, policy hash, resume after power loss |

---

## 7. Is there novelty?

**Not novel on their own.** These are established in the literature, and we shouldn't claim them as new:
- lightweight CNN + Transformer staging;
- multi-modal EEG + EOG fusion;
- knowledge distillation;
- the multi-scale CNN front end;
- grammar-constrained decoding;
- running LLMs locally.

**Plausibly novel, as far as our searches found.** These come from targeted web searches, **not a
systematic review**, so word any paper claim as "to our knowledge":

1. **A verified, evidence-cited sleep-report pipeline with a small local LLM.** The published LLM work
   on PSG uses general LLMs on report documents and evaluates afterwards. We found no system where every
   generated claim must cite a packet field and pass a frozen deterministic verifier *before* release,
   with exact numeric fidelity (1.000) and human-review routing.
2. **Uncertainty carried end to end.** Calibration, per-stage reliability tiers, the night-confidence
   tier and the N1 flag go *into* the report. The verifier enforces that unreliable values are
   hedged (`safe_item_hedged` / `unsafe_item_not_hedged` codes).
3. **A sub-200K student that beats its 412K teacher**, through gated EEG + EOG fusion that starts at the
   teacher-distilled function. It degrades gracefully without EOG (κ 0.744, still above the shipped model).
   This is near the top of published Sleep-EDF-78 per-class N1 and REM results at a fraction of the size.
4. **A data-quality finding.** A label-alignment bug in a common drop-then-trim preprocessing pattern:
   5–7% of labels shifted, worth +1.3 to +3 points by itself. It is a cautionary result for anyone using
   similar pipelines (we didn't check other public codebases).
5. **Proposed:** an agentic, auditable, offline pipeline with edge simulation. The agent decides
   workflow, never clinical content. The design has explicit invariants and fault-injection tests. We
   found edge *staging* work (phones, microcontrollers), but not end-to-end staging + verified reporting
   on a single-board computer. **It isn't built yet**, so it isn't a result yet.

**What would strengthen the novelty claim:**
- run the standard Sleep-EDF-78 protocol once (B1);
- evaluate on one external dataset;
- complete the Pi 5 measurements;
- compare the verifier against an unverified baseline (e.g. the same LLM without grammar or verifier) on
  violation rates. The base Llama-3.2-3B scores 2/31 here, which is already a strong data point.

---

## 8. Caveats

- **Published figures come from secondary tables** (LMCSleepNet, LightSleepX), which sometimes disagree
  (e.g. AttnSleep's size). Cite the original papers in any publication.
- **Our test set** has 15 subjects (3 ST) and has now been used for this model line.
- **Everything marked [E]** is planned or estimated.
- **The novelty assessment** comes from a limited literature search on 14 October 2026.

## Sources

- LMCSleepNet (Tables 4–5: per-class F1, parameters, channels): https://pmc.ncbi.nlm.nih.gov/articles/PMC12527093/
- LightSleepX (Tables 1–2: Sleep-EDF-20/78 comparison, parameter counts): https://arxiv.org/html/2609.19062
- SleepSatelightFTC: https://www.biorxiv.org/content/10.1101/2024.08.02.606301v2.full
- L-SeqSleepNet: https://ar5iv.labs.arxiv.org/html/2301.03441
- Micro SleepNet (mobile, 48K parameters, 2.8 ms/epoch): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10416229/
- ChatGPT-4o on PSG interpretation: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11890353/
- Locally run LLMs for OSA (real-world PSG): https://pmc.ncbi.nlm.nih.gov/articles/PMC12257169/
- BrainAgent (multi-agent brain-signal framework with a sleep-report tool): https://arxiv.org/pdf/2606.25400
- SleepLM: https://www.emergentmind.com/papers/2602.23605
- Benchmarking automated staging in insomnia (YASA, U-Sleep and others): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12592834/
- Our own results: `ADDING EOG VERSION - FINAL SUMMARY/README.md` and `numbers.json`;
  `ADDING EOG VERSION - PIPELINE INTEGRATION/INTEGRATION_REPORT.md`; `AGENTIC AND SIMULATION/`
