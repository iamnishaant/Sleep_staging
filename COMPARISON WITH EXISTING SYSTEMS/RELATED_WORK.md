# Related Work: Closest Systems and Fair Comparisons

**Sleep-Staging · Team 40 · 24 October 2026**

Published research and products closest to our system, how they compare, and where the research gaps
are. It comes from targeted web searches on 24 October 2026, **not a systematic review**. Where this
document says something wasn't found, that means in these searches.

- **What we plan to do about the gaps:** [ROADMAP.md](ROADMAP.md).
- **Visual version:** the "Sleep-Staging Related Work" page.
- **Staging benchmark tables:** [COMPARISON_WITH_EXISTING_SYSTEMS.md](COMPARISON_WITH_EXISTING_SYSTEMS.md).

**Comparison rules.** A number is set against another only when the dataset, cohort and protocol
match. Otherwise the two are listed side by side and marked "not comparable".

---

## 1. Closest work

### 1.1 Personalised sleep guidance from wearable data (arXiv 2609.22463, Sep 2026)

**What it does.** A multi-agent LLM pipeline writes guidance from wearable sleep records. That reasoning
is distilled into small language models by supervised fine-tuning, with Best-of-N selection at
inference. Quality is judged by human evaluation.

**Relation to ours.** The training idea is the same as ours: a large model's outputs become fine-tuning
targets for a small one.

| | Them | Us |
|---|---|---|
| Check on generated text | Human evaluation after generation | Deterministic verifier before release |
| Generation | Best-of-N | One generation per night, temperature 0 |
| Input / output | Wearable records → advice | Clinical EEG (+ EOG) → factual report with hedged uncertainty |
| Human evaluation | Yes | Not yet |

### 1.2 SleepVLM (arXiv 2603.26738, Mar–Aug 2026)

**What it does.** A vision-language model reads rendered PSG. For each epoch it gives a stage, the AASM
rules it applied and a rationale. The reasoning is checked by an automated AASM-feature audit and by
experts. Reported κ: 0.767 on MASS-SS1, 0.743 on an external cohort.

**Relation to ours.**
- **Level of explanation:** it explains epochs; we report the night.
- **Kind of check:** its audit is part of the evaluation; our verifier is a gate before release.
- **External cohort:** it has one; we don't.
- **κ values are not comparable:** MASS vs Sleep-EDF, with different protocols.

### 1.3 SleepLM (ICML 2026 spotlight)

A sleep-language foundation model trained on 100K+ hours of PSG from 10,000+ people. It does captioning,
retrieval, event localisation and insight generation. No claim-level check against the data is
described. It is far larger in scale and task breadth than ours, and weaker on verifiability.

### 1.4 Locally run LLMs on PSG reports (Seifen et al., Nature and Science of Sleep, 2025)

- **What they tested:** Gemma2, Llama3 and Mistral Nemo, run locally, interpreting 30 real OSA cases.
- **Their conclusion:** performance "needs significant improvement" before real-world use.
- **The sister study:** ChatGPT-4o matched physicians' diagnosis in 7/10 cases and therapy in 22/50.

**Relation to ours.**
- **It's the motivation for our design.** Untrained local models are unreliable, and our own base models
  scored 0/31.
- **Their task is different:** apnea diagnosis and therapy. Ours is sleep architecture, so the two pass
  rates are not comparable.

### 1.5 Boundary-aware EEG agent, NeuraDock (arXiv 2606.26519, Jun 2026, not sleep)

A deterministic local EEG engine produces the numbers. The LLM sees only a summary and a versioned
"context pack" of hardware and scientific limits, and repeated runs give identical outputs.

**Relation to ours.** It's the closest design philosophy. It grounds the model's context but describes no
per-claim verifier or render gate.

### 1.6 Other EEG agents (not sleep)

- **EEGAgent** (AAAI 2026) and **BrainAgent** (arXiv 2606.25400): workflow agents.
- **CogEEGAgent** (arXiv 2607.25045): checks inferential claims against successful statistical results.
- **A hybrid EEG report system** (arXiv 2411.09874): checks 512 reports with three LLMs by majority vote,
  so its checker is itself an LLM.

### 1.7 Staging tools and foundation models

- **U-Sleep and YASA:** open and pretrained. In a 904-patient insomnia benchmark, macro-F1 was 0.62
  (U-Sleep) and 0.52 (YASA).
- **SleepFM** (Nature Medicine, Jan 2026): about 585,000 hours of PSG from about 65,000 people; staging
  comparable to U-Sleep and YASA, plus 130 disease predictions.
- **SleepTransformer** (IEEE TBME 2022): entropy-based uncertainty for deferring epochs to an expert.
- **Distillation:** LEEGNet (0.9M), SleepKD (IJCAI 2023) and Micro SleepNet (48K, phone).

### 1.8 Cleared products

| Product | What it is | Clearance | Reported agreement |
|---|---|---|---|
| EnsoSleep (EnsoData) | AI scoring of PSG and home tests | FDA 2017, expanded; AASM pilot-certified | No peer-reviewed κ found |
| Somnolyzer / Sleepware G3 (Philips) | Lab PSG autoscoring | FDA 2013; first AASM-certified | κ 0.78 vs manual, children (2025) |
| SOMNUM v3.0 (HoneyNaps) | Staging, arousal, respiratory, legs | FDA K253390 | Not found |
| Cerebra Sleep System | Home/facility test with autoscoring | FDA K213007; technologist review required | Staging κ 0.72, 138 studies |
| Dreem 3S (Beacon) | Home dry-EEG headband | FDA 2023 | Vendor claim: as good as individual experts |
| Sleep Profiler, Somfit, Zmachine | Home EEG devices | FDA (various) | Not found |

Products check their output with a technologist's review before a physician signs. Several also score
respiratory events, arousals and leg movements.

---

## 2. Scorecard

Ours are measured results from this repository. "Not comparable" marks rows where the protocols differ.

| Aspect | Best existing (who, figure) | Ours | Verdict |
|---|---|---|---|
| Automatic check of generated text | Human review; LLM-as-judge | Deterministic gate: 0 violations, numeric fidelity 1.000 (v1, historical test) | **Ahead** |
| Complete, correct report from a local model | Local LLMs: "needs significant improvement" (different task) | 29/29 at 5/5 (v1, historical test) | **Ahead**, different task |
| Uncertainty in the report | Reported, not enforced in text | Mandatory hedging with validation error | **Ahead** |
| Offline and private | Local LLM study | Fully offline on CPU | **Ahead** |
| Student vs teacher | KD students usually approach the teacher | Student beats teacher: κ 0.756 vs 0.732 (historical test) | **Ahead** |
| Size for the accuracy | LightSleepX 49K, κ 0.77 (EDF-78) | 176K, κ 0.776 (our CV, SC) | **Level**, protocols close but not identical |
| N1 F1 | SalientSleepNet 54.2 | 54.1 (our CV, SC) | **Level**, same caveat |
| REM F1 | SalientSleepNet 85.8 | 85.8 (our CV, SC) | **Level**, same caveat |
| Staging accuracy | MultiChannelSleepNet 85.0 %, κ 0.79, 13M | 83.5 %, κ 0.776, 0.176M (our CV, SC) | **Behind** ~1.5 pts; B1 protocol needed |
| N3 F1 | MultiChannelSleepNet 81.8 | 79.0 | **Behind** |
| External validation | SleepVLM; SleepFM (SHHS) | None | **Behind** |
| Training data | SleepFM ~65,000 people | 85 subjects | **Behind** by design |
| Clinician evaluation | Seifen et al.; guidance-SLM human eval | None | **Behind** |
| Clinical scope | Products: apnea, arousals, legs | Staging only | **Behind** |
| Regulatory clearance | FDA 510(k), AASM | None | **Behind**, out of scope |
| Edge hardware | Dreem 3S; Micro SleepNet on phone | x86; Pi 5 planned | **Behind**, planned |

**Summary:** ahead on 5, level on 3, behind on 8.
- **Ahead:** everything about trustworthy generated reports.
- **Level:** model size and the hard stages.
- **Behind:** the evidence a clinic expects.

---

## 3. Research gaps our work addresses

1. **A release gate on generated sleep text.** We found no sleep system where each claim must cite
   packet evidence and pass a deterministic verifier before release.
2. **Uncertainty carried into the wording.** Reliability tiers decide what may be stated plainly.
3. **Offline and small end to end,** where the local-LLM literature reports poor reliability.

### What would strengthen these claims

| Evidence | Roadmap item |
|---|---|
| The violation rate with and without the gate | R2 |
| An exploratory clinician study | R6 |
| An external cohort | R7 |
| Measured edge performance | R9 |

---

## Sources

- Personalised sleep guidance SLMs: https://arxiv.org/abs/2609.22463
- SleepVLM: https://arxiv.org/abs/2603.26738
- SleepLM: https://arxiv.org/abs/2602.23605
- Locally run LLMs for OSA: https://pmc.ncbi.nlm.nih.gov/articles/PMC12257169/
- ChatGPT-4o on PSG: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11890353/
- NeuraDock EEG agent: https://arxiv.org/abs/2606.26519
- EEGAgent: https://arxiv.org/abs/2511.09947
- CogEEGAgent: https://www.alphaxiv.org/abs/2607.25045
- BrainAgent: https://arxiv.org/abs/2606.25400
- Hybrid EEG report system: https://arxiv.org/abs/2411.09874
- SleepFM: https://pmc.ncbi.nlm.nih.gov/articles/PMC12920147/
- Insomnia staging benchmark: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12592834/
- SleepTransformer: https://arxiv.org/abs/2105.11043
- Products:
  - Cerebra K213007: https://fda.innolitics.com/device/K213007
  - Somnolyzer in children: https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1617530/pdf
  - SOMNUM v3.0: https://aasm.org/fda-clears-honeynaps-somnum-v3-0-sleep-analysis-software/
  - Dreem 3S: https://www.biospace.com/beacon-biosignals-receives-fda-clearance-for-ai-assisted-sleep-monitoring-device-dreem-3s
  - AASM certification: https://aasm.org/about/industry-programs/autoscoring-certification/
- Staging tables: see `COMPARISON_WITH_EXISTING_SYSTEMS.md` and its sources
