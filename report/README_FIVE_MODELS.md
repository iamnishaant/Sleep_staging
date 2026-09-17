# The five-model comparison: why we ran it, what it found, and what was chosen

**Sleep-EDF sleep-staging report pipeline · Team 40 · Project 48**
**Written 15 September 2026.** It covers the candidate runs of 12–14 September
2026 on branch `team-40`.

This README is the full record of the comparison between five small local
language models. It covers:

- why the comparison was run, and how the five were picked;
- how they were compared;
- every comparison table, with its values and what each shows;
- which model was chosen, and why;
- what the comparison means for the project.

Every value comes from the committed results: `candidates/results/*/summary.json`,
`candidates/results/comparison.txt`, and `deploy/results/`.

Related documents: `report/README_SLM.md` (the wider SLM work),
`report/README_DISTILLATION.md` (what the chosen model would be trained for),
and `report/PHASE2_NOTES.md` (the lab record).

---

## 1. The answer in one paragraph

**None of the five models meets the report contract on any of the 31 dev
nights.** Not one of the 155 generations both passes verification and states
all five mandatory facts.

- **Numbers are not the problem.** All the models copy numbers exactly (four
  of five at numeric fidelity 1.000).
- **Structure is.** They fail on which claim type goes with which evidence
  item, and on which evidence a key must cite. Each fails in a different way.
- **Size did not fix it.** It did not fix it between 1.5B and 3.8B.

**The recommended model is Qwen2.5-1.5B,** as the student to train if the
project's pre-registered rule routes to distillation. Llama-3.2-3B would be
trained alongside it as the capacity comparison. The reasons are in section 7.

---

## 2. Why compare five models

**The question.** Phase 2 asks whether a language model small enough to run
on the device can fill the verified report contract. The device is a
Raspberry Pi 5-class target, running overnight batch. The first model tried,
Qwen2.5-1.5B, reached 0 of 31 nights at 5/5 and never used `hedged_value`.

**One model cannot answer the question.** Qwen's failure could mean either of
two things:

1. **A size limit.** A 1.5B model is too small, and a slightly larger model
   would do it.
2. **A property of the task.** No small model does it when prompted, however
   it is chosen.

These lead to different projects. If (1), the answer is to pick a bigger
small model. If (2), prompting is not enough, and the local model has to be
*trained* (distillation). Five models from 1.5B to 3.8B, from five different
families, under identical settings, separate the two.

**The comparison answers two further questions:**

- **Which model should be trained, if training is needed?** The best student
  is not necessarily the best prompted model. What matters is which strengths
  it already has, and which errors training must fix.
- **Which models are deployable?** Memory and speed on a Pi 5-class device,
  where file size turns out to be misleading.

---

## 3. How the five were picked

**The target range is 1–4B parameters,** small enough for an 8 GB device at
4-bit quantisation.

**One model from each of five families:** Alibaba (Qwen), Hugging Face
(SmolLM2), Google (Gemma), Meta (Llama) and Microsoft (Phi). A result that
held across all five would not be one family's quirk.

**The same format for all:** instruction-tuned, with a chat template, as
Q4_K_M GGUF files for llama.cpp.

**Provenance.** Qwen and SmolLM2 come from their official GGUF repositories.
The other three have no ungated official GGUF, so they come from bartowski's
quantisations. Every file's sha256 matches the value Hugging Face publishes.

| model | repository | file size | sha256 (first 16) |
|---|---|---:|---|
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | 1,066 MiB | `6a1a2eb6d15622bf` |
| SmolLM2-1.7B-Instruct | `HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF` | 1,007 MiB | `decd2598bc2c8ed0` |
| Gemma-2-2b-it | `bartowski/gemma-2-2b-it-GGUF` | 1,629 MiB | `e0aee85060f168f0` |
| Llama-3.2-3B-Instruct | `bartowski/Llama-3.2-3B-Instruct-GGUF` | 1,926 MiB | `6c1a2b4116103267` |
| Phi-3.5-mini-instruct | `bartowski/Phi-3.5-mini-instruct-GGUF` | 2,282 MiB | `e4165e3a71af97f1` |

**Architecture,** read from each file's GGUF header:

| model | parameters | layers | attention heads / KV heads | grouped-query attention | head dim | KV cache per token |
|---|---:|---:|---:|---|---:|---:|
| Qwen2.5-1.5B | 1.78 B | 28 | 12 / 2 | yes | 128 | 28 KiB |
| SmolLM2-1.7B | 1.71 B | 24 | 32 / 32 | **no** | 64 | 192 KiB |
| Gemma-2-2b | 2.61 B | 26 | 8 / 4 | yes | 256 | 104 KiB |
| Llama-3.2-3B | 3.21 B | 28 | 24 / 8 | yes | 128 | 112 KiB |
| Phi-3.5-mini | 3.82 B | 32 | 32 / 32 | **no** | 96 | 384 KiB |

**Why this table matters.** Grouped-query attention sets how much memory each
generated token costs. The two models without it cache 6.9× (SmolLM2) and
13.7× (Phi) more per token than Qwen. On a memory-bound device, this matters
more than parameter count or file size (section 5.9).

---

## 4. The rules of the comparison

A comparison is only as fair as its settings, so every model ran under
exactly the same ones:

| setting | value | why |
|---|---|---|
| prompt | P1: the frozen `build_prompt` body plus claim-shape rules | chosen from one-night diagnostics; examples were excluded, because they steered which items got reported |
| chat template | each model's own (`--jinja -cnv -st`) | each model sees the prompt format it was trained on |
| Llama's date | pinned to "26 Jul 2024" (`--chat-template-file`) | Llama 3.2's template writes today's date into the prompt; pinning makes the run reproducible |
| grammar | `report/claims.gbnf` | the output is always a legal claim list, so failures are about meaning, not syntax |
| decoding | temperature 0, seed 0 | deterministic: a regenerated night comes out byte-identical |
| limits | 3,000-token cap, context 12,288 | the longest correct answer is about 973 tokens, so hitting the cap means looping |
| protocol | one generation per night, no retries | a retry would be best-of-N by another name |
| nights | the 31 dev nights (11 high, 10 medium, 10 low confidence) | the 29 test nights stay locked until the end |
| scoring | the frozen verifier and evaluator | identical for every model, and for the reference |

**The main metric is the count of nights at 5/5 mandatory,** not a mean. A
report is usable only if it states all five mandatory facts and passes
verification.

**`hedged_value` is reported as two numbers:** nights that use it at least
once, out of 31, and total `hedged_value` claims, out of 434 (14 unsafe items
× 31 nights).

---

## 5. The comparison, table by table

### 5.1 The headline

| model | nights at 5/5 | mean mandatory | schema-valid | passed | rendered | hedged nights | hedged claims | hit the cap | violations | claims |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0/31 | **0.497** | 30/31 | 0 | 0 | 0/31 | 0/434 | 1/31 | 106 | 156 |
| SmolLM2-1.7B | 0/31 | 0.019 | 1/31 | 0 | 0 | 0/31 | 0/434 | 30/31 | 46 | 17 |
| Gemma-2-2b | 0/31 | 0.219 | 31/31 | 1 | 1 | 31/31 | 50/434 | 0/31 | 43 | 143 |
| Llama-3.2-3B | **2/31** | 0.297 | 11/31 | 0 | 0 | 11/31 | **159/434** | 20/31 | 180 | 359 |
| Phi-3.5-mini | 0/31 | 0.368 | 31/31 | **9** | **9** | 20/31 | 102/434 | 0/31 | 95 | 204 |
| *Reference, 8 scored nights (interim)* | *8/8* | *1.000* | *8/8* | *8* | *8* | *8/8* | *112/112* | *0* | *0* | |

**What it shows:**

- **No model reaches the contract.** Nothing passes at 5/5.
- **The "best" model depends on the metric.** Qwen covers the most mandatory
  facts, Llama hedges the most, and Phi produces the most clean reports. Each
  is best at a different part of the task, and none at all of it.
- **Phi's 9 passes are thin.** They all stop at 4 of 5 facts.
- **The reference row is interim.** It covers 8 scored nights, 6 of them
  high-confidence, and should be expected to fall as the harder tiers arrive.

### 5.2 How many mandatory facts each night got

| model | 0 of 5 | 1 of 5 | 2 of 5 | 3 of 5 | 4 of 5 | 5 of 5 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 1 | 0 | 15 | 13 | 2 | 0 |
| SmolLM2-1.7B | 30 | 0 | 0 | 1 | 0 | 0 |
| Gemma-2-2b | 0 | 28 | 3 | 0 | 0 | 0 |
| Llama-3.2-3B | 20 | 0 | 0 | 0 | 9 | 2 |
| Phi-3.5-mini | 5 | 15 | 0 | 2 | 9 | 0 |

**What it shows:** the shape of each distribution is each model's failure
mode.

- **Qwen is consistent but incomplete.** Most nights are at 2–3 of 5.
- **Llama and Phi are bimodal.** Llama either loops (0) or covers nearly
  everything (4–5). Phi either writes its clean high-night template (4) or
  very little (0–1).
- **Gemma is stuck at 1.** Its one fact is usually the confidence tier.

### 5.3 By confidence tier

| model | tier | schema-valid | mean mandatory | nights at 5/5 | discretionary | rendered | hedged (nights · claims) | hit the cap | violations |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| **Qwen2.5-1.5B** | high | 11/11 | 0.473 | 0 | 0.000 | 0 | 0/11 · 0/154 | 0 | 43 |
| | medium | 9/10 | 0.380 | 0 | 0.000 | 0 | 0/10 · 0/140 | 1 | 42 |
| | low | 10/10 | 0.640 | 0 | 0.000 | 0 | 0/10 · 0/140 | 0 | 21 |
| **SmolLM2-1.7B** | high | 1/11 | 0.055 | 0 | 0.000 | 0 | 0/11 · 0/154 | 10 | 26 |
| | medium | 0/10 | 0.000 | 0 | 0.000 | 0 | 0/10 · 0/140 | 10 | 10 |
| | low | 0/10 | 0.000 | 0 | 0.000 | 0 | 0/10 · 0/140 | 10 | 10 |
| **Gemma-2-2b** | high | 11/11 | 0.255 | 0 | 0.117 | 1 | 11/11 · 19/154 | 0 | 17 |
| | medium | 10/10 | 0.200 | 0 | 0.143 | 0 | 10/10 · 21/140 | 0 | 16 |
| | low | 10/10 | 0.200 | 0 | 0.071 | 0 | 10/10 · 10/140 | 0 | 10 |
| **Llama-3.2-3B** | high | 6/11 | 0.436 | 0 | 0.539 | 0 | 6/11 · 86/154 | 5 | 92 |
| | medium | 3/10 | 0.240 | 0 | 0.300 | 0 | 3/10 · 42/140 | 7 | 49 |
| | low | 2/10 | 0.200 | 2 | 0.200 | 0 | 2/10 · 31/140 | 8 | 39 |
| **Phi-3.5-mini** | high | 11/11 | **0.764** | 0 | 0.000 | 9 | 0/11 · 0/154 | 0 | 4 |
| | medium | 10/10 | 0.100 | 0 | 0.300 | 0 | 10/10 · 72/140 | 0 | 61 |
| | low | 10/10 | 0.200 | 0 | 0.000 | 0 | 10/10 · 30/140 | 0 | 30 |

**What it shows: tier effects are behaviours, not the model reading the
tier.**

- **Qwen's higher low-tier mean (0.640)** comes from adding a low-confidence
  flag to almost every night, which is only correct on low nights.
- **Phi's high-tier strength (0.764, 9 reports)** is one fixed template that
  happens to fit high nights. On medium and low nights, the same model hedges
  the facts that are safe to state.
- **Medium and low nights are where every model struggles.** Those are also
  the tiers the reference run has barely sampled so far.

### 5.4 Violations by rule (overall, then high / medium / low)

| rule | Qwen | SmolLM2 | Gemma | Llama | Phi |
|---|---|---|---|---|---|
| key cites the wrong evidence (`text_key_dependency_missing`) | **53** (22/18/13) | – | 20 (3/7/10) | – | 27 (2/25/0) |
| key's claim is false for this night (`text_key_predicate_false`) | 17 (10/7/0) | – | **22** (13/9/0) | 9 (6/3/0) | 8 (2/6/0) |
| both REM latencies in one claim (`rem_latency_double_count`) | 26 (11/7/8) | – | – | – | – |
| unsafe item stated as a plain value (`unsafe_item_not_hedged`) | 9 (0/9/0) | 14 (14/0/0) | 1 (1/0/0) | **145** (78/39/28) | – |
| safe item hedged (`safe_item_hedged`) | – | – | – | 6 (3/0/3) | **60** (0/30/30) |
| cut off, not valid JSON (`malformed_json`) | 1 (0/1/0) | **30** (10/10/10) | – | 20 (5/7/8) | – |
| wrong number (`value_mismatch`) | – | 1 | – | – | – |
| number without evidence (`uncited_quantity`) | – | 1 | – | – | – |
| **total** | **106** | **46** | **43** | **180** | **95** |

**What it shows: each model has one dominant error, and it is different for
each.**

- **Qwen:** keys citing the wrong evidence.
- **SmolLM2:** loops that leave malformed JSON.
- **Gemma:** keys that are false for the night.
- **Llama:** unsafe items restated as plain values.
- **Phi:** safe items hedged.

None of these is an arithmetic error: only one wrong number appears across
all 155 generations. The errors are about *which claim type and key go with
which item*. That is a fixed mapping from packet to claims, which is what
supervised training teaches.

**Low totals are not good scores.** SmolLM2 and Gemma have the fewest
violations because they say the least: SmolLM2's malformed outputs contain
no readable claims, and Gemma writes 4–7 claims a night.

### 5.5 Finding the evidence against getting it right

| model | mandatory facts cited, verified or not | mandatory facts verified | discretionary cited | discretionary verified |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.774 | 0.497 | 0.555 | 0.000 |
| SmolLM2-1.7B | 0.019 | 0.019 | 0.030 | 0.000 |
| Gemma-2-2b | 0.219 | 0.219 | 0.111 | 0.111 |
| Llama-3.2-3B | 0.355 | 0.297 | 0.353 | 0.353 |
| Phi-3.5-mini | 0.800 | 0.368 | 0.159 | 0.097 |

**What it shows:** "cited" counts any claim that points at the item, right or
wrong; "verified" counts claims that pass.

- **The gap between the two is claims that found the evidence but got its
  shape wrong.**
- **Qwen and Phi find 77–80% of the mandatory facts** and get only 37–50%
  right.
- **Qwen finds 55% of the discretionary items but verifies none of them,**
  because it never hedges.

This is the clearest evidence that the models can *read* the packet. What
they lack is the mapping.

### 5.6 Hedging: the most diagnostic number

| model | nights using `hedged_value` | hedged claims (of 434) | what it hedges |
|---|---:|---:|---|
| Qwen2.5-1.5B | 0/31 | 0 | nothing: the form never appears |
| SmolLM2-1.7B | 0/31 | 0 | nothing |
| Gemma-2-2b | 31/31 | 50 | only the two REM-latency items |
| Llama-3.2-3B | 11/31 | 159 | all 14 unsafe items on nights it finishes, but it also restates each as a plain value |
| Phi-3.5-mini | 20/31 (0 high, 10 medium, 10 low) | 102 | the three facts that are *safe* to state, on medium and low nights |
| *Reference (interim)* | *8/8* | *112/112* | *exactly the 14 unsafe items, every night* |

**What it shows:** the rule is to hedge exactly the 14 unsafe items and
nothing else.

- **Three models learned the hedged form.** None applies the rule: each
  hedges the wrong set, or hedges and duplicates.
- **This is the rule a trained student would be shown on every training
  night.**

### 5.7 Looping: generations that ran to the 3,000-token cap

| model | overall | high | medium | low | median time per night |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 1/31 | 0 | 1 | 0 | 14.9 s |
| SmolLM2-1.7B | **30/31** | 10 | 10 | 10 | 204.1 s |
| Gemma-2-2b | 0/31 | 0 | 0 | 0 | 31.7 s |
| Llama-3.2-3B | **20/31** | 5 | 7 | 8 | 275.4 s |
| Phi-3.5-mini | 0/31 | 0 | 0 | 0 | 139.1 s |

**What it shows:**

- **51 of 155 generations looped.** Every one is cut off mid-JSON, so it
  fails as malformed, although every claim inside is one the grammar allows.
- **The grammar never forces the claim list to close.** Whether that causes
  the loops is what the designed 2G grammar ablation tests.
- **Looping also drives cost.** The two looping models take 14–19× longer per
  night than Qwen.

### 5.8 Run log: time and memory on the x86 evaluation host

These figures belong in the run log only: context 12,288, not the deployment
context.

| model | total time | time per night: min / median / max | peak memory | predicted peak |
|---|---:|---|---:|---:|
| Qwen2.5-1.5B | 588 s | 13.2 / 14.9 / 99.7 s | 2,038 MiB | 2,036 MiB |
| SmolLM2-1.7B | 6,708 s | 95.1 / 204.1 / 289.6 s | 4,153 MiB | 4,151 MiB |
| Gemma-2-2b | 957 s | 27.2 / 31.7 / 41.3 s | 3,594 MiB | 3,983 MiB |
| Llama-3.2-3B | 7,796 s | 164.7 / 275.4 / 318.9 s | 4,743 MiB | 4,736 MiB |
| Phi-3.5-mini | 4,699 s | 63.1 / 139.1 / 423.8 s | 8,170 MiB (max) | 8,176 MiB |

**What it shows:**

- **How the prediction is built:** the deployment table's peak at context
  4,096, plus the KV cache for the 8,192 extra positions.
- **It predicts four of the five peaks** to within about 7 MiB, so the memory model
  used for deployment is sound.
- **Gemma is the exception,** about 390 MiB lower. That is consistent with
  its sliding-window layers, whose cache llama.cpp does not grow with
  context; the explanation is unverified.
- **The total run was about 5 h 46 min.**

### 5.9 Deployment on a Raspberry Pi 5 (analytical)

Nothing was run on a Pi.

- **Memory** was measured on the host, at the deployment context of 4,096.
- **Throughput** is predicted under an **assumed** 7–10 GB/s effective
  bandwidth.

| model | file size | peak memory | share of 8 GB | KV cache per token | predicted tokens/s | predicted generation time (floor) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 1,066 MiB | **1,812 MiB** | 22% | **28 KiB** | **6.7–9.5** | **124–177 s** |
| SmolLM2-1.7B | 1,007 MiB | 2,615 MiB | 32% | 192 KiB | 4.6–6.5 | 169–241 s |
| Gemma-2-2b | 1,629 MiB | 3,151 MiB | 38% | 104 KiB | 3.6–5.1 | 227–325 s |
| Llama-3.2-3B | 1,926 MiB | 3,840 MiB | 47% | 112 KiB | 3.1–4.4 | 266–380 s |
| Phi-3.5-mini | 2,282 MiB | 5,103 MiB | 62% | 384 KiB | 2.1–3.0 | 365–522 s |

**What it shows:**

- **All five fit in 8 GB.** At minutes per report, all are plausible for
  overnight batch use and not for interactive use.
- **File size ranks them wrongly.** SmolLM2 has the smallest file, yet
  because it lacks grouped-query attention it peaks 800 MiB above Qwen and
  spends 31% of its decode traffic on the KV cache.
- **Qwen is best on every deployment measure:** memory, cache per token,
  speed and time per report.

### 5.10 Robustness: one irrelevant line moved 28 of 31 outputs

Llama was first run with its template's date unpinned, then re-run with the
date pinned.

| Llama-3.2-3B | unpinned (archived) | pinned (official) |
|---|---:|---:|
| nights at 5/5 | 4 (SC4581G0, SC4582G0, ST7081J0, ST7151J0) | 2 (SC4111E0, ST7081J0) |
| mean mandatory | 0.310 | 0.297 |
| hedged claims | 156/434 | 159/434 |
| hit the cap | 20/31 | 20/31 (five started, five stopped) |
| outputs identical across the two runs | 3 of 31 | |

**What it shows:** a date string with no task content changed 28 of 31
outputs, and only one night is at 5/5 in both runs.

- **At temperature 0 with one generation per night, a single night is a
  point observation,** for every model here. The failure mode held; the
  individual nights did not.
- **The reportable unit is the population figure across 31 nights.**
- **Llama's 2/31 could as easily have been 0 or 4.**

---

## 6. How each model fails

- **Qwen2.5-1.5B states the facts, but never hedges.** It has the highest
  mandatory coverage and finds 77% of mandatory facts, and it almost never
  loops. But it never uses `hedged_value`, attaches the N1 key to the wrong
  evidence, and bundles both REM latencies into one claim.
- **SmolLM2-1.7B lists values until it is cut off.** It writes only plain
  values: it walks the 17 numeric items once, then repeats the five stage
  fractions until the cap, on 30 of 31 nights.
- **Gemma-2-2b is short, and hedges one pair.** It writes 4–7 claims a night,
  is always valid and never loops. It hedges only the two REM latencies, and
  rarely states the basic facts.
- **Llama-3.2-3B says everything twice, when it finishes.**
  - It loops on 20 of 31 nights.
  - On the 11 that finish, it covers nearly everything and hedges all 14
    unsafe items, but also restates each as a plain value.
- **Phi-3.5-mini writes one template per tier.** On high nights it is clean,
  with 4 of 5 facts, and renders. On medium and low nights it hedges the safe
  facts. It never loops.

---

## 7. What was chosen, and why

**What "chosen" means here.** No model is usable as prompted, so nothing is
chosen for deployment as it stands. The choice is **which model to train**,
if the pre-registered 2F rule, at 31 reference nights, routes to
distillation. Formally, this is a recommendation. The selection itself is
made at 2G, on dev.

### The decision matrix

| criterion | Qwen2.5-1.5B | SmolLM2-1.7B | Gemma-2-2b | Llama-3.2-3B | Phi-3.5-mini |
|---|---|---|---|---|---|
| finds the mandatory evidence (cited) | **0.774** | 0.019 | 0.219 | 0.355 | **0.800** |
| states the mandatory facts correctly | **0.497** | 0.019 | 0.219 | 0.297 | 0.368 |
| copies numbers exactly | **1.000** | 0.941 | **1.000** | **1.000** | **1.000** |
| finishes its answer (no loop) | **30/31** | 1/31 | **31/31** | 11/31 | **31/31** |
| valid JSON | 30/31 | 1/31 | **31/31** | 11/31 | **31/31** |
| uses the hedged form | never | never | narrowly | **broadly, with duplicates** | wrongly |
| peak memory (context 4,096) | **1,812 MiB** | 2,615 MiB | 3,151 MiB | 3,840 MiB | 5,103 MiB |
| KV cache per token | **28 KiB** | 192 KiB | 104 KiB | 112 KiB | 384 KiB |
| predicted time per report (Pi 5) | **124–177 s** | 169–241 s | 227–325 s | 266–380 s | 365–522 s |
| kind of error | semantic: hedging, keys | looping | omission | duplication and looping | fixed template |
| **verdict** | **lead student** | not recommended | reserve | **capacity comparison** | not recommended |

### Why Qwen2.5-1.5B leads

A student is judged by what training would have to add. The ideal starting
point already has the skills that are hard to teach, and lacks only the
mapping the training targets show on every night.

1. **It already has the hard skills.** It finds the evidence (77% of
   mandatory facts cited), copies numbers exactly (1.000), finishes its
   answers (30 of 31) and writes valid JSON (30 of 31). No other model does
   all four.
2. **Its errors are exactly what the targets teach.**
   - It never hedges, and it attaches keys to the wrong evidence.
   - Every training target shows the correct hedging of all 14 items and the
     correct key bindings, on every one of 123 nights.
3. **It is by far the best to deploy.** It has the smallest KV cache (28 KiB
   per token, thanks to grouped-query attention), the lowest memory (1.8 GB)
   and the fastest predicted reports (124–177 s on a Pi 5). It is also the
   fastest to train and evaluate.
4. **It has the highest mandatory coverage as prompted (0.497).** It starts
   closest to the part of the contract that matters most.

### Why Llama-3.2-3B trains alongside it

- **It is the only model that hedges all 14 items** on the nights it
  finishes, and the only one to reach 5/5 at all. Those are point
  observations, though: 2 nights pinned, 4 unpinned.
- **Training it answers whether capacity is the limit.** If Qwen fails and
  Llama succeeds, the next student should be larger. If both succeed, Qwen
  wins on footprint.
- **It is not the lead because** it loops on 20 of 31 nights, duplicates
  every hedged item, needs about twice Qwen's memory, and is about 2× slower
  per report.

### Why not the other three

- **Gemma-2-2b is the reserve.** It never loops, is always valid, and uses
  the hedged form. But it rarely states the basic facts (0.219), hedges only
  one pair, and caches 3.7× Qwen's KV per token.
- **Phi-3.5-mini is not recommended.**
  - Its nine clean reports are one fixed template for high nights, with 4 of
    5 facts and no discretionary items.
  - On other nights it hedges the facts that are safe to state.
  - It lacks grouped-query attention (13.7× Qwen's cache per token), and it
    is the heaviest and slowest of the five.
- **SmolLM2-1.7B is not recommended.** It loops on 30 of 31 nights, so it
  rarely produces a usable answer at all. It also lacks grouped-query
  attention, and it is not the lightest model despite having the smallest
  file.

---

## 8. Significance for the project

1. **It settles the size question, for this range.** Across five families from
   1.5B to 3.8B, none meets the contract, and more parameters moved the
   failure without removing it. **Prompting a small model is not enough.**
   This is the evidence that makes distillation necessary, rather than
   choosing a bigger model.
2. **It tells us what training must teach.** The errors are about mapping,
   not reading or arithmetic: the models find the evidence and copy the
   numbers, but put them in the wrong claim type or under the wrong key. That
   is a narrow, learnable skill, and every training target demonstrates it.
3. **It picks the student on evidence.** Qwen2.5-1.5B combines the best
   existing skills with the best deployment profile. Llama-3.2-3B, trained
   alongside, tests whether capacity matters.
4. **It corrects a common assumption about deployment.** File size ranks the
   models wrongly. Grouped-query attention decides memory and speed on a
   Pi-class device, and a smaller file can cost more memory.
5. **It shaped the evaluation method.**
   - Llama's date sensitivity showed that single nights are point
     observations, so all reporting uses population figures across 31
     nights.
   - Every generation logs whether it hit the token limit.
   - Chat templates are pinned wherever they read the date.

---

## 9. Caveats

- **The reference is interim:** 19 of 31 saved and 8 scored, 6 of those high
  confidence. Whether the contract is achievable is decided at 31 nights, and
  training depends on that rule.
- **"Chosen" means recommended, not selected.** The formal selection happens
  at 2G, on dev, after the rule.
- **Single nights, and small counts built from them, are point
  observations.** The 5/5 counts in particular.
- **Deployment figures are analytical.** Memory was measured on x86, and
  throughput was predicted under an assumed bandwidth. Nothing was run on a
  Pi.
- **Host timings are x86 run-log figures,** at context 12,288, and are not
  deployment figures.

---

## 10. Where the data lives, and how to reproduce it

| path | contents |
|---|---|
| `candidates/cache/<model>/P1/` | all 155 generations, keyed by recording, model, prompt id and prompt hash (plus the pinned template, for Llama) |
| `candidates/cache/Llama-3.2-3B-Instruct/P1-unpinned-date/` | the archived unpinned Llama run |
| `candidates/logs/runs.jsonl` | every generation's time, tokens, peak memory, exit status and token-limit flag |
| `candidates/results/<model>/summary.json` | each model's full scoring, overall and per tier |
| `candidates/results/comparison.txt` | the five-model table |
| `deploy/results/` | GGUF architecture, memory and throughput figures |

```
python -m candidates.run --model Qwen2.5-1.5B-Instruct          # plan only, sends nothing
python -m candidates.run --model Qwen2.5-1.5B-Instruct --go     # generate any uncached nights
python -m candidates.score --model Qwen2.5-1.5B-Instruct        # score one model
python -m candidates.score --compare                            # rebuild the five-model table
```

**Key commits:**

- `02697f9`: Qwen on 31 nights;
- `60866d0`: the other four models;
- `4ded089`: Llama's pinned re-run.

**Integrity:** 394 tests pass, and the test packets have been untouched
throughout (md5 `050fffe46d035008d643435ee826dd92`).
