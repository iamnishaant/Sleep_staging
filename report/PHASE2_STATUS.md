# Phase 2 status report: the language-model tier

**Sleep-EDF sleep-staging report pipeline · Team 40 · Project 48**
**Status as of 14 September 2026 · branch `team-40`**

This is the readable overview. The full record, with every decision, number
and correction, is `report/PHASE2_NOTES.md`. The deterministic tier (schema,
grammar, verifier, renderer, coverage, oracle, evaluator) is documented in
`report/README.md`.

---

## 1. Where we are, in one paragraph

Phase 1 built a deterministic safety layer that turns one night of sleep
staging into a verified report. A language model may only emit structured
claims. Every claim is checked against the night's evidence packet, and only a
fully clean claim set is rendered into fixed clinician-register wording.

Phase 2 asks how well a model can fill that contract.

The deterministic tier is **finished, frozen and tested** (394 tests). The
**local small-model tier** has been measured: all five candidates, on all
31 dev nights, under P1. None reaches the contract. The best is 2 of 31
nights at 5/5 (Llama-3.2-3B, date pinned), and neither passes. The
**reference run** with a strong hosted model (Gemini 3.8 Flash) is **in
progress**, at 8 of 31 dev nights. **Deployment numbers** for a Raspberry Pi
5-class target are done: measured memory and analytical throughput.

Next:

- finish the reference run;
- apply the pre-registered rule.

The 2G grammar ablation is designed and waiting in
`report/PHASE2G_ABLATION.md`.

---

## 2. Headline results so far

### 2.1 The reference model satisfies the contract: interim, 8 of 31 nights

Gemini 3.8 Flash was run with thinking off, temperature 0, the structured-output
schema, and prompt P1.

| figure | result, over 8 responses |
|---|---|
| verifier violations | **0 on all 8** |
| mandatory coverage (the 5 robust facts) | **5/5 on all 8** |
| discretionary coverage (the 14 hedged facts) | **14/14 on all 8** |
| numbers transcribed exactly | 17/17 on every night |
| unsafe items correctly hedged | 14 `hedged_value` claims per night |
| reports that render | 8 of 8 |
| the reference set's own oracle recovery | 1.0 on every night |

**This is interim and not the decision basis.** The 8 are six high-confidence
nights, one medium and one low, in manifest order, so the harder tiers are
under-sampled. The pre-registered decision rule applies at 31 nights (§6).

### 2.2 The local small models do not: one night, then all 31, then all five

This was Qwen2.5-1.5B-Instruct (Q4_K_M, llama.cpp, grammar-constrained,
temperature 0) on dev night SC4111E0, run under five prompt variants. It is
exploratory: one night and one seed per run, and it selected nothing.

| run | prompt | claims | violations | mandatory verified | hedged values | renders |
|---|---|---:|---:|---|---:|---|
| raw | no chat template | 18 | 17 | 2/5 | 0 | no |
| A | chat template | 5 | 3 | 3/5 | 0 | no |
| B | A + 2 examples | 3 | **0** | 2/5 | 1 | yes |
| B′ | A + 2 different examples | 4 | 1 | 2/5 | 1 | no |
| C | A + claim-shape rules | 5 | 2 | **4/5** | 0 | no |

What the five runs showed:

- **It finds the evidence and copies it exactly.** The raw run cited 18 of the
  19 items, and all 19 numbers across the runs were exact.
- **Examples control which items get reported.** B reported its examples' two
  items; B′, given different examples, reported exactly the new two.
- **Rules moved coverage, not shape.** C reached 4/5 but used no hedged
  values, bundling 13 items into one observation.
- **It never binds the N1 key to its evidence.** The key appears in four runs,
  always on the wrong item.
- **A clean report can be incomplete.** B had zero violations with only 2/5
  mandatory items. That is why the evaluator reports coverage alongside
  verification.

On the same night, the reference model scored 5/5 with zero violations.

**Across all 31 dev nights**, Qwen2.5-1.5B under P1 (the same settings, run on
13 September 2026) gave:

- **0 of 31** nights at 5/5 mandatory;
- a mandatory mean of 0.497, with a median of 2/5 and a best of 4/5, reached
  on two nights;
- `hedged_value` in **0 of 31 nights (0 of 434 possible claims)**;
- **0** reports rendered.

So the one-night diagnostic generalises: the model never learns the hedged
form. See PHASE2_NOTES, "Local candidates on the dev population".

**All five local candidates** were run at the same settings on 13 and 14
September 2026:

| model | nights at 5/5 | mandatory mean | hedged packets | hedged claims | rendered | hit the cap |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0/31 | 0.497 | 0/31 | 0/434 | 0/31 | 1/31 |
| SmolLM2-1.7B | 0/31 | 0.019 | 0/31 | 0/434 | 0/31 | 30/31 |
| Gemma-2-2b | 0/31 | 0.219 | 31/31 | 50/434 | 1/31 | 0/31 |
| Llama-3.2-3B | 2/31 | 0.297 | 11/31 | 159/434 | 0/31 | 20/31 |
| Phi-3.5-mini | 0/31 | 0.368 | 20/31 | 102/434 | 9/31 | 0/31 |

- **No candidate satisfies the contract on any night.** Llama's two 5/5
  nights also restate the hedged items as plain values. Phi's nine clean
  reports stop at 4 of 5 facts.
- **Each fails differently:**
  - **Qwen** never hedges.
  - **SmolLM2** loops to the token cap.
  - **Gemma** hedges only the REM-latency pair, and omits most facts.
  - **Llama** hedges all 14 items but duplicates each as a plain value.
  - **Phi** hedges the safe facts on medium and low nights.
- **Numbers are never the problem:** numeric fidelity is 1.000 for four of
  the five.
- **One Phi packet (SC4081E0) first failed on host memory,** before
  generating a token. It was generated once on 14 September, and is
  included above.
- **Llama was re-run with its date pinned,** using
  `--chat-template-file student/templates/Llama-3.2-3B-Instruct.pinned.jinja`.
  Only 3 of its 31 outputs matched the unpinned run, and its 5/5 count moved
  from 4 to 2. Its figures here are the pinned run's.
- **The reference model, for contrast:** on its 8 nights so far, it scored
  5/5, with 14 hedged values and a rendered report, every night.

See PHASE2_NOTES, "all five models under P1".

### 2.3 Deployment: plausible within a Raspberry Pi 5 envelope, for batch use

Resource-constrained evaluation was performed on x86 under Windows, with the
Raspberry Pi 5 (8 GB, 4 cores, LPDDR4X-4267) as the named analytical target.
Throughput figures are analytical bandwidth-bound estimates, not on-device
measurements. Peak memory and file size are measured directly on the x86
evaluation host. **Nothing was run on a Pi 5.**

| model | file size | peak memory (x86 host) | vs 8 GB | predicted tok/s, assumed 7–10 GB/s | predicted generation time, prefill excluded |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 1,066 MiB | 1,812 MiB | 22% | 6.7–9.5 | 124–177 s |
| SmolLM2-1.7B | 1,007 MiB | 2,615 MiB | 32% | 4.6–6.5 | 169–241 s |
| Gemma-2-2b | 1,629 MiB | 3,151 MiB | 38% | 3.6–5.1 | 227–325 s |
| Llama-3.2-3B | 1,926 MiB | 3,840 MiB | 47% | 3.1–4.4 | 266–380 s |
| Phi-3.5-mini | 2,282 MiB | 5,104 MiB | 62% | 2.1–3.0 | 365–522 s |

- **The bandwidth is an assumption.** The 7–10 GB/s effective range is an
  assumed input, about 40–60% of the Pi 5's theoretical 17 GB/s. It is not a
  measured property of the hardware.
- **Report times are floors.** Generation times exclude prefill, so an
  end-to-end report takes strictly longer.
- **The conclusion is plausibility, not a demonstration.** The reporting
  workload is plausible within a Pi 5-class envelope **for overnight batch
  reporting**, and not plausible for interactive use.
- **A research finding: file size ranks the models wrongly.** Quantized size,
  the usual edge-feasibility metric, gives the wrong ranking here. SmolLM2
  and Phi-3.5 have no grouped-query attention (verified from the model
  headers: 32 KV heads each). So their KV cache per token is 6.9× and 13.7×
  Qwen's. SmolLM2 has the smallest file, yet it has the second-largest cache
  per token, peaks 800 MiB above Qwen, and spends the largest share of decode
  traffic on its cache (31%).

---

## 3. What was built, and why

Each step exists to make a later measurement trustworthy.

| step | what | why |
|---|---|---|
| Phase 1 | Claim schema, GBNF grammar, two-layer verifier (30 violation codes), deterministic renderer | A model supplies keys and numbers, never prose. Anything it cannot back with evidence is inexpressible or rejected. |
| 2A | The 31-night dev set; manifests; split guards; the 29-night test set locked (md5 `050fffe…26dd92`) with an access log | Every selection decision is made on dev. The test set stays untouched until the end. |
| 2B | Tier predicates (night-confidence high, medium or low) and a duplication policy | Keys are predicates checked against the packet, not free labels. |
| 2C | Coverage split: 5 mandatory, 14 discretionary; the oracle, a maximal verified claim set, verified on all 60 nights | Every other metric improves when a model says less. Coverage is the counterweight, and the oracle is its ceiling. |
| 2D | Serializer: `build_prompt`, prompt version 2D.1 | One frozen prompt body, so prompt wrappers are the only variable. |
| 2E | Evaluator: per-rule counts, validity rates with denominators, stratified by tier | Per-rule failure shapes, with small strata flagged, without flattering denominators. |
| Pre-2F | Clinician register (A) chosen; renderer provenance audit (a report derives from its packet alone); six golden pins | The rendered text states only facts the packet holds. |
| Local runtime | llama.cpp b10927 (CPU); grammar verified; 5 candidates downloaded and sha256-checked | Local models decode under the same grammar. |
| Diagnostics | Five prompt variants on one night; a `cited_coverage` diagnostic in the evaluator | Separates "found the evidence but mis-shaped the claim" from "didn't find it". |
| 2F | The reference model, a generated `responseSchema`, a cached and resumable runner, and a scorer | Establishes whether the contract is satisfiable at all by a strong model. |
| Deployment | Measured size and peak memory; analytical throughput; Pi 5 named as target | A resource envelope for 2G, without claiming an on-device result. |

### Correctness fixes and safeguards found along the way

- **Rule 10.** An invalid review flag could satisfy it. Rule 10 now checks
  only claims that survive every other check.
- **Render gate.** A report renders only from a result with zero violations.
  A `render_unverified` escape hatch exists for tests only, and a guard stops
  it being called from `report/`.
- **Deployment wording.** Every figure is labelled measured, analytical or
  derived. "Seconds per report" became "predicted generation time (prefill
  excluded)".
- **Runner safeguards:**
  - a per-model daily budget that counts failed attempts;
  - skip-and-requeue after two consecutive 503s on one packet;
  - a session breaker after six consecutive failures;
  - per-session summaries;
  - narration of every decision;
  - each request logs its caller.
- **Incidents.** An unexplained launch was traced to a manual command in the
  IDE terminal. A session that looked idle was in fact running silently, which
  prompted the narration. A test count stated as 350 was corrected to 349.

---

## 4. The reference run: operations

**The setting.** Gemini 3.8 Flash, free tier: 5 requests per minute, 20 per
day, and **failed requests count against the day**. Its thinking is off, to
match the local candidates, which don't reason. The cheaper Flash-Lite model
was **rejected**: it refuses a zero thinking budget (HTTP 400).

**The schedule.** P1 only, across the 31 dev nights. P2 (a prompt-design
variant) is held in reserve.

| Pacific day | session | attempts | responses | 503s | ended |
|---|---|---:|---:|---:|---|
| 12 Sep | 21:07 UTC | 7 | 3 | 4 | daily budget |
| 13 Sep | 09:08 UTC | 4 | 0 | 4 | four failures in a row (old runner) |
| 13 Sep | 09:50 UTC | 16 | 5 | 11 | daily budget |
| **total** | | **27** | **8** | **19 (70%)** | |

**Progress: 8 of 31.** 23 remain. At about 5 responses per 20-attempt day,
that is **roughly 4–5 more days**.

**Running a session** (after 07:00 UTC, when the Pacific day resets):

```
python -m reference.run          # plan only; sends nothing
python -m reference.run --go     # one session; it narrates every attempt, wait and skip
```

A person starts each session; nothing runs on a timer.

---

## 5. Integrity

- **394 tests pass.** Test-packet md5 `050fffe46d035008d643435ee826dd92`,
  unchanged throughout.
- **The frozen tier** (`report/` rules, schema, grammar, coverage, oracle,
  evaluator) is untouched by the 2F and deployment work, which lives in
  `reference/`, `deploy/` and `candidates/`.
- **Nothing is evaluated on the test set** until every selection is final.
- **The API key** lives only in the git-ignored `.env`.

---

## 6. What happens next

### Immediate: this week

1. **Finish the P1 reference run.** One session a day after 07:00 UTC, until
   all 31 are saved.
2. **Score it and apply the rule fixed before any results:**
   - **25 or more of 31 at 5/5 mandatory:** the contract is satisfiable, P2
     is not needed, and we move to 2G;
   - **near zero:** the contract itself is the ceiling;
   - **mid-range, around 12 of 31:** run P2, because prompt design plausibly
     explains the gap.
3. **Done: the five local models on the same 31 dev nights, under P1**
   (§2.2).
   - No candidate reaches the contract on any night.
   - The best is Llama-3.2-3B, at 2 of 31 nights at 5/5 with its date
     pinned, neither of which passes.
   - One Phi packet first failed on host memory. It was generated once on
     14 September, and its failed record is kept in `P1-host-failures/`.

### What we will use

| component | what | status |
|---|---|---|
| Safety layer | The frozen schema, grammar, verifier and renderer. Every model output passes through it. | decided |
| Prompt | P1: the frozen `build_prompt` body plus run C's claim-shape rules. P2 is held in reserve. | in use |
| Reference model | Gemini 3.8 Flash, thinking off, structured output. Its 31 responses decide the 2F rule and measure the gap. | 8 of 31 |
| Local runtime | llama.cpp b10927 (CPU), Q4_K_M, grammar-constrained. The grammar stays, and 2G measures what it buys. | in use |
| Local model, as prompted | None. Not one of the 155 generations both passes verification and covers all five mandatory facts. | not usable |
| Approach | Distillation on the 137 training nights (69 subjects, disjoint from dev and test), with the oracle's verified claim sets as targets. Design: `report/PHASE2_DISTILLATION.md`. | decided; runs if 2F routes there |
| Training | On Kaggle. The training set is built: `student/trainset`, with 123 training and 14 validation nights. | decided |
| Student model | **Qwen2.5-1.5B**, with Llama-3.2-3B trained alongside as the capacity comparison. | recommendation |
| Deployment target | Raspberry Pi 5 (8 GB), overnight batch reporting. | analytical |

**Why Qwen2.5-1.5B:**

- **Coverage:** the highest mandatory coverage of the five (0.497).
- **Clean output:** it loops only once in 31, and its numbers are exact.
- **Size and speed:** by far the smallest and fastest.
  - 28 KiB of KV cache per token;
  - 1.8 GB peak at the deployment context;
  - 15 s a night on the host;
  - 124–177 s per report predicted on a Pi 5.
- **Its failures are trainable.** It never hedges, and it attaches text keys
  to the wrong evidence. Both are semantic errors, which is exactly what
  training on a clean reference set targets.

**Why Llama-3.2-3B alongside:**

- **Capacity:** it is the only model to reach 5/5 (2 nights, date pinned),
  and the only one that hedges all 14 items when it finishes.
- **What it tests:** whether that extra capacity is worth about twice the
  memory, and 266–380 s per report on a Pi 5.

**Not recommended:**

- **SmolLM2:** it loops on 30 of 31 nights, and has no grouped-query
  attention.
- **Phi-3.5:** a fixed per-tier template that hedges the safe facts. It has no
  grouped-query attention (13.7× Qwen's KV cache per token) and the slowest
  Pi estimate.
- **Gemma-2:** it rarely states the basic facts, and hedges only the
  REM-latency pair. It is kept as the reserve.

**This is a recommendation, not a selection.** Selection decisions are made on
dev at 2G, after the reference run reaches 31 and the pre-registered rule is
applied.

### Then: 2G

- **The gap:** the local candidates against the reference, per rule and per
  tier, with cited coverage showing *why* each misses.
- **The grammar ablation:** constrained against unconstrained generation, to
  measure what the grammar buys.
  - **Status:** designed and pre-registered, not run.
  - **Where:** `report/PHASE2G_ABLATION.md`.
  - **What it fixes in advance:** the arms, the strict scoring and how each
    outcome will be read, before any unconstrained output exists.
  - **Its built-in check:** the grammar rules out 14 of the 30 violation
    codes, so any of them in the constrained arm means a harness fault.
- **Deployment framing:** the Pi 5-class envelope, written as plausibility for
  batch use.
- **If the gap is large and the contract satisfiable:** distillation, as
  designed in `report/PHASE2_DISTILLATION.md`.
  - **Done:** the 137 training packets are built, and the oracle's claim set
    verifies and renders on all of them.
  - **Decided:** the oracle's claim sets are the targets, and training runs
    on Kaggle.
  - **Done:** the training set, `student/trainset`, with 123 training and 14
    validation nights. Every target is checked against the grammar, the
    verifier and the oracle.
  - **Ready, not run:** the Kaggle training kit (`student/KAGGLE.md`). Its
    chat-template parity is checked against llama.cpp's own rendering.

### Last

- **One evaluation on the locked test set,** with every choice (model,
  prompt, settings) frozen beforehand.

### Deferred, and why

| item | why deferred |
|---|---|
| Bind each verification result to its packet (`recording_id`) | It changes the verification object, which is outside the frozen tier. Candidate for the next non-frozen window. |
| Structured `error_unit` and `scope` fields in the packets | These would rebuild the test packets. |
| A packet-specific grammar | It would hide exactly the hedging errors being measured. |
| A Pi 5-matched container (4 cores, 8 GB) | It would make the memory envelope demonstrated rather than notional. It still would not reproduce ARM or the Pi's memory bandwidth. |
| Prefill cost in the throughput model | Compute-bound, so it isn't captured by the bandwidth model. |

---

## 7. Where things are

| path | contents |
|---|---|
| `report/` | the frozen deterministic tier, and `PHASE2_NOTES.md`, the full lab record; `PHASE2G_ABLATION.md`, the 2G ablation design; `PHASE2_DISTILLATION.md`, the distillation design |
| `reference/` | the reference-model tier: schema, client, runner, scorer, prompts, cache, logs |
| `deploy/` | size, memory and throughput analysis; the GGUF reader; results |
| `candidates/` | the local-candidate harness: runner, scorer, cache, logs, per-model results |
| `student/` | the distillation training set and its builder, the chat-template references, and the Kaggle training kit (`KAGGLE.md`) |
| `tests/` | 394 tests |
| `distillation/results/` | evidence packets and split manifests |
| `distillation/results/phase2_train_packets/` | the 137 training packets for distillation |
