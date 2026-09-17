# The small-language-model tier: what we did, why, and what it means

**Sleep-EDF sleep-staging report pipeline · Team 40 · Project 48**
**Written 15 September 2026.** It covers the work of 12–15 September 2026 on
branch `team-40`.

This README explains the small-language-model (SLM) work in one place: what
was built, why each piece exists, what it found, and why it matters for the
project. Every figure comes from the committed results. The full lab record,
with every decision and correction, is `report/PHASE2_NOTES.md`. The status
overview is `report/PHASE2_STATUS.md`.

---

## 1. In one paragraph

Phase 1 built a deterministic safety layer. It turns one night of sleep
staging into a verified report, and a language model may only fill it with
structured claims that a verifier checks against the night's evidence.
Phase 2 asks whether a model can fill that contract.

We tested five small local models (1.5B to 3.8B parameters) on the same 31 dev
nights, under identical settings. **None met the contract on any night.** Of
155 generations, not one both passes verification and states all five
mandatory facts. The models copy numbers exactly. They fail on *structure*:
which claim type goes with which item, and which evidence a key must cite.
Each model fails differently, and more size did not fix it.

A strong hosted reference model meets the contract on all 8 nights scored so
far. That result is interim, and those 8 nights are mostly the easy ones. The
reference run's final result, at 31 nights, decides what happens next through
a rule written before any results.

---

## 2. Why a small language model at all

The project's end goal is a sleep report produced **on a small device**. The
named analytical target is a Raspberry Pi 5 (8 GB, 4 cores), running
overnight in batch.

A hosted model cannot be the deployed component, but a small model that
runs locally can. So the question is not "can any model do this?" It is
**"can a model small enough to run on the device do this?"**

**The safety design shapes the whole task.**

- **The model never writes prose.** It emits a JSON array of *claims*, such
  as "total sleep time is 410 minutes, citing `arch.total_sleep_time`".
- **Every claim is checked** by a frozen two-layer verifier: 30 violation
  codes across structure and policy.
- **Only a fully clean claim set becomes a report,** and it is rendered into
  fixed clinician wording.

This makes the model's job narrow and checkable: choose the right claims,
with the right types, citing the right evidence.

---

## 3. The contract the model must meet

Each night's evidence packet has **19 evidence items.**

- **5 mandatory items** are safe to state as plain values: total sleep time,
  time in bed, sleep efficiency, the night-confidence tier and the N1
  reliability warning.
- **14 discretionary items** carry measurement error, so they may only be
  stated as `hedged_value` claims, never as plain `value`s. The renderer then
  adds their error bound.

**A report meets the contract** when it:

- passes verification with zero violations;
- covers all five mandatory facts ("5/5");
- hedges the unsafe items correctly.

**What we measure**, all through the frozen verifier and evaluator:

- the count of nights at 5/5 mandatory;
- mandatory and discretionary coverage;
- the claims cited;
- oracle recovery;
- numeric fidelity;
- reports rendered;
- `hedged_value` as two numbers: nights using it, and claims out of 434
  (14 × 31).

We report these overall and per confidence tier (11 high, 10 medium and 10
low nights).

---

## 4. How we measure, and why this way

- **Dev only.** All 31 dev nights are the selection set. The 29 test nights
  stay locked, and their md5 is checked before and after every step.
- **One generation per night, at temperature 0 and seed 0.** llama.cpp is
  deterministic here: regenerating a night reproduces it byte for byte. A
  poor output is therefore an observation, and re-running it would be
  best-of-N under another name.
- **Grammar-constrained decoding** (`report/claims.gbnf`). The output is
  always a syntactically legal claim list, unless it runs out of tokens.
- **The unit of reporting is the population of 31 nights, not single
  nights.** Section 10 shows why.

---

## 5. The reference model: is the contract achievable at all?

**Why it exists.** If no model can meet the contract, the local models'
failures would say nothing about size. We needed to know whether the task is
achievable by a capable model.

**The model:** Gemini 3.8 Flash, with thinking off (to match the local models,
which don't reason), temperature 0, structured output, and the same prompt
(P1).

**The rule, fixed before any results,** applied at 31 nights:

| nights at 5/5 | reading | next |
|---|---|---|
| 25 or more | the contract is satisfiable | 2G, then distillation if the gap is large |
| around 12 | prompt design may be the limit | run the reserve prompt P2 |
| near zero | the contract itself is the ceiling | report that finding |

**Where it stands (17 September).** 19 of 31 responses are saved, and 8 are
scored. All 8 scored nights reach 5/5 with zero violations, 14/14
discretionary items, 17/17 numbers exact and 14 `hedged_value` claims each.
Every one renders.

**The caution.** The run's manifest order front-loads high-confidence nights:

- 9 of the 19 saved nights are high-confidence;
- the 12 still to come are 2 high, 4 medium and 6 low.

Interim figures should be expected to fall as the harder tiers arrive. A
mid-range final result would be consistent with the interim, not a reversal
of it.

**Operations.** The free tier allows 20 requests a day, and failed requests
count against it.

- **So far:** 87 attempts, 19 saved and 68 HTTP 503 "high demand" errors
  (78%). That is about 4 saved nights per full day.
- **The runner** saves every response, never re-requests a saved night, moves
  on after two 503s on one night, and stops after six failures in a row.
- **A person starts every session.** Nothing runs on a timer.

---

## 6. The local runtime, and why the grammar matters

**The setup:**

- **Runtime:** llama.cpp build b10927, on CPU.
- **Models:** all five as Q4_K_M quantised weights, each checked against its
  published sha256.
- **Invocation:** `--jinja -cnv -st`, so each model's own chat template is
  used, with a 3,000-token cap and a context of 12,288.

**The grammar makes 14 of the 30 violation codes impossible to generate:**

- 13 of the 15 structural codes;
- the subject check.

A 15th code cannot occur on this data either.

- **Why that matters:** the grammar makes structural failure impossible, so
  the local models' failures are about *meaning*, not formatting.
- **What it does not do:** it never forces the claim list to close. That
  becomes important in section 9.

---

## 7. First look: one model, one night

Before the full run, Qwen2.5-1.5B was probed on a single dev night
(SC4111E0) under five prompt variants. This was exploratory and selected
nothing.

| run | prompt | violations | mandatory | hedged values |
|---|---|---:|---:|---:|
| raw | no chat template | 17 | 2/5 | 0 |
| A | chat template | 3 | 3/5 | 0 |
| B | A + 2 examples | 0 | 2/5 | 1 |
| B′ | A + 2 different examples | 1 | 2/5 | 1 |
| C | A + claim-shape rules | 2 | 4/5 | 0 |

**What it taught:**

- **Examples decide what gets reported.** B reported its examples' two
  items; B′ reported exactly the new two. So examples are excluded from the
  prompts.
- **A clean report can still be incomplete.** B had zero violations with 2 of
  5 facts, which is why coverage sits beside verification.
- **The chosen prompt.** C's claim-shape rules became prompt **P1**, used for
  both the reference and the local models.

---

## 8. The five-model gap measurement

**The models:** Qwen2.5-1.5B, SmolLM2-1.7B, Gemma-2-2b, Llama-3.2-3B and
Phi-3.5-mini.

**Identical settings:** P1, the grammar, temperature 0, seed 0, one
generation per night and no retries.

**The cache key** is (recording, model, prompt id, prompt hash). Llama's
chat-template date is also pinned (section 10).

| model | nights at 5/5 | mean mandatory | hedged nights | hedged claims | rendered | hit the cap | top violation |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B | 0/31 | 0.497 | 0/31 | 0/434 | 0/31 | 1/31 | key dependency missing (53) |
| SmolLM2-1.7B | 0/31 | 0.019 | 0/31 | 0/434 | 0/31 | 30/31 | malformed JSON (30) |
| Gemma-2-2b | 0/31 | 0.219 | 31/31 | 50/434 | 1/31 | 0/31 | key predicate false (22) |
| Llama-3.2-3B | 2/31 | 0.297 | 11/31 | 159/434 | 0/31 | 20/31 | unsafe item not hedged (145) |
| Phi-3.5-mini | 0/31 | 0.368 | 20/31 | 102/434 | 9/31 | 0/31 | safe item hedged (60) |
| *Reference, 8 scored nights (interim)* | *8/8* | *1.000* | *8/8* | *112/112* | *8/8* | *0* | *none* |

**Mandatory facts verified, nights per band:**

| model | 0–1 of 5 | 2–3 of 5 | 4 of 5 | 5 of 5 |
|---|---:|---:|---:|---:|
| Qwen | 1 | 28 | 2 | 0 |
| SmolLM2 | 30 | 1 | 0 | 0 |
| Gemma | 28 | 3 | 0 | 0 |
| Llama | 20 | 0 | 9 | 2 |
| Phi | 20 | 2 | 9 | 0 |

**By tier (high · medium · low):**

| model | nights at 5/5 | mean mandatory | rendered |
|---|---|---|---|
| Qwen | 0 · 0 · 0 | .47 · .38 · .64 | 0 · 0 · 0 |
| SmolLM2 | 0 · 0 · 0 | .06 · .00 · .00 | 0 · 0 · 0 |
| Gemma | 0 · 0 · 0 | .25 · .20 · .20 | 1 · 0 · 0 |
| Llama | 0 · 0 · 2 | .44 · .24 · .20 | 0 · 0 · 0 |
| Phi | 0 · 0 · 0 | .76 · .10 · .20 | 9 · 0 · 0 |

**Across the five models:**

- **Numbers are never the problem.** Numeric fidelity is 1.000 for four of
  the five. SmolLM2's one valid output scored 0.941.
- **Three models learn the hedged form, and none applies the rule.** The
  rule is to hedge exactly the 14 unsafe items and nothing else.
- **51 of 155 generations ran to the 3,000-token cap** and were cut off
  mid-JSON. Every claim in them is grammar-legal.

---

## 9. How each model fails

- **Qwen2.5-1.5B states the facts but never hedges.** It has the highest
  mandatory coverage of the five, but it attaches the N1 key to the wrong
  evidence (53 times) and bundles both REM latencies into one claim (26).
- **SmolLM2-1.7B lists values until it is cut off.** It writes only plain
  values: it walks the 17 numeric items once, then repeats the five stage
  fractions until the token cap, on 30 of 31 nights.
- **Gemma-2-2b is short and hedges one pair.** It writes 4–7 claims a night
  and never loops. It uses the hedged form every night, but only for the two
  REM latencies, and it rarely states the basic facts.
- **Llama-3.2-3B says everything twice, when it finishes.**
  - It loops on 20 of 31 nights.
  - The 11 nights that finish cover nearly everything and hedge all 14
    unsafe items, but they also restate each one as a plain value: 145
    violations.
- **Phi-3.5-mini writes one template per tier.**
  - On high nights it is clean, states 4 of 5 facts, and renders (9
    reports).
  - On medium and low nights it hedges the three facts that are safe to
    state (60 violations).
  - It never loops.

**The pattern is that these are errors of mapping, not of reading.** Every
model finds the evidence, but none reliably maps each item to its correct
claim type and key.

---

## 10. Llama's date sensitivity, and what it means for reporting

**Llama 3.2's chat template writes the current date into every prompt.** Its
first run carried the date of the day it ran. To make the run reproducible,
it was re-run with the date pinned to the template's own fallback
("26 Jul 2024"), using
`--chat-template-file student/templates/Llama-3.2-3B-Instruct.pinned.jinja`.

| Llama-3.2-3B | unpinned (archived) | pinned (official) |
|---|---:|---:|
| nights at 5/5 | 4 | 2 |
| mean mandatory | 0.310 | 0.297 |
| hedged claims | 156/434 | 159/434 |
| hit the token cap | 20/31 | 20/31 |
| outputs identical to the other run | | 3 of 31 |

**A date string that says nothing about the task changed 28 of 31 outputs.**

- Five nights started looping, and five stopped.
- Ten nights changed their count of mandatory facts.
- Only one night, ST7081J0, is at 5/5 in both runs.

**The inference, which applies to every model here:** at temperature 0 with
one generation per night, a single night is a point observation. The overall
behaviour held, but individual nights, and small counts built from them, do
not.

- **The reportable unit is the population figure across 31 nights.**
- **Llama's 2/31 could as easily have been 0 or 4** under another irrelevant
  change.

---

## 11. Run integrity

- **Determinism, checked.** One night was regenerated for Gemma, Phi and
  SmolLM2, outside the cache. All three came out byte-identical to the
  committed outputs (52 s, 137 s and 301 s).
- **Token-limit flag.** Every generation logs `hit_token_limit`, so looping
  is visible rather than inferred from malformed JSON.
- **A host-memory failure, recorded and resolved.** Phi's first night could
  not allocate its 4.8 GB KV cache and produced no tokens. On the user's
  decision it was generated once, and the failed record is kept.
- **Session restarts.** Two packets were interrupted when a session ended.
  Neither was saved, and each ran once on resume.

---

## 12. Host measurements, and a check on the memory model

The peak memory of each generation was measured on the x86 evaluation host,
at context 12,288. This belongs in the run log only, never in the deployment
section.

**The prediction to check:** the deployment table's peak at context 4,096,
plus the KV cache for the 8,192 extra positions.

| model | median time per night | predicted peak | measured peak |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 14.9 s | 1,812 + 224 = 2,036 MiB | 2,038 MiB |
| SmolLM2-1.7B | 204.1 s | 2,615 + 1,536 = 4,151 MiB | 4,153 MiB |
| Gemma-2-2b | 31.7 s | 3,151 + 832 = 3,983 MiB | 3,594 MiB |
| Llama-3.2-3B | 275.4 s | 3,840 + 896 = 4,736 MiB | 4,743 MiB |
| Phi-3.5-mini | 139.1 s | 5,104 + 3,072 = 8,176 MiB | 8,170 MiB (maximum) |

- **The model predicts four of the five peaks** to within 7 MiB.
- **Gemma is the exception.** It comes in about 390 MiB lower, consistent
  with its sliding-window layers, whose cache llama.cpp does not grow with
  context. This explanation is not verified.
- **Total run time was about 5 h 46 min** for all five models.

---

## 13. The deployment envelope: Raspberry Pi 5, analytical

Nothing was run on a Pi.

- **Memory** was measured on the x86 host, at deployment context 4,096.
- **Throughput** is a bandwidth-bound prediction, under an **assumed** 7–10
  GB/s effective memory bandwidth. The Pi 5's theoretical peak is about
  17 GB/s.

| model | file size | peak memory | KV cache per token | predicted tokens/s | predicted generation time (floor) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 1,066 MiB | 1,812 MiB | 28 KiB | 6.7–9.5 | 124–177 s |
| SmolLM2-1.7B | 1,007 MiB | 2,615 MiB | 192 KiB | 4.6–6.5 | 169–241 s |
| Gemma-2-2b | 1,629 MiB | 3,151 MiB | 104 KiB | 3.6–5.1 | 227–325 s |
| Llama-3.2-3B | 1,926 MiB | 3,840 MiB | 112 KiB | 3.1–4.4 | 266–380 s |
| Phi-3.5-mini | 2,282 MiB | 5,104 MiB | 384 KiB | 2.1–3.0 | 365–522 s |

- **Plausible for overnight batch, not for interactive use.** A report takes
  minutes, and these generation times exclude prompt processing, so they are
  floors.
- **File size ranks the models wrongly.**
  - SmolLM2 and Phi lack grouped-query attention, so they cache 6.9× and
    13.7× more per token than Qwen.
  - SmolLM2 has the smallest file of the five, yet it peaks 800 MiB above
    Qwen.

---

## 14. The grammar ablation (2G): designed, not run

**What it asks:** what does the grammar buy?

**How:** the same five models on the same 31 nights, with the grammar removed
and everything else byte-identical. That is 155 new generations; the P1 run
is the grammar-on arm.

**Fixed in advance** (`report/PHASE2G_ABLATION.md`):

- **Strict scoring:** a code fence or a line of prose counts as malformed.
- **One declared exploratory reading,** which strips a single wrapping code
  fence.
- **Paired per-night tables.**
- **A written reading for each possible outcome.**

**Its most useful question:** does the open-ended claim list cause the loops?

**The harness is built and tested.** It runs only once the 2F rule routes to
2G.

---

## 15. What this means for the project

1. **The gap is not a 1.5B quirk.** Across five models from 1.5B to 3.8B, none
   meets the contract. Larger models moved the failure around, but did not
   remove it. For these models, prompting alone does not reach the contract.
2. **The failures are learnable in kind.** The models read the evidence and
   copy numbers exactly; what they lack is a fixed mapping from each item to
   its claim type and key. That is the kind of skill supervised training
   teaches, which is what makes distillation a sensible next step
   (`report/README_DISTILLATION.md`), *if* the rule routes there.
3. **Whether the contract is achievable is still open.** The reference run
   answers it at 31 nights, and its remaining nights are the harder ones.
4. **The evaluation method is itself a result.**
   - Population figures, not single nights.
   - A pinned chat template.
   - A logged token-limit flag.
   - A rule written before the results.

   These make the numbers defensible.
5. **Deployment is plausible, not demonstrated.**
   - **Memory:** Qwen fits a Pi 5-class envelope easily, with a peak of
     1.8 GB at deployment context.
   - **Speed:** the predicted report times suit overnight batch use.
   - **Model choice:** grouped-query attention matters more than file size.

---

## 16. What comes next

1. **Finish the reference run.** 17 nights remain, about 4 more full daily
   sessions.
2. **Apply the pre-registered rule at 31.**
3. **Depending on the result:**
   - **25 or more:** 2G (the gap analysis and the grammar ablation), then
     distillation;
   - **around 12:** run P2;
   - **near zero:** report the contract as the ceiling.
4. **One evaluation on the locked test set,** with every choice frozen first.

---

## 17. Where things are, and how to reproduce them

| path | contents |
|---|---|
| `report/` | the frozen tier, `PHASE2_NOTES.md` (the lab record), `PHASE2_STATUS.md` (the status overview), `PHASE2G_ABLATION.md` (the 2G design) |
| `reference/` | the reference runner, scorer, prompts, cache and logs |
| `candidates/` | the local-candidate harness, the cache of all 155 generations, the run log, per-model results, and `results/comparison.txt` |
| `deploy/` | the GGUF reader, memory measurement and throughput model |
| `student/templates/` | the chat-template references, including Llama's pinned template |

```
python -m candidates.run --model Qwen2.5-1.5B-Instruct          # plan only, sends nothing
python -m candidates.run --model Qwen2.5-1.5B-Instruct --go     # generate the uncached nights
python -m candidates.score --model Qwen2.5-1.5B-Instruct        # score one model
python -m candidates.score --compare                            # the five-model table
python -m reference.run --go                                    # one reference session
```

**Integrity:** 394 tests pass, and the test-packet md5
(`050fffe46d035008d643435ee826dd92`) has been unchanged throughout.

**Key commits:**

- `60866d0`: the five-model run;
- `4ded089`: Llama's pinned re-run;
- `b14de32`: the ablation harness;
- `99db416`: the reference run at 14 of 31.
