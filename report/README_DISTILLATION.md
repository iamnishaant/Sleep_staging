# Distilling the report writer: what we did, why, and what it means

**Sleep-EDF sleep-staging report pipeline · Team 40 · Project 48**
**Written 15 September 2026.** It covers the work of 14–15 September 2026 on
branch `team-40`.

This README explains the distillation work: what was built, why each piece
exists, and why it matters for the project. The pre-registered design is
`report/PHASE2_DISTILLATION.md`, the Kaggle instructions are
`student/KAGGLE.md`, and the full lab record is `report/PHASE2_NOTES.md`.

> **Two kinds of distillation live in this repository.**
>
> - **Phase 1** distilled the *sleep-staging* network into a small student,
>   `student_N4kd`, whose outputs build every evidence packet. That work lives
>   in `distillation/`.
> - **This README is about Phase 2:** training a small *language model* to
>   write the verified claim sets that become the report.

---

## 1. In one paragraph

No small local model meets the report contract when prompted. That was
measured on 31 dev nights across five models (`report/README_SLM.md`). The
failures are errors of mapping, which training can teach.

We have therefore built everything needed to train a small model, if the
project's pre-registered rule calls for it:

- **137 training nights,** kept apart from dev and test by subject;
- **verified targets** for every one;
- **a checked training set;**
- **a Kaggle training kit** whose chat-template formatting is verified
  against what llama.cpp shows the model at evaluation.

**Nothing has been trained.** Training runs only if the reference run clears
the pre-registered bar at 31 nights.

---

## 2. Why distil

**The evidence** comes from the five-model run:

- none of 155 generations both passes verification and states all five
  mandatory facts;
- every model except SmolLM2 copies numbers exactly;
- what they get wrong is which claim type goes with which evidence item,
  which evidence each key must cite, and where to stop.

These errors are about a fixed mapping from packet to claims, and a fixed
mapping is what supervised fine-tuning teaches.

**The goal** is still a report written on a small device (a Raspberry Pi
5-class target). A hosted model cannot be deployed, and a prompted small
model is not good enough. Training a small model to reproduce verified claim
sets is the remaining route.

---

## 3. When: the gate

Distillation runs only if both of these hold:

1. **The 2F rule,** applied at 31 reference responses, finds the contract
   satisfiable: 25 or more of 31 nights at 5/5.
2. **2G confirms the gap** between the reference and the local models.

If the rule lands mid-range (around 12), the reserve prompt P2 runs first. If
it lands near zero, the contract itself is the ceiling, and distillation is
moot.

**Where it stands (15 September).** The reference run is at 14 of 31 nights,
8 of them scored.

- **The interim result:** all 8 scored nights reach 5/5.
- **Why it is front-loaded:** the run's order puts high-confidence nights
  first. 8 of the 14 saved are high, and the 17 to come are 3 high, 7 medium
  and 7 low.

The gate is genuinely open.

---

## 4. Training data: which nights, and why

**Not the dev nights.** They are the selection set, so training and then
scoring on them would leak. At 31 nights, they are also too few.

**Not the test nights.** They are locked until the final evaluation.

**The 137 training nights** come from 69 subjects. The staging split
(`distillation/splits.json`) is made at the subject level, so no training
subject appears in dev (16 subjects) or test (15 subjects). A test pins this.

**How they were built:**

1. **Probabilities.** `distillation/cache_train_probs.py` caches the staging
   student's probabilities on the training split.
   - It reuses the existing evaluator's loader and provenance guard,
     unchanged.
   - It writes only to its own folder.
   - It refuses to run if training and held-out recordings overlap.
   - The existing evaluator was not used, because it also rewrites
     held-out results.
2. **Packets.** `build_packet.py --split train` then built 137 packets, with
   none skipped. The same generator and staging model built the dev and
   test packets.
3. **Integrity.** The test-packet md5 was unchanged before and after.

**What was checked:**

- **Same structure as dev.** Every training packet carries the dev packets'
  19 evidence ids and safe-to-state flags exactly.
- **One difference: attribution is null.** No attribution artefact exists
  for the training split, and the prompt builder never reads attribution, so
  the prompts take the same form.

**A measured caveat: these nights look easier.** The staging model was
trained on these nights, so its probabilities are sharper on them. Median
entropy is 0.443 nats, against 0.523 on validation. Night confidence is
computed from that entropy, so the training nights lean high-confidence:

| nights | high | medium | low | total |
|---|---:|---:|---:|---:|
| training | 78 (57%) | 24 (18%) | 35 (26%) | 137 |
| dev | 11 (35%) | 10 (32%) | 10 (32%) | 31 |

**The rule: no rebalancing.** It was fixed after the mix was measured and
before any training.

- **Every tier is represented,** with at least 24 nights, and the tiers
  differ only in the tier key and the low-night review flag.
- **A medium-tier weakness would still be visible,** because dev results
  are reported per tier.

---

## 5. Targets: what the student should write

| option | what | cost |
|---|---|---|
| **A (chosen). Oracle witness** | the oracle's maximal verified claim set for each night | free, instant and deterministic; verified by construction |
| B. Reference model | Gemini 3.8 Flash writes each night | about a month of free-tier quota, at ~4 saved nights a day |
| C. A, checked by B | A for all nights, with B on a sample | about five days of quota |

**Why A** (decided 14 September 2026, applying if training runs):

- The reference model's responses so far recover exactly the oracle's
  content. That is interim: its 8 scored nights are 6 high-confidence.
- So B would spend a month of quota producing what A produces for free, and
  A's targets are clean by construction.

**The oracle works on the training nights too.** Its claim set passes the
verifier and renders on all 137 training nights.

**A problem caught before it mattered: field order.**

- **The problem.** The oracle writes each claim's fields in the order
  `claim_id, subject, cites, claim_type, …`, but `claims.gbnf` requires
  `claim_id, claim_type, cites, subject, …`. The grammar rejects the oracle's
  raw text.
- **Why it matters.** A student trained on it would learn a form that
  llama.cpp blocks at inference.
- **The fix.** Every target is rewritten in the grammar's order.

**Every target is then checked three ways:**

- the frozen grammar matcher (`report/gbnf.py`) accepts it;
- the frozen verifier passes it, and it renders;
- it parses back to exactly the oracle's claims.

**The matcher itself is cross-checked against llama.cpp.** It accepts every
grammar-constrained output from the five-model run that finished normally.

---

## 6. The training set

`python -m student.build_set` writes `student/trainset/`: `train.jsonl`,
`valid.jsonl` and `manifest.json`, with each file's sha256 in the manifest.

| split | nights | subjects | high | medium | low | used for |
|---|---:|---:|---:|---:|---:|---|
| train | 123 | 62 | 70 | 21 | 32 | training |
| valid | 14 | 7 | 8 | 3 | 3 | loss and early stopping; dev is never touched |

- **The held-out subjects** are SC407, SC451, SC476, SC480, SC482, ST706 and
  ST712. The rule was seed 0, redrawn until all three tiers were covered, and
  seed 0 already covered them.
- **Each record** is a user message (the exact P1 prompt the harness sends at
  evaluation) and an assistant message (the target).
- **Length.** The longest prompt and target, in each student's own
  tokenizer, fit well inside the 3,072-token limit:

  | student | prompt | target | total |
  |---|---:|---:|---:|
  | Qwen | 1,077 | 973 | 2,050 tokens |
  | Llama | 1,044 | 933 | 1,977 tokens |

---

## 7. Students and method

**The students:**

- **Qwen2.5-1.5B is the lead.**
  - It has the highest mandatory coverage of the five, loops only once in 31,
    and copies numbers exactly.
  - It is by far the smallest and fastest, with 28 KiB of KV cache per token.
  - Its errors (it never hedges, and misplaces keys) are what the targets
    demonstrate on every night.
- **Llama-3.2-3B is the capacity comparison.** It is the only model to reach
  5/5, though only as point observations. It is the only one that hedges all
  14 items when it finishes. Training it answers whether twice the memory buys
  anything.

**The method: LoRA supervised fine-tuning.** The loss is on target tokens
only. These settings were fixed before training:

| setting | value |
|---|---|
| LoRA | rank 16, alpha 32, dropout 0.05, on the 7 attention and MLP projections |
| optimiser | learning rate 2e-4, cosine schedule, no warmup |
| epochs | 3; one declared fallback of 5 |
| batch | effective batch 8, micro-batch 1 |
| seed | 0 |
| maximum sequence | 3,072 tokens |
| selection | best epoch by validation loss |

---

## 8. Chat-template parity: training on what evaluation will show

**Why it matters.** A student must be trained on exactly the text llama.cpp
feeds it at evaluation, template and all. Otherwise training and evaluation
see different prompts.

**What was done:**

- **The references.** `llama-server` (b10927) rendered one real training
  prompt per student. Those renderings, and the templates embedded in the
  model files, are saved in `student/templates/`.
- **Qwen:** llama.cpp adds Qwen's default system message, and there is no
  date. Training matches evaluation with no changes.
- **Llama: the date problem.** Llama 3.2's template writes the current date
  into every prompt; unpinned, it rendered "14 Sep 2026".
- **Llama: the fix.** The date is pinned to "26 Jul 2024", the template's own
  fallback, in both places:
  - in training, as `date_string`;
  - at evaluation, with
    `--chat-template-file student/templates/Llama-3.2-3B-Instruct.pinned.jinja`.

  Both render identically.
- **BOS.** llama.cpp omits the BOS text from its rendering and adds the token
  when tokenising, and the check allows for that.
- **Checked offline.** Rendering the saved templates the way transformers
  does reproduces llama.cpp's renderings exactly, for both students.
- **Enforced on Kaggle.** The training script refuses to train unless the
  real tokenizer reproduces them too.

**The same pin now applies in the evaluation harness.** Pinning it changed
28 of Llama's 31 outputs. That is how much a single irrelevant line can
move a model at temperature 0, and why the evaluation reports population
figures, not single nights.

---

## 9. Where and how it trains: the Kaggle kit

**Why Kaggle.** This machine cannot realistically train the students.

- Its GPU is an RTX 3050 Ti Laptop with 4 GB, and its PyTorch is CPU-only.
- Llama 3B would not fit, and Qwen 1.5B barely could.
- The project already trained its staging models on Kaggle (16 GB GPUs).

**The kit:**

- **`student/kaggle_sft.py`** trains with the fixed settings.
  - **It refuses to train on:** a data-hash mismatch, a chat-template parity
    failure, a sequence over 3,072 tokens, or a non-finite loss.
  - **It keeps** the best epoch by validation loss, and merges it.
  - **It converts** the result to an f16 GGUF with llama.cpp's own converter
    at b10927.
- **`python -m student.pack_kaggle`** builds `student/sft_bundle.zip`, the
  single upload (49 KiB, 8 files).
- **`student/KAGGLE.md`** gives the steps: upload, GPU, training,
  conversion, then quantising back here.

**Then, back on this machine:**

1. Quantise to Q4_K_M with `llama-quantize` b10927.
2. Evaluate with the unchanged P1 harness: one generation per dev night,
   with the grammar, at temperature 0. The trained model is measured on
   exactly the same footing as the five candidates.

---

## 10. How success is judged

**Fixed before any student exists:**

| complete, clean reports on dev | reading |
|---|---|
| **25 or more of 31** | the student meets the contract; it goes to the test set |
| 12 to 24 | partial: report it, and use the one declared fallback (5 epochs) |
| fewer than 12 | the student does not learn the contract at this size; Llama shows whether capacity is the limit |

**Guards and limits:**

- **Numeric fidelity must stay 1.000.** A student that recomputes numbers
  fails, whatever else it does.
- **Each student gets at most two dev evaluations.**
- **If both students pass,** the smaller deployment footprint (Qwen) goes to
  the test set.
- **The 29 test nights are evaluated once,** with every choice frozen.

---

## 11. What distillation will and will not show

**Being honest about the framing.** The oracle can already write a complete,
verified report for any night, with no model at all. So in this design, the
language model adds no content the deterministic tier lacks.

**What distillation tests** is whether a small, on-device model can *learn a
verified reporting contract*. That is the precondition for any richer use in
which no oracle exists.

- **If Qwen meets the bar:** a 1.5B model in a Pi 5-class memory envelope
  writes complete reports that pass verification, and the verifier
  guarantees every claim.
- **If it does not:** the rule-by-rule gap between the oracle and a small
  model is the finding.

**Why the preparation matters now.** Training can begin the day the rule
fires, because:

- the data is clean;
- the targets are verified;
- the template is checked;
- the success criteria were fixed in advance.

A result reached that way can't be tuned after the fact.

---

## 12. Risks, and how they are handled

| risk | handling |
|---|---|
| Training nights look easier (the staging model saw them) | measured, tier mix reported, no rebalancing, per-tier dev results |
| A student learns a form the grammar forbids | targets in the grammar's field order, checked by the grammar matcher |
| Training and evaluation see different prompts | chat-template parity checked offline and enforced on Kaggle; Llama's date pinned |
| Over-tuning on dev | at most two dev evaluations per student; settings fixed in advance |
| Leakage from dev or test | subject-disjoint split; tests pin it; test packets never opened |
| Reading too much into single nights | population figures across 31 nights are the reportable unit |

---

## 13. Status and next steps

| step | status |
|---|---|
| Training packets (137) | ✅ built |
| Targets (oracle, grammar order) | ✅ decided, built and checked |
| Training set (123 + 14) | ✅ built |
| Kaggle kit and parity references | ✅ built and tested, not run |
| **The gate (2F rule at 31)** | ⏳ waiting on the reference run: 14 of 31 saved |
| Training, dev evaluation and the test set | ⬜ only if the rule routes here |

**Tests:** 17 cover this work. The training packets have 3, the training set
8 and the Kaggle kit 6. The full suite of 394 passes.

**Key commits:**

- `1cc5af3`: the packets and the design;
- `0100dd1`: the training set;
- `223fc05`: the Kaggle kit;
- `c06db86`: the wording that makes training conditional.
