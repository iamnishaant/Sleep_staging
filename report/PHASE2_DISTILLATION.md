# Distillation: design

**Written 14 September 2026, before any student model exists.** The
training packets (section 3) and the training set (section 4.2) are built.
Nothing has been trained. This document fixes the following before any student is trained:

- the training data;
- the students;
- the method;
- the success criteria;
- the test-set rule.

That is the same discipline as the 2F decision rule and the 2G ablation
(`report/PHASE2G_ABLATION.md`).

---

## 1. Why, and when

**Why.** No local model satisfies the contract when prompted:

- **No passing report.** None of the 155 generations in the five-model run
  both passes verification and covers all five mandatory facts.
- **The failures are about structure, not numbers.** Every model except
  SmolLM2 copies numbers exactly. What they get wrong is which claim type
  goes with which item, and which evidence a key must cite.
- **That is a learnable mapping.** Supervised fine-tuning teaches exactly
  this kind of thing.

**When.** Distillation runs only if both of these hold:

1. The 2F rule, applied at 31 reference responses, finds the contract
   satisfiable: 25 or more of 31 at 5/5.
2. 2G confirms the gap between the reference and the local models.

If the rule lands mid-range, P2 runs first. If it lands near zero, the
contract itself is the ceiling, and distillation is moot.

## 2. What the student must learn

The contract, as the verifier enforces it. For every packet:

- the five mandatory facts;
- the 14 unsafe items as `hedged_value` claims;
- tier and N1 keys bound to their own evidence;
- a verified review flag on low nights.

**All of this is determined by the packet.** That is why a small model can
plausibly learn it, and why the oracle can write it (section 7).

## 3. Training data

### 3.1 Which nights

**Not the dev nights.** They are the selection set, so training on them and
then scoring on them would leak. At 31 nights, they are also too few.

**Not the test nights.** They stay locked until the end.

**The 137 training nights,** from 69 subjects:

- **Disjoint from dev and test by subject.** `distillation/splits.json` is
  split at the subject level, and the cache script refuses any overlap.
- **The same generator and staging model** (`student_N4kd`) as the dev and
  test packets.
- **How they are built:** `distillation/cache_train_probs.py` caches the
  student's probabilities on the training split, writing only
  `results/probs_student_N4kd_train/`. Then
  `build_packet.py --split train --out distillation/results/phase2_train_packets`
  builds the packets.

**One difference from the dev packets:** attribution is null, because no
gate-3a artefact exists for the training split. `build_prompt` never reads
attribution, so the prompts take the same form.

**A measured caveat.** The staging model was trained on these nights, so its
probabilities are sharper there than on unseen subjects. Its median
mean-entropy per night is 0.443 nats on the training nights (range
0.211–0.882), against 0.523 on validation. Night confidence is computed from
that entropy, so the tier mix leans high:

| nights | high | medium | low | total | cohorts |
|---|---:|---:|---:|---:|---|
| training | 78 (57%) | 24 (18%) | 35 (26%) | 137 | SC 107, ST 30 |
| dev | 11 (35%) | 10 (32%) | 10 (32%) | 31 | SC 23, ST 8 |

**Measured on the 137 built packets:**

- **Same structure as dev.** Every packet carries the dev packets' 19
  evidence ids and safe-to-assert flags exactly. The N1 warning is `low` on
  all 137, as it is on dev.
- **The oracle works on all of them.** Its claim set verifies and renders on
  all 137 nights, at about 2,800 characters each (dev: 2,807). Option A's
  targets are therefore usable as they stand.

The low and medium nights are what teach the review flag and the tier keys.

### 3.2 Targets: decided, option A

| option | what | cost | verified clean |
|---|---|---|---|
| **A. Oracle witness** | The oracle's maximal verified claim set for each training packet | Free and instant, and deterministic | Yes, by construction |
| B. Reference model | Gemini 3.8 Flash under P1, on each training packet | About four weeks at the free tier's ~5 successful responses a day | Only after verification; failures would need handling |
| C. A, checked by B | A for all 137, and B on a sample of 20 nights, to show the two agree | About four days of quota | A yes; B checked |

**Recommendation: A.**

- **The reference adds no content.** Its 8 responses so far recover the
  oracle's content exactly (oracle recovery 1.0 on every night).
- **B's cost buys nothing.** It would spend four weeks of quota to produce
  what A produces deterministically.
- **C is worth having only as evidence** that the reference and the oracle
  agree beyond the dev nights. It is not needed as training data.

**Decided on 14 September 2026: option A.**

**Target format:** each witness is serialised in the grammar's field order:
`claim_id`, `claim_type`, `cites`, `subject`, then the type's own fields. The
claims keep the oracle's order and claim ids.

- **Why the order matters.** The oracle emits its fields in another order
  (`claim_id`, `subject`, `cites`, `claim_type`), and `claims.gbnf` rejects
  that. A student trained on the oracle's raw text would learn a form the
  grammar forbids at inference.
- **How every target is checked:**
  - the frozen grammar matcher (`report/gbnf.py`) accepts it;
  - it passes the verifier and renders;
  - it parses back to the oracle's claims.
- **The matcher itself is cross-checked** against llama.cpp. It accepts
  every grammar-constrained output from the five-model run that ended
  normally.

### 3.3 Tier balance

**No rebalancing.** This was fixed on 14 September, after the tier mix was
measured and before any training.

- **Every tier is well represented.** It has at least 24 nights, and three
  epochs give the student each tier's key and flag pattern many times over.
- **Where each tier differs** is only the tier key, and the review flag on
  low nights.
- **A medium-tier weakness would still show,** because per-tier dev results
  are reported.

**The early-stopping hold-out:**

- 7 of the 69 training subjects;
- drawn with seed 0, and redrawn with seed 1, 2 and so on until the held-out
  nights include all three tiers;
- the seed used is recorded with the training run.

## 4. Students and method

**Students.**

- **Qwen2.5-1.5B-Instruct** is the lead.
- **Llama-3.2-3B-Instruct** is trained alongside it as the capacity
  comparison.

The reasons are in `report/PHASE2_STATUS.md`, "What we will use".

**Method.** Supervised fine-tuning with LoRA.

- **Input:** the chat-templated P1 prompt, exactly as `candidates.run` builds
  it.
- **Target:** the JSON array. The loss is computed on target tokens only.

**Fixed before training:**

| setting | value |
|---|---|
| LoRA | rank 16, alpha 32, dropout 0.05, on all attention and MLP projections |
| optimiser | learning rate 2e-4, cosine schedule, 3 epochs, effective batch 8, seed 0 |
| maximum sequence | 3,072 tokens |

The sequence limit covers a prompt of about 1,100 tokens, a witness of about
805, and the template overhead.

**Training-time validation.** Seven of the 69 training subjects (about 14
nights) are held out of training, drawn as section 3.3 fixes. They are used
for loss and early stopping.
Dev is not touched during training.

**Dev evaluations are limited.** Each student gets at most two: the
configuration above, and one declared fallback (5 epochs, otherwise
unchanged). That is the only tuning on dev.

**How the student is evaluated:**

1. Merge the LoRA weights.
2. Convert to GGUF, and quantise to Q4_K_M with the same llama.cpp build
   (b10927).
3. Run exactly the P1 harness (`python -m candidates.run`): the same prompt,
   template, grammar, temperature 0, seed 0, 3,000-token cap, context 12,288,
   and one generation per packet.

The distilled model is then measured on exactly the same footing as the five
candidates.

### 4.1 Where to train: decided, Kaggle

This host cannot realistically train the students.

- **Its GPU** is an RTX 3050 Ti Laptop with 4 GB of memory.
- **Its PyTorch** is a CPU-only build.
- **Qwen 1.5B** might just fit with 4-bit QLoRA and gradient checkpointing,
  but only after installing a CUDA build of PyTorch.
- **Llama 3B** would not fit.

**Decided on 14 September 2026: Kaggle's GPUs** (16 GB). The project already trained its
staging students there, and `evaluate_student.py` reproduces the validation
figures Kaggle reported. Conversion, quantisation and evaluation then run
locally, with the existing harness.

### 4.2 The training set

`python -m student.build_set` writes `student/trainset/train.jsonl`,
`valid.jsonl` and `manifest.json`, about 1 MB in all. The manifest carries
each file's sha256.

| split | nights | subjects | high | medium | low |
|---|---:|---:|---:|---:|---:|
| train | 123 | 62 | 70 | 21 | 32 |
| valid (held out) | 14 | 7 | 8 | 3 | 3 |

- **The held-out subjects** are SC407, SC451, SC476, SC480, SC482, ST706 and
  ST712. The seed-0 draw already covered all three tiers.
- **Each record** is a user message (the exact P1 prompt) and an assistant
  message (the target), with the night's tier and prompt hash.
- **Lengths:** the longest prompt and target come to 1,077 + 973 tokens for
  Qwen, and 1,044 + 933 for Llama. These were measured with `llama-tokenize`,
  excluding the template, and sit well inside the 3,072 limit.
- **A requirement for training: template parity.** The trainer must render
  each chat template exactly as llama.cpp `--jinja` does at evaluation.
  - Qwen2.5 inserts its default system message.
  - Llama 3.2 writes a date into its system header, so the same date must be
    pinned for training and evaluation.
  - Check by rendering one prompt both ways before any training.

## 5. Success criteria on dev

These are fixed now, before any student exists.

**Primary: complete, clean reports.** A night counts when its report passes
verification and covers all five mandatory facts. That means it renders with
full mandatory coverage.

**The bar is 25 of 31,** the same bar the 2F rule sets for the reference
model.

**Guard: numeric fidelity must stay at 1.000.** A student that recomputes
numbers fails, whatever else it does.

**Reported alongside, overall and per tier:**

- per-rule violation counts;
- mandatory and discretionary coverage;
- oracle recovery;
- `hedged_value` as two numbers;
- `hit_token_limit`;
- renders.

**How each result will be read:**

| complete, clean reports on dev | reading |
|---|---|
| 25 or more of 31 | The student meets the contract. It goes to the test set. |
| 12 to 24 | Partial. Report it, and use the one declared fallback. |
| fewer than 12 | The student does not learn the contract at this size. Report it and stop. Llama's result shows whether capacity is the limit. |

**If both students meet the bar,** the one with the smaller deployment
footprint goes to the test set: Qwen.

## 6. The test set

The test set is used once, after everything is frozen:

- the chosen student and the sha256 of its GGUF;
- the P1 prompt hash;
- the harness settings.

The 29 test packets then go through the same harness, with one generation
each.

**The reference model is not run on test** unless that is decided separately,
because it would spend quota.

## 7. What distillation will show, and what it will not

**The oracle can already write this report.** It produces a maximal verified
claim set for any packet, with no language model involved. So in this design
the model contributes no content that the deterministic tier lacks. The
write-up has to say so plainly.

**What distillation tests** is whether a small on-device model can learn a
verified reporting contract. That is the precondition for any richer use in
which no oracle exists.

- **If Qwen meets the bar:** a 1.5B model, in a Pi 5-class memory envelope,
  produces complete reports that pass verification, with the verifier
  guaranteeing every claim.
- **If it does not:** the gap between the oracle and a small model,
  measured rule by rule, is the finding.

## 8. Guardrails

- **Dev packets are used for evaluation only.** The test packets stay
  untouched, with md5 `050fffe46d035008d643435ee826dd92`.
- **The training nights are disjoint by subject.** A test asserts that no
  training packet's recording or subject appears in dev or test.
- **The frozen tier is unchanged.** Every student is evaluated through it.
- **No reference-model requests** are sent without explicit instruction.

## 9. Steps

| step | status | gate |
|---|---|---|
| 1. Build the training packets | **done**: 137 built, 0 skipped; test md5 unchanged | none: local, touches no dev or test file |
| 2. Choose the targets (section 3.2) | **decided**: A, the oracle's claim sets | none |
| 3. Choose where to train (section 4.1) | **decided**: Kaggle | none |
| 4. Build the training set (prompt and target pairs), with tests | **done**: `student/trainset`, 123 training and 14 validation nights; 8 tests | steps 2 and 3 |
| 5. Train Qwen, then Llama | not started | the 2F rule and 2G |
| 6. Convert, quantise, and evaluate on dev with the P1 harness | not started | step 5 |
| 7. Evaluate once on the test set | not started | every choice frozen |
