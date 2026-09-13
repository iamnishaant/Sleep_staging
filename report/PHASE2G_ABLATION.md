# Phase 2G grammar ablation: design

**Written 14 September 2026, before any unconstrained generation exists.
Nothing in this document has been run.** It fixes the arms, the scoring and
the reading of each outcome in advance, the same way the 2F decision rule
was fixed before the reference run.

---

## 1. The question

The local candidates generate under `report/claims.gbnf`. The ablation asks
what that grammar buys on this contract, and what it costs.

1. **Structure.** How much of Layer 1 validity comes from the grammar rather
   than from the model?
2. **Content.** Where both arms produce valid structure, does constrained
   decoding change what the model says: mandatory coverage, hedging, policy
   violations? Constraining can force tokens the model would not have chosen,
   so the effect could run either way.
3. **Termination.** Does the grammar contribute to generations that run to
   the 3,000-token cap? The grammar allows the array to close after any claim
   but never requires it. Under P1, Qwen2.5-1.5B reached the cap on 1 of 31
   nights (SC4171E0). The five-model run logs `hit_token_limit` for every
   generation.

The ablation is not a model selection, and it does not reopen P1.

---

## 2. What the grammar enforces, and what it does not

This is read from `report/claims.gbnf`, which is generated from
`report/claim_schema.py`. `tests/test_grammar_agreement.py` pins the two
together.

**Enforced:**

- The output is one JSON array of claim objects.
- Each claim is one of five per-type productions, with an exact field set and
  field order.
- `claim_id` is `"c"` followed by digits.
- **Cites:**
  - `value` and `hedged_value` cite exactly one of the 17 numeric evidence
    ids;
  - `review_flag` cites exactly one of the 2 tier ids;
  - `observation` and `population_association` cite one or more of all 19.
- `unit` (5 values), `text_key` (4), `reason_key` (2) and `subject` (fixed per
  claim type) are enumerations.
- Numbers match `-?[0-9]+(.[0-9]+)?`.

**Not enforced:**

- That `claim_id` values are unique.
- That a value matches the packet, or that a unit matches the item.
- Hedging policy, coverage, text-key dependencies and tier predicates. That is
  every Layer 2 rule except the subject.
- Any end to the output. The claim list is unbounded, and so is whitespace.

**What this means for the violation codes.** Of the 30 codes, 14 cannot be
generated under the grammar, except by truncation:

- **Layer 1, 13 of 15:** `not_an_array`, `not_an_object`, `missing_base_field`,
  `unknown_claim_type`, `missing_required_field`, `unknown_field`,
  `forbidden_derived_field`, `bad_field_type`, `unknown_evidence_id`,
  `unknown_reason_key`, `unknown_subject`, `unknown_text_key` and
  `bad_cites_arity`.
- **Layer 2, 1 of 15:** `bad_subject_for_type`.

Three more are limited:

- **`L2.cited_id_not_in_packet`** cannot occur on this data either, because
  every packet carries the same 19 ids that the grammar lists.
- **`L1.malformed_json`** occurs in the constrained arm only through
  truncation at the cap.
- **`L1.duplicate_claim_id`** remains possible.

**This gives the ablation a built-in check.** Any of the 14 codes in the
constrained arm means a harness fault, to be investigated before any result
is read. Qwen's P1 run is consistent with the check: its only Layer 1
violation was one truncation.

---

## 3. The arms

| | arm A: constrained | arm B: unconstrained |
|---|---|---|
| grammar | `report/claims.gbnf` | none |
| prompt | P1, byte-identical (same `prompt_hash`) | same |
| chat template | `--jinja -cnv -st` | same |
| decoding | temperature 0, seed 0, `-n 3000`, `-c 12288` | same |
| runtime and models | llama.cpp b10927; the five Q4_K_M files | same |
| packets | the 31 dev packets | same |
| new generations | **none**: the P1 run's cache is arm A | 155: one per packet per model |

**The command.** Arm B's command is arm A's with exactly two arguments
removed: `--grammar-file` and its path. A test asserts that difference in
the argument list.

**The format instruction.** Without the grammar, the prompt still carries
one: `build_prompt` ends *"Output a JSON array of claim objects and nothing
else."* Arm B therefore tests whether the model follows that instruction,
with nothing added to help it.

**The P1 rules stay unchanged:** one generation per packet, no retries, and
no raised cap.

---

## 4. Scoring

**Primary: strict, identical to arm A.**

- Arm B's output goes to the frozen verifier exactly as arm A's does. Only
  llama.cpp's end-of-text display marker is removed.
- A Markdown code fence, or a sentence before or after the array, is
  `L1.malformed_json`.
- This matches the deployed situation: the verifier is the gate, and it does
  not repair output.

**Secondary: one declared normalisation, exploratory.**

- **The rule.** If the whole output, trimmed, is a single Markdown code fence
  with no other fence inside it, the body is scored instead. The fence may
  carry a language tag. Nothing else is repaired.
- **Why.** It separates a formatting habit from an inability to produce the
  claims.
- **Fixed in advance.** The rule is written in code, with tests, before arm B
  runs, and is not adjusted after its outputs are seen.
- **Exploratory.** Like every exploratory figure, it does not enter selection
  decisions.

**Reported per model and per arm, overall and per tier:**

- schema validity;
- per-rule counts;
- mandatory coverage and the count at 5/5;
- discretionary coverage;
- cited mandatory and cited discretionary;
- oracle recovery and unrecovered available;
- numeric fidelity;
- reports rendered;
- `hedged_value` as two numbers: packets using it (of 31) and claims (of 434);
- `hit_token_limit`;
- output tokens.

**Paired, per model.** The same packets appear in both arms, so each model
gets three 2×2 tables: schema-valid, 5/5 mandatory, and renders. Each table
crosses arm A (yes or no) with arm B (yes or no), and the discordant cells
are the result. An exact McNemar p-value may accompany each table as a
description only. With 31 packets and five models, it is not a significance
claim.

**Host figures stay in the run log,** as in the P1 run: wall clock per
packet, and peak RSS with its context length.

---

## 5. How each outcome will be read

This section was written before any arm-B output existed.

| outcome | reading |
|---|---|
| B's strict schema validity is well below A's, and content is similar on packets valid in both arms | The grammar buys syntax, not content. The contract gap is semantic, so effort belongs on the model: distillation, if 2F routes there. |
| B's strict validity is close to A's | The prompt alone carries the format for that model. The grammar is a safeguard rather than a necessity. |
| B beats A on 5/5, hedging or violations, on packets valid in both arms | Constrained decoding distorts content for that model. That is a cost of the grammar, and it is recorded. It does not remove the grammar on its own, because the verifier still requires valid structure. |
| B reaches the token cap markedly less often than A | The open-ended claim list contributes to degenerate repetition. A bounded claim count in the grammar would be a frozen-tier change, so it is recorded as a candidate, not made. |
| B's outputs are mostly prose or fenced | Strict and secondary scores are reported side by side. The gap between them is the size of the formatting habit. |

More than one row can hold at once, and the reading can differ between
models.

**The grammar stays in the pipeline.** The ablation measures its effect.
Changing the grammar would need an explicit decision, because it is part of
the frozen tier.

---

## 6. Gate and guardrails

**It runs only after both of these:**

- all five models' P1 generations are cached and scored;
- the 2F decision rule, applied at 31 reference responses, routes to 2G.

If the rule routes to P2 instead, the ablation waits, and whether it runs
under P1 or P2 is decided then.

**Guardrails:**

- **Dev packets only.** The test packets stay untouched, with md5
  `050fffe46d035008d643435ee826dd92` checked before and after.
- **No change** to P1, the grammar, the frozen tier or the arm-A cache. The
  grammar's sha256 is recorded with every arm-B summary.
- **The reference model is not ablated.** Its structured-output schema is the
  hosted counterpart of the grammar, but ablating it would spend free-tier
  quota. It is out of scope unless asked for.

---

## 7. Harness changes, to be built at 2G

- **`candidates/run.py --arm unconstrained`:**
  - the command is arm A's minus `--grammar-file <path>`;
  - the cache goes under `candidates/cache/{model}/P1-nogrammar/`, and the
    key gains `"grammar": "none"`;
  - arm A's keys and entries are left exactly as they are;
  - the log gains an `arm` field.
- **`candidates/score.py --arm`:** the arm-B table, the secondary
  normalisation, and the paired tables.
- **Tests:**
  - the command differs from arm A's by exactly the two grammar arguments;
  - arm A's cache is never written;
  - the arm-B key differs from arm A's;
  - the normalisation strips one wrapping fence, and leaves prose, two fences
    and unfenced text unchanged.

---

## 8. Cost

Arm B is 155 generations on the evaluation host. The P1 run gives the scale:

- Qwen2.5-1.5B took 588 s for its 31 packets.
- SmolLM2-1.7B, which mostly ran to the cap, took about 190–290 s per packet.

Expect a similar total to the five-model P1 run: somewhere between 3 and 9
hours, depending on how often each model loops. Without the grammar there is
no grammar-sampling overhead, but a model that loops still runs to the
3,000-token cap.
