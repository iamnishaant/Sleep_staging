# Report Tier — Phase 1 Build Record

**Nishant Shah · Team 40 · Project 48 — Explainable Deep Learning for Sleep Disorder Detection**
**Completed: 9 September 2026**
**Status: complete. 93 tests, all passing. No language model, no network, no ground-truth file touched.**

---

## 0. What this phase is, in one paragraph

The staging model produces one **evidence packet** per night (JSON, schema 1.3,
29 packets). Downstream, a small language model will read a packet and propose
**structured claims**; a deterministic verifier will check those claims against
the packet; only verified claims will be rendered into a human-readable report.

Phase 1 built **the deterministic core only** — schema, grammar, verifier,
renderer, and an adversarial test suite. The language model arrives in a later
phase. Everything here is testable, and tested, with nothing running.

The governing rule, above every specific rule:

> **The verifier never infers missing evidence.** If the packet does not
> explicitly contain what a claim needs, the claim fails. No interpolation, no
> rounding allowance, no unit conversion, no benefit of the doubt.

And the corollary that shaped almost every design decision:

> **The safest rules make a bad claim inexpressible rather than rejected.** A
> closed whitelist can be shown complete; a blocklist of forbidden statements
> cannot. Prefer removing a capability over policing it.

---

## 1. What was built

| file | lines | what it is |
|---|---:|---|
| `report/claim_schema.py` | 260 | claim types, field specs, key predicates, the citeable vocabulary |
| `report/violations.py` | 105 | 30 violation codes as an enum, plus the `Violation` object |
| `report/verify_structure.py` | 153 | **Layer 1** — structural, packet-independent |
| `report/verify_policy.py` | 368 | **Layer 2** — policy, enrichment, coverage |
| `report/render.py` | 191 | deterministic templates |
| `report/gbnf.py` | 302 | grammar **generator** + a matcher for the GBNF subset |
| `report/claims.gbnf` | 49 | **generated** — do not edit by hand |
| `report/__init__.py` | 78 | the `verify_report` pipeline entry point |
| `report/README.md` | 244 | the reference: rules, codes, coverage definition |
| `tests/_packets.py` | 72 | packet loading + the shared valid-claim-set builder |
| `tests/test_schema.py` | 120 | 16 tests |
| `tests/test_grammar_agreement.py` | 209 | 19 tests |
| `tests/test_adversarial.py` | 349 | 36 tests |
| `tests/test_render.py` | 176 | 16 tests |
| `tests/test_no_leak.py` | 166 | 6 tests |

**93 distinct tests.** 2,549 lines of Python, of which 1,092 (43%) are tests —
which is about the right ratio for a component whose entire purpose is refusing
things.

---

## 2. The first thing done: read the packets

The build spec described the packet accurately but not exhaustively, and said
the file on disk is authoritative. Reading all 29 first turned out to matter —
**seven** things differed from the description, three of which changed the
design (§8). Two examples of what only reading finds:

- `night.confidence` and `model.n1_reliability_warning` hold **strings**
  (`"high"` / `"low"`) with `unit: "tier"`, and carry **no**
  `metric_reliability` field.
- `arch.light_deep_ratio` is an `int` in some packets and a `float` in others.

An invariant survey across all 29 packets established what the schema could
safely hard-code: identical 19 evidence ids in identical order, exactly 5
`safe_to_assert` items, every item `assertion_level: "factual"`, night-confidence
tiers split 12 high / 4 medium / 13 low. Every one of those is now a test, so
if a future packet breaks the assumption the suite says so rather than the
verifier silently mis-handling it.

---

## 3. The claim schema

A claim contains **only what the model must decide**:

```json
{"claim_id": "c17", "claim_type": "hedged_value", "cites": ["arch.rem_latency"],
 "subject": "this_recording", "value": 119.0, "unit": "minutes"}
```

**Five claim types**, closed whitelist: `value`, `hedged_value`, `observation`,
`review_flag`, `population_association`. An unrecognised type is a hard reject.

**13 forbidden fields** the model may never supply, because the verifier derives
every one: `safe_to_assert`, `metric_reliability`, `model_reliability`,
`mean_abs_error`, `caveat`, `confidence`, `assertion_level`, and so on. A model
that writes `safe_to_assert` is asserting something about its own
trustworthiness, which is precisely the move this design removes.

**`cites` accepts only the 19 evidence-item ids.** `attribution`,
`attribution_quality`, `probabilities`, `provenance`, `per_stage`, `decoding`
and `n1_confidence_flag` are not citeable by any type. This is the safety
boundary, and it is a *boundary* rather than a rule: an attribution-based claim
is not rejected, it cannot be constructed.

### Keys are predicates, not labels

Without this, `observation` would be the loophole the rest of the design closes
— a model could cite `stage.N1.fraction` with a key meaning "N1 is low" while N1
is high, and nothing would catch it. Every key declares required evidence **and**
a predicate the packet must satisfy:

| key | kind | requires | predicate |
|---|---|---|---|
| `tier_is_low` | text | `night.confidence` | `night_confidence.tier == "low"` |
| `n1_reliability_is_low` | text | `model.n1_reliability_warning` | item value `== "low"` |
| `low_night_confidence` | reason | `night.confidence` | `night_confidence.tier == "low"` |
| `n1_low_reliability` | reason | `model.n1_reliability_warning` | item value `== "low"` |

Every predicate is an **exact equality against a packet field**. No thresholds,
no clinical norms, no invented cut-offs. "N1 is abnormally low" needs a
population norm the verifier cannot source, and an unsourced cut-off inside the
verifier would undercut the one thing this system claims — that its evidence
contract is empirically derived.

---

## 4. The grammar

`report/claims.gbnf` constrains llama.cpp generation. Two decisions.

**It is generated from the schema, not hand-written.** Grammar and schema
drifting apart is a silent failure mode: generation would stay inside a space
the verifier rejects, or — worse — the grammar would permit a shape Layer 1 was
never written to see. `report/gbnf.py` builds the grammar from the same
constants the verifier uses, and a test asserts the committed file is
**byte-identical** to the generated text. They cannot disagree, because there is
one source.

**It is a union of per-type productions**, not one generic claim object:

```
claim ::= value-claim | hedged-claim | observation-claim | review-claim | assoc-claim
```

GBNF is context-free, so each type's exact field set, field order and enum
values are expressible at the token level. That makes malformed per-type output
**ungeneratable** rather than merely rejected — the closed-world principle
applied to decoding.

It goes further than the spec asked, where a context-free grammar can:

- `value` and `hedged_value` may cite only the **17 numeric** ids
- `review_flag` may cite only the **2 tier** ids

So a model physically cannot emit a `value` claim about `night.confidence`.

**What the grammar cannot do, and Layer 1 therefore must:** `claim_id`
uniqueness, and any agreement between two fields of the same claim. Both need
state. That gap is asserted explicitly in the agreement test rather than left
implicit.

### Testing agreement needed a recogniser

"The grammar accepts exactly the syntactically valid claim space" is a claim
*about a grammar*, and checking it needs something that can run one. So
`gbnf.py` also implements a matcher for the GBNF subset used here — literals,
alternation, sequence, grouping, `?`/`*`/`+`, character classes — and the test
enumerates the claim space, comparing grammar acceptance against Layer 1
acceptance claim by claim.

Two limits on "exactly", both inherent to context-free grammars and both stated
in the test rather than glossed:

- **Field order.** The grammar fixes a canonical order; Layer 1 is order-blind.
  A model decoding under the grammar always emits canonical order, so the
  comparison is made on canonically-ordered serialisations.
- **Cross-claim state.** The grammar accepts a duplicate `claim_id`; Layer 1
  rejects it.

---

## 5. Layer 1 — structure (15 codes)

Packet-independent. Closed-world throughout: unknown claim type, unknown field,
unknown evidence id, unknown enum value are all hard rejects. The point is not
catching a model behaving badly — it is that a claim outside the schema has no
meaning the renderer could give it, so there is nothing to salvage.

`L1.malformed_json` · `L1.not_an_array` · `L1.not_an_object` ·
`L1.missing_base_field` · `L1.unknown_claim_type` · `L1.missing_required_field` ·
`L1.unknown_field` · `L1.forbidden_derived_field` · `L1.duplicate_claim_id` ·
`L1.bad_field_type` · `L1.bad_cites_arity` · `L1.unknown_evidence_id` ·
`L1.unknown_text_key` · `L1.unknown_reason_key` · `L1.unknown_subject`

**Why the evidence-id check lives here.** `cites: ["attribution"]` must fail as
a name the schema does not have, not as a policy the verifier enforces. If it
reached Layer 2, the safety boundary would have been implemented as a blocklist
— and a blocklist has to anticipate every forbidden name, while a closed schema
does not. Layer 1 is packet-independent, so this required making the 19 ids a
**schema constant**; all 29 packets carry exactly those ids, asserted by test.
Layer 2 still checks membership in *this* packet.

---

## 6. Layer 2 — policy (15 codes, 14 rule functions)

Packet-dependent. Every rule is its own function with its own code, so an
evaluation can report violation rate **per rule** rather than pooled. Nothing
uses a model, an embedding, or a similarity score: every check is exact lookup,
set membership, or numeric equality.

| # | code | rule |
|---|---|---|
| 1 | `cited_id_not_in_packet` | cited id exists in **this** packet |
| 2 | `value_mismatch` | value equals the packet value **exactly** |
| 3 | `unit_mismatch` | unit string matches exactly |
| 4 | `unsafe_item_not_hedged` | `safe_to_assert == false` ⇒ must be `hedged_value` |
| 5 | `safe_item_hedged` | `safe_to_assert == true` ⇒ must not hedge |
| 6 | `stage_tier_unavailable` | a `stage.*` claim must be renderable with its tier |
| 7 | `text_key_dependency_missing` / `text_key_predicate_false` | key's evidence cited, and predicate true |
| 8 | `rem_latency_double_count` | the two REM latencies are not two findings |
| 9 | `rem_error_differenced` | their errors use different references |
| 10 | `missing_review_flag` | a low-confidence night must carry a flag |
| 11 | `uncited_quantity` | no factual quantity without cited evidence |
| 12 | `not_associative_evidence` | `population_association` needs `associative_only` |
| 13 | `bad_subject_for_type` | subject matches the type |
| — | `tier_item_not_valuable` | forced by the packet's shape (deviation D1) |

**Rule 2 has no tolerance of any kind.** No epsilon, no rounding, no
significant-figure allowance, no unit conversion. Both sides come from the same
JSON parser, so `==` is exact; an epsilon would hide exactly what this rule
measures — a model recomputing instead of transcribing.

**Enrichment** happens here, not in the model and not in the renderer. A
verified claim is paired with the packet fields the verifier looked up, which is
what lets a hedged value carry its measured error and its caveat even though the
model never saw either. A caveat cannot be softened or dropped, because the
model was never holding it.

---

## 7. Coverage

```
reportable_set = {safe_to_assert == true}
               ∪ {safe_to_assert == false AND (caveat or mean_abs_error present)}

verified_evidence_coverage = |ids cited by VERIFIED claims ∩ reportable_set|
                             / |reportable_set|
```

The denominator is fixed by the packet — 19 on every current packet — never by
anything the model chose. **An empty claim set scores 0.0 while passing every
safety rule**, which is the point: without that, "100% verifier pass rate" is
trivially gamed by emitting nothing.

**One ceiling to know before reading a coverage number.** On a
non-low-confidence night, `night.confidence` is **not coverable** — the only
keys referencing it have a `tier == "low"` predicate. Maximum coverage is
therefore **18/19 on high and medium nights, 19/19 on low nights**. That is a
real ceiling, not a bug, and it is the strongest argument for adding
`tier_is_high` / `tier_is_medium` keys in Phase 2.

---

## 8. Where the packet contradicted the build spec

Seven deviations. The spec said to follow the packet and report what differed.

**D1 — two evidence items carry a string, not a number.** `night.confidence`
and `model.n1_reliability_warning` have `value` `"high"`/`"low"`,
`unit: "tier"`, and no `metric_reliability`. The spec's Layer 1 rule required
`value` to be a number, which would make the only two tier-valued items — both
`safe_to_assert: true` — unclaimable. **Resolved** by restricting
`value`/`hedged_value` to the 17 numeric items and reaching tier items through
`review_flag` and `observation`, which is what those types are for. New code
`L2.tier_item_not_valuable`; the grammar makes it ungeneratable per type too.

**D2 — five claim types, not six.** The spec's Layer 1 rule 3 says "the
six-item whitelist" while its own table lists five and `comparison` is
explicitly removed. Implemented as **five**; a test asserts `comparison` is
absent, and adversarial case 16 asserts it rejects as an unknown type.

**D3 — the stage reliability tier lives on the evidence item.** Rule 6 said to
implement wherever the packet puts it and say which. It is on the evidence item
as `model_reliability`, and also in `per_stage.<S>.model_reliability_tier`. This
reads the **evidence item**, because that is the object the claim cites. A test
asserts all three sources agree on all 29 packets, so the choice carries no risk.

**D4 — `arch.light_deep_ratio` is `int` in some packets, `float` in others.**
`value` accepts both, and `5 == 5.0` is a faithful transcription. `bool` is
rejected explicitly, since `True == 1` in Python.

**D5 — `population_association` is unreachable today.** Every evidence item in
every packet is `assertion_level: "factual"`. Implemented and tested anyway,
including a synthetic packet where the path succeeds, so it exists when risk
evidence lands.

**D6 — rule 8 needed an operational definition.** "Presented as mutually
corroborating" is not directly checkable, so it is implemented as: **no single
claim may cite both REM-latency ids.** Two separate claims are permitted,
because each then carries its own caveat, and the caveat is what stops the
double reading. Both behaviours are tested.

**D7 — `claim_id` shape had to be pinned.** The grammar emits `c<digits>`; a
Layer 1 accepting any string would describe a larger space than the grammar can
generate. `CLAIM_ID_RE` closes the gap in both directions.

---

## 9. The test suite

### `test_schema.py` — 16 tests
The schema hard-codes a 19-id vocabulary so Layer 1 can reject unknown ids
without a packet. That is only safe if the vocabulary is what every packet
actually contains, so it is checked against all 29. Also: exactly five claim
types, `comparison` absent, no type exposes a forbidden field, every key
declares evidence and a predicate, the three N1-tier sources agree, the
coverage denominator is 19 everywhere.

### `test_grammar_agreement.py` — 19 tests
The committed `.gbnf` is byte-identical to the generated text. Then the claim
space is enumerated and grammar acceptance is compared against Layer 1
acceptance, both directions. Includes the explicit assertion that duplicate
`claim_id`s are a **Layer 1-only** catch.

### `test_adversarial.py` — 36 tests
All 22 required cases plus 14 more. Every case asserts its **specific** expected
code — asserting only that something failed would pass even when the wrong rule
fired, which would make per-rule violation rates meaningless.

Two carry more weight than the rest:

- **`cites: ["attribution"]` must fail at Layer 1.** The test asserts not just
  the code but that *no policy rule fired at all*. If a policy rule fires, the
  boundary was built as a blocklist.
- **The empty claim set** passes every safety rule at 0.0 coverage.

Also here: a **positive suite** that builds a valid claim set from each packet's
own values and asserts it passes cleanly on all 29.

### `test_render.py` — 16 tests
Purity (rendered twice, compared byte-for-byte), "a hedged value is never a bare
number" as an invariant over **every hedgeable item in every packet**, stage
claims always carrying their tier, the low-confidence banner, the cohort-scope
footer, and exact-text pins so a refactor cannot quietly reword a clinical
report.

### `test_no_leak.py` — 6 tests
Two channels, both real.

- **Files.** `open` is instrumented and asserted to stay shut for the entire
  run. "Opened nothing" is a stronger and simpler invariant than "opened only
  approved things".
- **Messages.** In a later phase, violation messages are fed back to the model
  for repair — which makes every message an output channel *out of* the
  verifier. Every value in every violation's structured `detail` must be
  traceable to the packet, the claim, or a closed schema constant.

This is why `Violation` carries a structured `detail` dict and derives its
`message` from it, rather than taking free-hand prose: the test can check the
structured form instead of parsing English.

---

## 10. Two defects found by running the thing

Worth recording, because neither was found by reading code.

**The grammar-agreement test passed for the wrong reason.** Its `canonical()`
serialiser only emitted fields in a known list, so the extra-field case silently
*dropped* the extra field and fed the grammar a perfectly valid claim. Fixed to
append unknown fields; the test then failed correctly, and the grammar was
confirmed to reject them.

**The renderer produced broken English.** Packet caveats are grammatically
heterogeneous — verb phrases (`"depends on a single epoch"`), noun phrases
(`"mean relative error 85% on validation."`), one full sentence. The template
`"This figure " + caveat` produced *"This figure mean relative error 85% on
validation."* Only a labelled `Caveat:` clause composes with all three forms.
Now tested over every caveat in every packet.

---

## 11. What Phase 1 deliberately does not do

- **No language model.** Not a limitation — the point. Every component is
  testable without one, and is tested without one.
- **No `comparison` claim type.** Two items sharing a unit can still be
  semantically incomparable (`sleep_efficiency` and `stage.N1.fraction` are both
  fractions over different denominators). A comparison-pair whitelist would be
  hand-maintained justification for a type only ~5 items per packet could
  populate. Removed rather than policed; adding it later is additive.
- **No threshold-based observations.** Phase 1 predicates are exact equality
  only. Anything needing a population norm waits for a citable reference.
- **No attribution claims.** `attribution` and `attribution_quality` are not
  citeable. The gate-3a verdict still reaches the report — as a fixed footer
  rendered straight from the packet, stating its **cohort** scope explicitly —
  but no model can phrase it and no claim can reference it.
- **No repair loop.** Violation messages are built to be fed back safely; the
  loop itself is a later phase.

---

## 12. Open items for Phase 2

1. **`tier_is_high` / `tier_is_medium` keys.** Both are exact-equality
   predicates, both additive, and together they lift the coverage ceiling from
   18/19 to 19/19 on high and medium nights.
2. **The model itself** — 1–4B under llama.cpp, decoding against
   `claims.gbnf`, with the repair loop fed by violation messages.
3. **An evaluator** reporting violation rate *per rule* and coverage per packet.
   The per-rule codes exist precisely so this is possible.
4. **`risk` is still null** in every packet. `population_association` is built
   and tested but unreachable until the disorder pipeline is integrated.
5. **A comparison type**, if it earns its place, with an explicit
   comparison-pair whitelist rather than a shared-unit heuristic.

---

## 13. Running it

```bash
cd tests
python test_schema.py             # 16
python test_grammar_agreement.py  # 19
python test_adversarial.py        # 36
python test_render.py             # 16
python test_no_leak.py            #  6
```

Regenerate the grammar after any schema change:

```bash
python -c "from report.gbnf import write_grammar; write_grammar()"
```

End-to-end on one night:

```python
import json
from report import verify_report
from report.render import render_report

packet = json.load(open("distillation/results/packets/SC4202E0-PSG.json"))
result = verify_report(model_output_json, packet)
if result.ok:
    print(render_report(result.enriched, packet))
print(result.coverage, result.codes)
```

---

## 14. Acceptance criteria

| # | criterion | status |
|---|---|---|
| 1 | every adversarial case fails with its **specific** code | met |
| 2 | positive cases pass and render identically across runs | met |
| 3 | empty claim set passes safety, scores 0.0 coverage | met |
| 4 | grammar accepts exactly the syntactically valid space | met, with the two stated limits (§4) |
| 5 | runs on all 29 packets without error | met |
| 6 | no model, no network, no ground-truth file touched | met — stdlib only, `open` asserted shut |

`report/` imports nothing but `json`, `re`, `enum`, `pathlib` and `typing`.

---

*Reference documentation: `report/README.md` — the rule list, violation codes,
and coverage definition. Packet contract: `distillation/EVIDENCE_PACKET.md`.*
