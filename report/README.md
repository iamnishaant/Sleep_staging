# Report tier — Phase 1: schema, grammar, verifier, renderer

**No language model is involved in this phase.** Every component is testable,
and tested, with no model running and no network reachable.

```
raw JSON ──► Layer 1 (structure, no packet) ──► Layer 2 (policy, needs packet)
                                                     │
                                                     ├─► enrichment ─► renderer
                                                     └─► coverage
```

The governing rule, above every specific rule below: **the verifier never infers
missing evidence.** If the packet does not explicitly contain what a claim
needs, the claim fails. No interpolation, no rounding allowance, no unit
conversion, no benefit of the doubt.

The corollary that shapes the design: the safest rules make a bad claim
**inexpressible** rather than rejected. Three capabilities were removed rather
than policed — the `comparison` type, packet-object names in `cites`, and
threshold-based observation keys.

---

## The claim object

```json
{"claim_id": "c17", "claim_type": "hedged_value", "cites": ["arch.rem_latency"],
 "subject": "this_recording", "value": 119.0, "unit": "minutes"}
```

Five types, closed whitelist. An unrecognised type is a hard reject.

| type | cites | requires | permitted when |
|---|---|---|---|
| `value` | 1 | `value`, `unit` | cited item `safe_to_assert == true` and numeric |
| `hedged_value` | 1 | `value`, `unit` | cited item `safe_to_assert == false` |
| `observation` | ≥1 | `text_key` | the key's predicate holds |
| `review_flag` | 1 | `reason_key` | cited item is a tier item |
| `population_association` | ≥1 | `subject == "population"` | cited item is `associative_only` |

**Forbidden as model-supplied fields** — the verifier derives every one:
`evidence_status`, `safe_to_assert`, `metric_reliability`, `model_reliability`,
`carries_error_bound`, `mean_abs_error`, `error_unit`, `error_measured_on`,
`caveat`, `language_mode`, `confidence`, `label`, `assertion_level`.

**`cites` accepts only the 19 evidence-item ids.** `attribution`,
`attribution_quality`, `probabilities`, `provenance`, `per_stage`, `decoding`
and `n1_confidence_flag` are not citeable by any type. This is the safety
boundary: an attribution-based claim is not rejected, it cannot be constructed.

### Keys are predicates, not labels

| key | kind | requires | predicate |
|---|---|---|---|
| `tier_is_low` | text | `night.confidence` | `night_confidence.tier == "low"` |
| `n1_reliability_is_low` | text | `model.n1_reliability_warning` | item value `== "low"` |
| `low_night_confidence` | reason | `night.confidence` | `night_confidence.tier == "low"` |
| `n1_low_reliability` | reason | `model.n1_reliability_warning` | item value `== "low"` |

Every predicate is an **exact equality against a packet field**. No thresholds,
no clinical norms. "N1 is abnormally low" needs a population norm the verifier
cannot source, and an unsourced cut-off inside the verifier would undercut the
one thing this system claims — that its evidence contract is empirically
derived. Such keys can be added later with a citable reference attached.

---

## Violation codes

### Layer 1 — structural, packet-independent

| code | meaning |
|---|---|
| `L1.malformed_json` | output is not JSON |
| `L1.not_an_array` | parsed, but not an array |
| `L1.not_an_object` | an element is not an object |
| `L1.missing_base_field` | `claim_id` / `claim_type` / `cites` / `subject` absent |
| `L1.unknown_claim_type` | type outside the five-item whitelist |
| `L1.missing_required_field` | a field the type requires |
| `L1.unknown_field` | a field the type does not have |
| `L1.forbidden_derived_field` | the model supplied a verifier-derived field |
| `L1.duplicate_claim_id` | cross-claim state; a grammar cannot catch this |
| `L1.bad_field_type` | wrong type, or `claim_id` not `c<digits>` |
| `L1.bad_cites_arity` | wrong number of citations for the type |
| `L1.unknown_evidence_id` | a name the schema does not have |
| `L1.unknown_text_key` / `L1.unknown_reason_key` | key outside the enum |
| `L1.unknown_subject` | subject outside the enum |

### Layer 2 — policy, packet-dependent

| # | code | rule |
|---|---|---|
| 1 | `L2.cited_id_not_in_packet` | cited id exists in **this** packet |
| 2 | `L2.value_mismatch` | value equals the packet value **exactly** |
| 3 | `L2.unit_mismatch` | unit string matches exactly |
| 4 | `L2.unsafe_item_not_hedged` | `safe_to_assert == false` ⇒ must be `hedged_value` |
| 5 | `L2.safe_item_hedged` | `safe_to_assert == true` ⇒ must not hedge |
| 6 | `L2.stage_tier_unavailable` | a `stage.*` claim must be renderable with its tier |
| 7 | `L2.text_key_dependency_missing` / `L2.text_key_predicate_false` | key's evidence cited, and predicate true |
| 8 | `L2.rem_latency_double_count` | the two REM latencies are not two findings |
| 9 | `L2.rem_error_differenced` | their errors use different references |
| 10 | `L2.missing_review_flag` | a low-confidence night must carry a flag |
| 11 | `L2.uncited_quantity` | no factual quantity without cited evidence |
| 12 | `L2.not_associative_evidence` | `population_association` needs `associative_only` |
| 13 | `L2.bad_subject_for_type` | subject matches the type |
| — | `L2.tier_item_not_valuable` | forced by the packet's shape; see D1 below |

**Rule 2 has no tolerance of any kind.** No epsilon, no rounding, no
significant-figure allowance, no unit conversion. Both sides come from the same
JSON parser, so `==` is exact; an epsilon would hide exactly what this rule
measures, which is a model recomputing instead of transcribing.

---

## Coverage

```
reportable_set = {safe_to_assert == true}
               ∪ {safe_to_assert == false AND (caveat or mean_abs_error present)}

verified_evidence_coverage = |ids cited by VERIFIED claims ∩ reportable_set|
                             / |reportable_set|
```

The denominator is fixed by the packet — 19 on every current packet — never by
anything the model chose. **An empty claim set scores 0.0 while passing every
safety rule**, which is the point of the metric: without that case, "100%
verifier pass rate" is trivially gamed by emitting nothing.

One consequence worth knowing before reading a coverage number: on a
non-low-confidence night, `night.confidence` is **not coverable**, because the
only keys that reference it have a `tier == "low"` predicate. Maximum coverage
is therefore 18/19 on high and medium nights and 19/19 on low nights. That is a
real ceiling, not a bug, and it is the strongest argument for adding
`tier_is_high` / `tier_is_medium` keys in Phase 2.

---

## Where the packet contradicted the build spec

Read from the packets, which are authoritative.

**D1 — two evidence items carry a string, not a number.**
`night.confidence` and `model.n1_reliability_warning` have `value` `"high"` /
`"low"`, `unit: "tier"`, and **no `metric_reliability` field**. The spec said
Layer 1 should require `value` to be a number; that would make the only two
tier-valued items — both `safe_to_assert: true` — unclaimable. Resolved by
restricting `value` / `hedged_value` to the 17 numeric items and reaching the
tier items through `review_flag` and `observation`, which is what those types
are for. New code `L2.tier_item_not_valuable`; the grammar makes it
ungeneratable per type as well.

**D2 — five claim types, not six.** The spec's Layer 1 rule 3 says "the
six-item whitelist" while the table lists five and `comparison` is explicitly
removed. Implemented as five; `tests/test_schema.py` asserts `comparison` is
absent and adversarial case 16 asserts it rejects as an unknown type.

**D3 — the stage reliability tier lives on the evidence item.** Rule 6 said to
implement wherever the packet puts it and say which. It is on the evidence item
as `model_reliability`, and also in `per_stage.<S>.model_reliability_tier`.
This reads the **evidence item**, because that is the object the claim cites.
A test asserts all three sources (`model.n1_reliability_warning.value`,
`stage.N1.fraction.model_reliability`, `per_stage.N1.model_reliability_tier`)
agree on all 29 packets, so the choice carries no risk.

**D4 — `arch.light_deep_ratio` is an `int` in some packets and a `float` in
others.** `value` therefore accepts both, and `5 == 5.0` is a faithful
transcription. `bool` is rejected explicitly, since `True == 1` in Python.

**D5 — `population_association` is unreachable today.** Every evidence item in
every packet is `assertion_level: "factual"`; no packet has associative
evidence. Implemented and tested anyway, including a synthetic packet where the
path succeeds, so it exists when risk evidence lands.

**D6 — rule 8 needed an operational definition.** "Presented as mutually
corroborating" is not directly checkable, so it is implemented as: **no single
claim may cite both REM-latency ids.** Two separate claims are permitted,
because each then carries its own caveat, and the caveat is what stops the
double reading. Both behaviours are tested.

**D7 — `claim_id` shape is pinned.** The grammar emits `c<digits>`; a Layer 1
that accepted any string would describe a larger space than the grammar can
generate. `CLAIM_ID_RE` closes the gap in both directions.

---

## Files

```
report/claim_schema.py       types, field specs, key predicates, vocabulary
report/gbnf.py               grammar GENERATOR + a matcher for its GBNF subset
report/claims.gbnf           GENERATED - do not edit by hand
report/verify_structure.py   Layer 1
report/verify_policy.py      Layer 2 + enrichment + coverage
report/render.py             deterministic templates
report/violations.py         violation codes
```

`claims.gbnf` is generated from the schema constants, and
`tests/test_grammar_agreement.py` asserts the committed file is byte-identical
to the generated text — grammar and schema cannot drift, because there is one
source. Regenerate with:

```bash
python -c "from report.gbnf import write_grammar; write_grammar()"
```

The grammar is a **union of per-type productions**, so each type's field set,
field order and enums are fixed at the token level: malformed per-type output is
ungeneratable rather than merely rejected. It goes further where a context-free
grammar can — `value` may only cite the 17 numeric ids, `review_flag` only the 2
tier ids. What it cannot do, and Layer 1 therefore must: `claim_id` uniqueness,
and any agreement between two fields of one claim. That gap is asserted
explicitly in the agreement test rather than left implicit.

---

## Tests

```bash
cd tests
python test_schema.py             # vocabulary vs all 29 packets, predicates
python test_grammar_agreement.py  # generated file committed; acceptance agrees
python test_adversarial.py        # one planted violation per rule, by code
python test_render.py             # purity, hedging, tiers, banner, exact text
python test_no_leak.py            # opens no files; messages are packet-derived
```

Every adversarial case asserts its **specific** expected code. Asserting only
that something failed would pass even when the wrong rule fired, which would
make per-rule violation rates meaningless.

Two cases carry more weight than the rest. `cites: ["attribution"]` must fail at
**Layer 1** — if it reaches Layer 2 the boundary was built as a blocklist, and a
blocklist has to anticipate every forbidden name. And the **empty claim set**
must pass every safety rule while scoring 0.0 coverage.

`test_no_leak.py` instruments `open` and asserts the verifier and renderer open
**nothing** — a stronger invariant than a path allowlist. It also checks that
every value in every violation's structured `detail` is traceable to the packet,
the claim, or a schema constant, because in a later phase those messages are fed
back to the model for repair, which makes them an output channel out of the
verifier.
