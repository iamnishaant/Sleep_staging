# Phase 2 — working notes

**Nishant Shah · Team 40 · Project 48**
**Started: 11 September 2026**
**Status: 2A-2E complete — 248 tests, all passing. The deterministic tier is
finished; the next step runs a model. No model has run yet.**

Phase 2 adds the language-model tier. Steps 2A–2C are deterministic and testable
with nothing running; this file records the audit that preceded them, the
decisions taken, and every access to the locked test set.

---

## The test-set access log

The 29 packets in `distillation/results/packets/` are locked for **selection**
decisions — model, prompt, K, grammar. They may be accessed for a documented
**correctness** fix, provided the fix is not chosen because it improves a test
score. The failure mode being avoided is *undisclosed* iteration, not iteration.

| date | reason | changed model? | prompt? | K? | rerun? |
|---|---|---|---|---|---|
| 2026-09-11 | Rebuilt all 29 to add `attribution_quality.evaluated_on_split` and `evaluated_on_n_recordings`. The renderer had `"test split of 29 recordings"` as a **literal**, which is a false sentence on any packet built from another split. Fixing it required the packet to carry its verdict's provenance. Correctness only — no metric was consulted, and no score exists yet to improve. | no | no | no | no |
| 2026-09-12 | Read (not modified) test packets SC4011E0, SC4022E0 and SC4202E0 to pin the register-A rendering, one per tier, as the pre-2F closeout specified. Rendering is deterministic and involves no model. Renderer wording is not a selection variable, and no score was computed. md5 `050fffe46d035008d643435ee826dd92` before and after. | no | no | no | no |

---

## Part 0 — the audit, and what it found

### The finding that mattered

`build_packet.py` hard-coded the split **inside the recording expression**:

```python
recs = sorted(r for s in sp["splits"]["test"] for r in sp["recordings_by_subject"][s])
```

`--splits` chose the *file*; nothing chose the *split*. The dangerous direction
is not the obvious one:

> `--out devdir` today writes **test** packets into a folder labelled dev. You
> get 29 files where you expected 31, in the right place, with the right schema,
> and every subsequent Phase 2 decision is tuned on the locked split. Silent, and
> it looks correct.

The reverse — a dev run clobbering `results/packets/` — at least destroys files
you would notice. The first direction leaves no trace at all.

### Audit answers

| # | question | finding |
|---|---|---|
| 1 | recording selection | derived from `splits.json`; the key `"test"` hard-coded in the expression |
| 2 | subject selection | inherited entirely from `splits.json → splits.test` |
| 3 | split logic | **subject-level**; `granularity: subject`, `subject_key: recording_id[:5]`, stratified by cohort |
| 4 | seed | 42, recorded in the artefact — randomised but seeded and committed, so fixed |
| 5 | input manifest | `splits.json` + per-recording `.npz` from `results/probs_{model}/` |
| 6 | output directory | `--out`, defaulted to `results/packets` |
| 7 | overwrite | **silent, no guard** |
| 8 | ordering | deterministic (`sorted`), not filesystem-dependent |
| 9 | contamination | possible in both directions; the silent one is the worse |
| 10 | separability | `--out` sufficed; two other blockers did not |

### The splits

| | subjects | recordings |
|---|---:|---:|
| test (locked) | 15 | 29 |
| val (dev candidate) | 16 | 31 |

Disjointness **already holds** in the upstream split: subject overlap ∅,
recording overlap ∅, val∩train ∅. `SC413` contributes a single night, which is
why 16 subjects give 31 recordings rather than 32.

Test subjects: `SC401 SC402 SC414 SC420 SC421 SC423 SC429 SC437 SC452 SC453
SC461 SC474 ST710 ST714 ST716`

Val subjects: `SC408 SC411 SC413 SC417 SC432 SC434 SC435 SC444 SC446 SC456
SC458 SC467 ST708 ST713 ST715 ST720`

---

## Changes made in 2A

### B1 — `--split`, required, no default

Both `build_packet.py` and `gate3a_attribution.py` had the same hard-coded
`splits["test"]`. Both now take `--split {train,val,test}` as a **required**
argument with no default, so a forgotten flag errors rather than silently
selecting the split that must not be tuned on.

In `gate3a_attribution.py` the **train** baseline stays train whatever is
attributed. It is the registered reference point, not a property of the split
under test; moving it with `--split` would silently redefine what every
attribution is measured against.

### B2 — the `_val` cache convention

`build_packet.py` was the only tool in the pipeline without it. `calibrate.py`,
`evaluate_student.py`, `fit_n1_flag.py`, `metric_reliability.py` and
`build_ensemble.py` all use `probs_{model}_val`. Closed the same way:
`probs_{model}` for test, `probs_{model}_{split}` otherwise.

### The two-key guard

Writing into `results/packets/` now requires **both** `--split test` and
`--allow-test-overwrite`. `--split` and `--out` are checked for consistency at
startup, before anything is loaded or written.

Verified by running every dangerous invocation and checking the test packets'
md5 was unchanged:

| invocation | outcome |
|---|---|
| `--out /tmp/x` (no `--split`) | argparse error — the old silent default |
| `--split val --out results/packets` | refused: names the contamination |
| `--split test --out results/packets` (one key) | refused: second key required |
| `--split val --out results/packets/sneaky` | refused — subdirectories too |

### A third blocker the audit surfaced

Gate 3a's `per_recording` covered **0 of 31** val recordings, so dev packets
would have shipped `attribution: null` while test packets ship it populated —
and `render_report` emits the gate-3a footer only when
`attribution_quality.status == "run"`. Dev reports would have had no footer at
all. Tuning on packets that render differently from the ones you are graded on
is the exact class of mistake this split discipline exists to prevent, so Gate
3a is being re-run on val.

Three things checked **before** committing 2.5 hours to that run:

1. The val artefact goes to `_g3a_n4kd_val.json`, never into the test artefact.
   A val run yields a **val** cohort verdict, and copying the test verdict into
   dev packets would drop a test-derived aggregate into the artefacts being
   tuned on.
2. `test_render.py`'s footer assertions check `"COHORT, not this recording"` and
   `"gate 3a"` — they do **not** pin the verdict string, so a 4/5 val verdict
   will not break them.
3. A 2-recording smoke run confirmed the val artefact carries
   `grouped_by: "predicted"` — `build_packet.py` exits if it does not, and that
   is better found in 90 seconds than after 2.5 hours.

### A false sentence in the renderer

`render_attribution_footer` had this as a **literal**:

> "This verdict was measured once over the pooled **test split of 29
> recordings**…"

True of the only packets that existed; false of any dev packet, whose verdict
comes from 31 val recordings. The deterministic renderer — the component whose
whole purpose is that no model can phrase a claim — would have stated it as
fact.

Fixed by having the packet carry `attribution_quality.evaluated_on_split` and
`evaluated_on_n_recordings`, and the renderer read them. **No default on
either**: a packet that cannot say which split its verdict came from gets no
sentence claiming to know, rather than a plausible guess. Rebuilding the 29 to
add those fields is the access-log entry above.

---

## Part 1 (2A) — the development packet set

**31 dev packets from the 16 validation subjects, in
`distillation/results/phase2_dev_packets/`.** The 29 test packets were byte-
identical before and after (md5 checked around the build).

### Gate 3a on val, and what it turned out to be worth

1h50m, not the 2.5h estimated — the machine was idle this time.

| | test | val |
|---|---|---|
| verdict | PASS, 3/5 | **PASS, 3/5** |
| met | N3, W, N2 | **N3, W, N2** |
| missed | REM, N1 | **REM, N1** |
| completeness | 0.03% | 0.03% |

Not just the same count — the same three met and the same two missed, on 16
disjoint subjects, against a registration written once before either run. That
is a **held-out confirmation of the registered predictions**, and it moves REM's
`rel_theta` miss and N1's better-than-predicted coherence from properties of one
draw to properties of the model. Written into PROJECT_REPORT §6j, with the
caveat that both runs share one trained model and one registration, so it is
replication across **subjects** — not across models or registrations.

### The invariant survey — every check that holds on 29, run on 31

| invariant | test (29) | dev (31) |
|---|---|---|
| identical 19 evidence ids, identical order | holds | **holds** |
| exactly 5 `safe_to_assert` | holds | **holds** |
| every item `assertion_level: "factual"` | holds | **holds** |
| `light_deep_ratio` int-or-float, never bool | holds (both types seen) | **holds (both types seen)** |
| the 2 tier items string-valued with `unit: "tier"`, and nothing else is | holds | **holds** |
| the three N1-tier sources agree (D3) | `(low, low, low)` | **`(low, low, low)`** |
| every `stage.*` item carries `model_reliability` | holds | **holds** |
| schema 1.3, `_ground_truth_withheld: true` | holds | **holds** |
| attribution populated, covering every epoch | holds | **holds** |

Nothing to stop for. The evidence contract behaves identically on the split
being tuned on.

### Night-confidence tier distribution

| | high | medium | low | n |
|---|---:|---:|---:|---:|
| test | 12 | **4** | 13 | 29 |
| dev | 11 | 10 | 10 | 31 |

**The skew is the opposite of the one worth worrying about, and it lands on
test.** Dev has 10 low nights (32%), so rule 10 (`missing_review_flag`) and the
`tier_is_low` predicate get ample exercise there. The thin cell is
`tier_is_medium` **on test: 4 packets**, against 10 on dev.

So the concern inverts: after 2B, `tier_is_medium` will be well exercised on the
split being tuned on and barely exercised on the split being graded on. A defect
in it would have four chances to surface at evaluation. Flagged for 2I; the
distribution is pinned by a test so it cannot drift unnoticed.

Every tier occurs at least once in both splits, so no 2B predicate has zero
exercise anywhere.

### The manifests, and the end of globbing

`distillation/results/splits/phase2_{dev,test}_manifest.json`, written by
`distillation/make_phase2_manifests.py`, which verifies **before** writing —
a manifest that fails its own invariant must not exist on disk for something
else to pick up.

`tests/_packets.py` now reads the manifest instead of `PACKET_DIR.glob("*.json")`.
This was the highest-value line in the change set: a glob answers *what is on
disk*, which is exactly the wrong question when the hazard is a stray packet in
the wrong folder — the glob would report the contamination as membership and
every test would pass on it. The manifest answers *what belongs here*, so
disagreement becomes detectable.

Verified by planting a dev packet in the test directory: **two independent tests
failed and named the offending recording**, and the directory was restored
byte-identical afterwards.

| | dev | test |
|---|---:|---:|
| recordings | 31 | 29 |
| subjects | 16 | 15 |
| cohorts | SC 23 / ST 8 | SC 23 / ST 6 |
| recording overlap | ∅ | |
| **subject overlap** | **∅** | |

---

## Part 2 (2B) — tier predicates and the duplication policy

Two keys added to the `text_key` enum, following the existing pattern exactly:

| key | requires | predicate |
|---|---|---|
| `tier_is_high` | `night.confidence` | `night_confidence.tier == "high"` |
| `tier_is_medium` | `night.confidence` | `night_confidence.tier == "medium"` |

Both exact equality against a packet field. The Phase 1 restriction holds.

### The ceiling this removes

`night.confidence` is one of the 5 `safe_to_assert` items and was coverable
only on a low night:

| | coverable before 2B | after |
|---|---|---|
| test | 13 / 29 | **29 / 29** |
| dev | 10 / 31 | **31 / 31** |

The "before" figures are exactly the low-night counts, which is the artefact
stated as a measurement: on 16 of 29 test and 21 of 31 dev packets, mandatory
coverage was capped at 4/5 before a model did anything.

### Mutual exclusivity — 60/60

Exactly one of the three tier predicates is true on every one of the 60 packets
(29 test + 31 dev). Two true at once would mean a model could cover
`night.confidence` with a key that does not describe the night. `TIER_TEXT_KEYS`
names the group so a fourth tier key cannot be added without the check noticing.

### Reason keys are not all tier-gated

Checked, because over-restricting here would have been easy and silent:

| reason key | cites | valid on |
|---|---|---|
| `low_night_confidence` | `night.confidence` | low nights only |
| `n1_low_reliability` | `model.n1_reliability_warning` | **all nights** |

`model.n1_reliability_warning` is `low` on every packet whatever the night tier,
so a high-confidence night can carry an N1 review flag. Tested on high, medium
and low explicitly. The repository already had this right — no change needed,
now asserted.

### The duplication policy: renderer, not verifier

On a low night `night.confidence` is coverable twice: a `review_flag` with
`low_night_confidence` (mandatory under rule 10) and an `observation` with
`tier_is_low`. Both verify, both cite the same id, coverage counts it once.

**The verifier permits both**, deliberately unlike rule 8. Rule 8 rejects citing
both REM latencies because they are two metrics resolving to one value, so
presenting them jointly is false corroboration. Here it is one fact serving two
reporting functions: the flag **warns and instructs**, the observation
**describes what was measured**. Rejecting one would shrink the valid claim
space for no safety gain.

The separation is textual. The old `tier_is_low` text ended *"...so the whole
recording warrants review"*, repeating the banner's action almost word for word;
the observations now give no instruction at all, and a test asserts that by
checking they contain none of "review", "check", "should", "warrants", "must" —
while the banner is asserted to contain "review".

The tier templates name the method ("validation-split tertiles"). That is a
constant of the method, not a fact about one split: `boundaries_fitted_on` is
identical and `requires_ground_truth` is false on all 60 packets. A test asserts
the wording still agrees with the packet, so it cannot drift the way the old
"test split of 29 recordings" literal did.

### The pin

"Does not read like duplication" is a prose criterion and not mechanically
checkable, so the exact rendered output is pinned instead — on one test packet
and one dev packet — the same technique the other exact-text render tests use.
A further test asserts the flag is in the banner and the observation is not.

---

## Part 3 (2C) — the coverage split and the oracle

### Why the single metric had to go

Every other metric in this framework improves when the model says less — an
empty array scores perfectly on violation rate, numeric fidelity and
unsupported-claim rate alike. Coverage is the only counterweight, which makes
its definition load-bearing. But 14 of 19 reportable items are
`safe_to_assert: false` and render with an error bound and a caveat, so a 19/19
report is mostly hedging and not a better report. Pooling also lets a model hide
a missing robust fact behind eleven hedged ones.

| metric | denominator | status |
|---|---:|---|
| `mandatory_coverage` | 5 | **hard requirement** |
| `discretionary_coverage` | 14 | descriptive — report, never optimise |
| `pooled_coverage` | 19 | continuity with the Phase 1 record, not the headline |

Both denominators are 5 and 14 on **all 60 packets**, asserted rather than
assumed, and `pooled_set` is asserted to be exactly their union rather than
maintained separately.

One trap the claim types make easy: `night.confidence` and
`model.n1_reliability_warning` are both mandatory and neither is reachable by
`value` (deviation D1). Coverage therefore counts a **cited ID**, never a claim
type — anything filtering by type would score those two permanently uncovered.

`CoverageRecord` carries the packet's tier, cohort and subject, so 2E can
stratify without re-opening packets.

### The oracle, measured on all 60

| | test (29) | dev (31) |
|---|---|---|
| `oracle_mandatory` | **5 on all 29** | **5 on all 31** |
| `oracle_discretionary` | **14 on all 29** | **14 on all 31** |
| unreachable items | none | none |

No residual ceiling. `oracle_discretionary == nominal_discretionary` everywhere,
which is the case where the two denominators coincide — and they are named
distinctly precisely because that coincidence is a finding, not a guarantee.

So **14 is the real denominator for every model result that follows**, on both
splits, and it is understood now rather than discovered at 2G.

### The oracle does not assume it is right

Candidates are built per item with the forced claim type, run through the
**normal** verifier path, and any claim that fails is dropped and the rest
re-verified to a fixpoint. An oracle verified by its own route would stop being
an upper bound on what the real pipeline accepts.

And "nothing is unreachable on all 60" would be a tautology if the oracle simply
returned all 19. Three tests plant a ceiling and check it is **found**: a stage
item stripped of its `model_reliability` drops out of `oracle_ids` and pushes
`oracle_discretionary` to 13 while `nominal_discretionary` stays 14; a corrupted
night tier drops `oracle_mandatory` below 5, which is the condition that stops
the phase.

### The witness as the strongest positive test

It exercises every coverable item on every packet, rather than the chosen few in
the hand-written `valid_claim_set`. The difference is visible: on a non-low
night the hand-written set reaches 4/5 mandatory — it predates 2B's tier keys —
while the oracle reaches 5/5.

Verification is **set-level**, not per-claim. Rule 10 is a property of the whole
claim set, so a lone `value` claim checked in isolation would fail it on all 23
low nights.

### `oracle_recovery` is null, not zero, when nothing was available

```
oracle_recovery       = |cited ∩ discretionary| / oracle_discretionary   ratio
unrecovered_available = oracle_discretionary - |cited ∩ discretionary|   count
```

A null excludes the packet from an aggregate; a zero would drag it down and
misreport a packet where recovery was never measurable. **No real packet hits
this case** — a test asserts `oracle_discretionary > 0` on all 60 — but it is
defined rather than left to divide by zero.

Neither figure is called a "gap". That name was used earlier in this project and
is wrong for a ratio.

---

## Carry-overs resolved before 2D

### 0a — the stale hand-written claim set

`tests/_packets.py::valid_claim_set` predates 2B's tier keys and reaches 4/5
mandatory on non-low nights where the oracle reaches 5/5. **Kept, not
regenerated**, with a guard.

The reason to keep it is independence. It is the one positive claim set in the
suite that is not produced by the code under test. Regenerating it from the
oracle witness would make `test_the_oracle_beats_the_hand_written_set` compare
the oracle with itself, removing the oracle's only external cross-check. A
hand-built input that happens to be incomplete is still a correct known-valid
input; what it must not be is a coverage *reference*.

So: its docstring says so, and `tests/test_coverage.py` fails if any module
under `report/` references it or imports `_packets`, or if the evaluator or
serializer tests use it. A third test measures, rather than remembers, that it
sits below the oracle on every non-low night.

The guard's first version was wrong in a useful way. It searched source text
for `_packets`, and fired on `evaluate.py` twice - once on a docstring that
mentions `tests/_packets.py`, once on the directory name `phase2_dev_packets`.
Neither uses anything. It now checks the AST for what a module actually imports
and names, and a test feeds it both real uses (which it must catch) and those
mentions (which it must ignore). A guard that cries wolf gets deleted, which is
worse than never having it.

### 0b — two coverage call sites, and they compute different things

| call site | computes | status |
|---|---|---|
| `report/coverage.py` | mandatory / discretionary / pooled, per-packet `CoverageRecord` with tier | **the authority** for every Phase 2 coverage number |
| `verify_policy.PolicyResult.coverage` | pooled only, over `reportable_set` (19) | the Phase 1 continuity figure; not the headline |

Both are kept, and **they must not be unified**. They agree on the pooled
figure by construction — `coverage.pooled_set` is `claim_schema.reportable_set`,
and `test_coverage.py` asserts pooled equals mandatory ∪ discretionary on all 60
— but `PolicyResult.coverage` is part of the frozen Phase 1 verifier and its
tests pin it. A later "cleanup" that routed it through `coverage.py`, or deleted
it, would be a Phase 1 redesign. The evaluator reads only `coverage.py`.

---

## Part 4 (2D) — the serializer

Context budgeting, not evidence selection: all 19 items in, every time; only
fields removed. The caveat and error bound are withheld for a reason other than
size - a model that sees a caveat paraphrases it, and a paraphrased caveat is
one the packet no longer guarantees.

### Worst-case size, estimated

The estimate is characters / 3.5, rounded up. That is deliberately conservative
for English-and-identifier text on BPE vocabularies, where around 4 characters
per token is more typical, so the true counts should come in lower. It is an
estimate until 2G measures a real tokenizer.

| | budget | test | dev |
|---|---:|---:|---:|
| evidence block | 1200 | 413 | 414 |
| full prompt | 2000 | 1106 (SC4532E0) | 1106 (SC4171E0) |
| complete 19-claim output (oracle witness) | 1200 | 805 | 805 |
| prompt + complete output | 3500 | 1909 | 1910 |

Roughly half of every budget is unused. The prompt varies by 5 tokens across
all 60 packets, because only the evidence values change.

### Three things reading the prompt changed

1. The evidence header said "Night confidence tier" while every key predicate
   said `night_confidence.tier`. Now one name for one thing.
2. "cites 1 or more item" - pluralised.
3. The `population_association` line says "no evidence item here is
   associative". That is a packet fact in a fixed template - the renderer's
   "test split of 29 recordings" mistake again. `build_prompt` now **refuses**
   a packet that makes it false, rather than instructing the model wrongly.

### The exclusion list is derived, not only written

The spec's exclusion table is in the test verbatim, and the test **adds** every
packet field that is not serialized, read from all 60 packets. So the check
covers fields the spec did not list - `error_unit`, `error_measured_on`,
`assertion_level`, `duplicates`, `definition`, `note`, `basis`,
`mean_entropy_nats`, `model_f1`, and the top-level `recording_id`, `cohort`,
`subject_id`, `risk` - and a field added to a future packet is checked without
anyone remembering to add it. The caveats' own text is also searched for, not
just the key.

---

## Part 5 (2E) — the evaluator

### Calibration: oracle witnesses score perfectly

On both splits, in every stratum: schema validity, policy pass, overall pass,
mandatory coverage, discretionary coverage, oracle recovery and numeric
fidelity all 1.000; unsupported claim rate 0.000; unrecovered available 0;
zero violations; every packet at 5/5. That is the gate - an evaluator that
cannot score a known-perfect input perfectly reports nothing trustworthy.

### The per-tier table, with the small-n flag

| split | overall | high | medium | low |
|---|---:|---:|---:|---:|
| dev | n=31 | n=11 | n=10 | n=10 |
| test | n=29 | n=12 | **n=4 !** | n=13 |

`!` fires on test medium and nowhere else, in the printed table as well as the
JSON, so a stratified number cannot be quoted at 2I without its warning. The
flag counts **packets**, the unit of independence. Claim-level metrics such as
numeric fidelity are flagged by their stratum's packet count too: test medium
holds ~68 value claims, but they sit inside 4 packets, and 68 does not make
them 68 independent observations.

### `UNSUPPORTED_CODES`

A claim is unsupported when it asserts content its cited evidence does not back:

`L1.unknown_evidence_id` · `L2.cited_id_not_in_packet` · `L2.value_mismatch` ·
`L2.unit_mismatch` · `L2.uncited_quantity` · `L2.text_key_predicate_false` ·
`L2.text_key_dependency_missing` · `L2.not_associative_evidence`

Excluded, each measured elsewhere: **form** errors (every other L1 code - a
malformed claim asserts nothing); **confidence-level** errors
(`unsafe_item_not_hedged`, `safe_item_hedged` - the fact is supported, stated at
the wrong confidence); and **presentation or combination** rules. The spec's
six candidates are all in; `unit_mismatch` and `not_associative_evidence` are
added because each asserts something the packet does not say.
`rem_error_differenced` asserts a derived quantity but always co-fires with
`uncited_quantity`, so leaving it out loses nothing.

### Where the repository contradicted the prompt

1. **Empty outputs do not score zero violations on low nights.** The prompt
   expected an empty array to score 0.0 coverage with zero violations. On a
   low night rule 10 requires a `review_flag`, so an empty array violates it -
   Phase 1 case 22b. On dev that is exactly 10 `missing_review_flag`
   violations, one per low night, and zero on high and medium. The test
   asserts that, rather than the prompt's expectation.
2. **"Of those reaching Layer 2" is not a per-output concept here.**
   `verify_report` drops claims that fail Layer 1 and runs Layer 2 on the
   rest, so an output with one malformed claim still reaches Layer 2 for its
   other claims. `policy_pass_rate` is therefore defined over **schema-valid**
   outputs - no Layer 1 violation at all - which is the cleanest reading that
   keeps the three validity rates nested.
3. **A missing output is not an empty one**, and the prompt did not say which
   it was. An empty array is a model that chose to say nothing; a missing file
   is a model that produced nothing usable. It scores zero coverage and does
   not pass, but carries no violation, so it is counted as `n_missing` and
   `passed + failed + missing == n` is asserted.
4. The dev manifest's rows say `split: "val"`, not `"dev"`, because the dev set
   is the upstream validation split. `PACKET_DIRS` maps `"val"`.

---

## Three checks before 2F (12 September 2026)

### 1. The empty-array contradiction - a wording defect, not a code defect

The Phase 1 record said "the empty claim set passes every safety rule at 0.0
coverage", and 2E found an empty array violating rule 10 on every low night.
Both cannot be unconditionally true.

The Phase 1 case, `test_22_empty_set_is_safe_and_scores_zero`, runs on
`SC4011E0-PSG` - a **high**-confidence night. It asserted zero violations and
0.0 coverage, and it passes. So there was no defect: rule 10 was never being
skipped. Both tests were right about their own packets. What was wrong was the
**wording** - the test's name and three sentences in `PHASE1_REPORT.md` (sections
7, 9 and 14) and two in `README.md` described one tier as if it were all of
them.

The low-night half already existed as `test_22b`, but it only asserted the
rule-10 code was *present*. Now:

- `test_22` is renamed `..._on_a_non_low_night_...` and asserts its packet's tier
- `test_22b` asserts **exactly one** violation, that it is
  `L2.missing_review_flag`, and that it is report-level (no claim id - there
  are no claims)
- `test_22d` sweeps all 60 packets: all 23 low nights fail exactly once, all 37
  others are clean

The documents now state the behaviour as it is, with a dated correction note in
the Phase 1 record rather than a silent rewrite of it.

### 2. The rem_error_differenced coupling - holds, and now tested

`UNSUPPORTED_CODES` excludes `L2.rem_error_differenced` because it co-fires with
`L2.uncited_quantity`. That is now an invariant, checked on every output of
twelve corruption types on both splits, on every one of the 60 packets with
both REM ids and both signs, and in a targeted single-packet case. **It holds
everywhere.**

It holds for a reason that is a property of the data, not of the code, and
that reason is tested too. Rule 9 fires when a claim's value equals the exact
difference of the two errors; rule 11 stays silent only if that value also
equals the cited REM latency. So they could decouple only if a REM latency
equalled 4.8403 minutes. Latencies are whole 30-second epochs, so they sit on a
0.5-minute grid, and 4.8403 does not. The test asserts both halves on all 60
packets - so if a future packet broke either, the test fails and the exclusion
has to be revisited, instead of quietly becoming unjustified.

One thing this surfaced and did not change, since rule semantics are frozen:
rule 9 only fires on the **exact float** `53.05 - 48.2097`. A model that writes
the difference rounded - `4.84`, or `4.8403` - evades rule 9. Nothing unsupported
escapes, because the same claim still trips `value_mismatch` and
`uncited_quantity`, both in the unsupported set. But rule 9's own per-rule count
will under-report differencing, and should be read that way at 2G.

**Reading the per-rule table, from 2G onwards:**

> `L2.rem_error_differenced` is an exact-value detector, not a rounded-value
> classifier. A differenced error written to fewer decimal places is still
> rejected, via `L2.value_mismatch` and `L2.uncited_quantity`, but is not
> attributed to this rule. Its per-rule count is therefore a lower bound on
> differencing attempts.

Pinned by `test_adversarial.py::test_12b_a_rounded_difference_is_rejected_but_not_attributed_to_rule_9`:
`4.84`, `4.8403` and both `round()`ed forms are rejected, and
`rem_error_differenced` is not among their codes. The rule is deliberately not
widened - a tolerance band would contradict the no-tolerance principle rule 2
rests on, and two numeric rules disagreeing is worse than one documented blind
spot.

### 3. The policy_pass_rate denominator - presentation fixed, metric unchanged

`policy_pass_rate` is computed over schema-valid outputs, so its denominator
varies by model: 72/80 = 90.0% and 88/98 = 89.8% are near-identical rates from
models 16 points apart overall. The metric is right; showing it without its
denominator is what misleads.

- Every validity rate now carries `_num` and `_den` in the JSON, beside the
  rate, in every stratum
- The printed cell is `88.7% (71/80) n=98` - the fraction and the stratum's
  packet count side by side, so a shrunken denominator is visible against `n`
- Schema validity and overall pass print **above** policy pass
- Tests walk the whole JSON and fail if `policy_pass_rate` appears anywhere
  without its numerator and denominator, parse every validity cell for its
  fraction, and encode the 72/80-vs-88/98 example directly

---

## GBNF runtime check (12 September 2026) - negative, and it changes 2F

The 2G ablation compares grammar-constrained with unconstrained generation, so
it needs a runtime that can apply a GBNF grammar. Checked before 2F rather than
discovered in it. **Step 1 failed, so steps 2, 3 and 5 were not run** - nothing
was installed and nothing downloaded to work around it.

| step | result |
|---|---|
| 1. runtime present | **absent.** No `llama-cli` / `llama-server` on PATH (`main` resolves to Windows' `main.cpl`, unrelated). No `llama-cpp-python`: `pip show` finds nothing and `import llama_cpp` raises `ModuleNotFoundError`. No Ollama executable in its three standard install locations; `~/.ollama` holds only keys and an empty `models/`, a leftover. No LM Studio. `transformers` 5.10 and `torch` 2.10 are installed, but neither applies GBNF natively. |
| 2. grammar accepted | not run - no runtime |
| 3. `claims.gbnf` loads | not run - no runtime |
| 4. model available | **one GGUF, unsuitable.** `F:\NLU result\nepglish-nlu-v3-q8.gguf`: 8.1 GB, llama architecture, `general.name` "Nepglish Merged", 8.0B parameters, Q8_0, context 8192 - read from the GGUF header, no runtime needed. An unrelated project's fine-tune, and twice the 1-4B target. Usable to test grammar mechanics once a runtime exists; not a 2F candidate. |
| 5. constrained output well-formed | not run - no runtime |

What installing would involve, recorded for the decision rather than acted on:
Python is 3.14.0, recent enough that prebuilt `llama-cpp-python` wheels may not
exist, which would mean a source build - `cmake` and MSVC `cl` are absent,
MinGW `gcc`/`g++` present, CUDA 12.9 `nvcc` present. A prebuilt llama.cpp
release binary driven by subprocess (`--grammar-file`) would avoid the binding
entirely. Ollama would not satisfy this check as specified: as far as is known
here, its API constrains through a JSON schema passed as `format`, not an
arbitrary GBNF file.

A note on how this section arrived: commit `898e403` said it recorded this
check, but the script meant to write it failed on an escape sequence and the
commit went ahead without it. The section landed in the commit after. The
message was wrong; nothing else in that commit was.

---

## Canonical register: A (clinician) - decided 12 September 2026

> **Canonical register: A (clinician).** The deciding factor is rule 10, which
> exists to force a review flag onto low-confidence nights. Register B softens
> that banner to a plain statement of uncertainty, which would leave the
> verifier enforcing a signal the reader no longer receives as one. B's other
> problems are structural rather than stylistic: caveats are the packet's
> verbatim words, so plain-language rewriting either breaks that guarantee or
> requires a per-caveat rewrite table that will drift. Register C writes rule
> numbers and field names into prose, so renumbering a rule would invalidate
> every report ever rendered.
>
> **B is recorded as future work, not rejected.** Verified claims are
> register-neutral - a claim carries `{claim_id, claim_type, cites, subject,
> value, unit}` and no prose - so a wellness-facing renderer is a second
> renderer over the same claim set, not a redesign of anything beneath it.
>
> A's cost is that it omits the tier mechanics. Accepted: those remain in the
> packet, which is where an auditor would look.

The observations that informed this, recorded as part of why B and C were not
adopted, and not reopened: B rendered caveats verbatim beside plain prose
(`mean relative error 85% on validation` next to "It took you about 29.5
minutes to fall asleep"), and could not round `21.5806 minutes` without
breaking "same numbers". C wrote `[arch.sleep_onset_latency, hedged]` and
`(rule 10)` into the report text itself.

---

## Renderer provenance audit (12 September 2026)

The property: **a rendered report is derivable from its packet alone** - the
renderer-side counterpart of rule 11. Anything in the output comes from one of
three legitimate sources: a packet field; a declared renderer constant, which
may be wording or a label and never a fact; or a deterministic consequence of a
verifier rule that already passed. Any other file is illegitimate.

### `open` during rendering: 0 files, on all 60

`builtins.open` and `io.open` were instrumented around `render_report` only.
Every packet was loaded and every claim set verified beforehand, so only
rendering is measured. The claim set is the oracle witness plus the N1 review
flag, which together reach every template a report can hit.

**0 files opened across 60 renders** (29 test, 31 dev). That held on the
pre-audit renderer, so there was nothing to report before changing anything.
It holds on the register-A renderer too, where it is now
`test_render_opens_no_files_on_all_60_packets`.

### Every output-reaching constant in `render.py`, classified

| constant | where | class | rests on |
|---|---|---|---|
| `Wake`, `N1`, `N2`, `N3`, `REM` | `STAGE_LABEL` | label | the evidence id it names |
| `validation`, `test`, `training` | `SPLIT_WORD` | label | `attribution_quality.evaluated_on_split`; an unlisted split passes through as the packet spells it |
| `: ` ... `.` | value, hedged value | wording | punctuation around the packet's `label` and value |
| `%` | `_quantity` | wording | display convention for `unit: fraction` |
| ` minutes` / ` minute` | `_minutes` | wording | the packet's `unit` string; singular only for exactly 1 |
| `mean absolute error ` | `_error_phrase` | label | names `mean_abs_error` |
| ` on the ` | `_error_phrase` | wording | joins `error_measured_on`, which is read with no default |
| ` percentage points`, ` per hour` | `_ERROR_DISPLAY` | **the one declared inference** - see below | the `(unit, error_unit)` pair |
| ` is a ` ... `-reliability stage for this model` | `_tier_clause` | wording | `model_reliability`, read from the cited item |
| ` Caveat: ` | hedged value | label | names `caveat`, which follows verbatim |
| ` is in the ` ... ` tier.` | `OBSERVATION_TEMPLATE` | wording | `label` and `value` of `night.confidence` |
| `{label}.` | `n1_reliability_is_low` | wording | the item's own label is the whole statement |
| `This recording falls in the ` ... ` night-confidence tier.` | `REVIEW_TEMPLATE` | wording | `night.confidence` `value` |
| `Review the full hypnogram before relying on any figure below.` | `REVIEW_TEMPLATE` | rule consequence | rule 10 forces this flag on every low night; "below" holds because `render_report` always places the banner first |
| `Review N1-scored epochs individually before relying on them.` | `REVIEW_TEMPLATE` | wording: an instruction, asserting nothing | present only once rule 7 has confirmed the key's predicate on the cited N1 item |
| `REVIEW REQUIRED` | `render_report` | rule consequence | rule 10 |
| `: associative evidence. It describes an association across a population and is not a statement about this recording.` | `population_association` | rule consequence | rule 12 admits only `associative_only` items; unreachable today |
| `Explainability check (gate ` | footer | label | names `attribution_quality` |
| `): `, `, `, ` of `, ` predictions met, across `, ` recordings.` | footer | wording | `gate`, `verdict`, `n_met`, `len(predictions_met)`, `evaluated_on_n_recordings`, split |
| `pre-registered ` | footer | wording, conditional | emitted only when `preregistration` is present |
| `It describes the model across that cohort, not this recording.` | footer | packet field, paraphrased | emitted only when `scope` begins `COHORT` |

Out of the table because they cannot reach output: docstrings, exception
messages, dict keys (claim keys and evidence ids, which are looked up and never
printed), format specs, and `_num`'s `rstrip("0")`.

`test_no_digit_in_any_output_constant` enforces the digit half of this. It is an
AST scan of every output-reaching string for a digit that is not part of a stage
name, because a number the packet did not supply is the commonest disguised
fact. A second test checks that the scan is not vacuous.

### What resisted classification

Two things. Both are in the table, and neither is hidden.

1. **The error display scale.** `error_unit` is `count_or_ratio` for every
   item that is not in minutes. The renderer still prints a fraction's error in
   percentage points and a rate's error per hour, which neither field says on
   its own. It is an inference: a mean absolute error is in the units of its
   quantity. It is declared as a five-row table keyed on the `(unit,
   error_unit)` pair, and any pairing outside the table raises `UndeclaredUnit`.
   Every pairing in the 60 packets is in the table (tested). The real fix is a
   packet field (`error_unit: "fraction"`, `"per hour"`). That would rebuild
   the test packets, so it is recorded here, not done, while the deterministic
   tier is frozen.
2. **The cohort sentence keys on a prefix of a prose field.** `scope` reads
   "COHORT, NOT THIS NIGHT. The verdict was evaluated once...". The renderer
   reads the `COHORT` token and paraphrases it, and renders no sentence without
   it. A structured scope field would make this a read rather than a parse.

### Undeclared facts found and removed

All of these were in the pre-audit renderer:

| was | problem | now |
|---|---|---|
| "N1 detection is the weakest part of this model." | a comparison across stages; the cited item states N1's reliability, not a ranking | the item's own label |
| population association: "In population studies, X has been reported as associated with sleep-disorder risk" | a literature claim, and an outcome the packet never names | only what rule 12 guarantees: the item is associative |
| footer "of 5" | a hardcoded count | `len(predictions_met)` |
| footer "gate 3a", "pre-registered" | hardcoded | `gate`; conditional on `preregistration` |
| footer "measured once", "No per-night version of it was measured." | true, but stated only in the `scope` prose, not in a field | removed |
| `error_measured_on or "validation"` | a default asserting where an error was measured | read, no default; if the field is absent, the clause is dropped |
| tier observations: the method, "tertiles fitted on validation-split entropy" | a constant stating how the tier was computed; register A drops the mechanics anyway | the tier only |
| a full stop appended to a caveat without one | an edit to the packet's words (never fired: every caveat ends with one) | verbatim |

### The five A lines

| line | finding | now |
|---|---|---|
| "on held-out validation nights" | **hardcoded** in the scratch A render the register was chosen from. "held-out" appears in no packet field, and the committed renderer had the `"validation"` fallback above | `on the {error_measured_on}`, which renders "on the validation split". A sentinel value appears verbatim, and an absent field drops the clause (tested) |
| "N2 is a high-reliability stage for this model" | read, not inferred | from the cited item's own `model_reliability`. A sentinel appears verbatim; a missing value raises `MissingField` rather than printing a tier the packet does not have |
| "percentage points" | neither read nor assumed for `stage.*`: it keys on `unit == "fraction"` | the declared inference above. An `arch.*` fraction made hedged in a mutated copy also gets percentage points (tested) |
| "mean relative error 85% on validation" | verbatim. The renderer *could* append a full stop, though it never did | appended with no edit, asserted on all 840 hedged lines (14 per packet across 60 packets) |
| "Caveat:" | wording only: a label for the `caveat` field | unchanged |

### Seen while pinning, and what was done

- **"1 minutes" was about to be pinned.** The WASO on dev SC4111E0 is 1.0. I
  fixed it as wording (grammatical number) before writing the goldens. The
  digit scan then caught the first version of the fix, which spelled the "1" as
  a literal; it now formats the packet value.
- **Two labels for one item.** The packet labels `stage.W.fraction` "W as a
  fraction of the night", while `STAGE_LABEL` calls it "Wake" in the tier
  clause. Both are labels and neither is a fact, so it is left as is.
- **The N1 review flag renders in the body**: an instruction outside the
  banner. It is kept there because the banner is reserved for the rule-10
  signal. The pins use the oracle witness, which carries no N1 flag, so this
  placement is not pinned.

### The six pins

`TestRegisterAPins` compares the rendered oracle witness byte-for-byte against
`tests/golden/register_a/`. There is one test packet and one dev packet per
night-confidence tier:

| split | high | medium | low |
|---|---|---|---|
| test | SC4011E0 | SC4022E0 | SC4202E0 |
| dev | SC4081E0 | SC4171E0 | SC4111E0 |

- A mismatch fails with a unified diff.
- A seventh test asserts that exactly these six files exist, so a stale golden
  cannot linger.
- The goldens are regenerated only by `tests/golden/make_register_a.py
  --write`. Without `--write`, it prints the diff and changes nothing.
- The 2B low-night pin (`TestLowNightBothClaims`) is rewritten in register A,
  on the same two packets as before.

---

## Recorded, not fixed: the dev packets' contract was fitted on the dev nights

`metric_reliability`, the decoder temperature and the night-confidence tier
boundaries were all fitted on the **validation** split. The dev packets are
built from those same nights. So a dev packet's *contract* — which metrics are
`safe_to_assert`, what the error bounds are, where the tier boundaries fall —
was fitted on the night it describes.

This is acceptable here, and the reason is narrow: **Phase 2 measures whether a
model respects a given contract, not whether the contract is accurate.** A model
that transcribes a value exactly and hedges what the packet says to hedge scores
the same however well-calibrated the underlying bound is.

It would **not** be acceptable for any claim about the error bounds themselves.
If Phase 3 wants to say "the reported bounds hold on unseen data", the dev split
cannot support it, and this note is the reason.

It is written here rather than left to be discovered.
