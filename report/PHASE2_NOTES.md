# Phase 2 — working notes

**Nishant Shah · Team 40 · Project 48**
**Started: 11 September 2026**
**Status: 2A-2E complete, register A pinned, local runtime verified, rule 10 fixed,
rendering gated on a clean result, cited coverage added, 2F preflight done, reference runner built — 364
tests, all passing. Reference run: 3 of 31 P1 responses cached, with P2 in
reserve; the reference model is Gemini 3.8 Flash. Deployment: size and peak memory
measured on the x86 evaluation host; throughput predicted analytically for the
named target, a Raspberry Pi 5. The deterministic tier is finished and frozen. One model has
run, once, to confirm the grammar holds mechanically. That single output has
been scored as an exploratory reading, not a measurement. The reference model
is chosen (the key is Flash-class only); its rate limits,
read from AI Studio, allow 20 requests a day.**

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

**Superseded the same day.** On instruction, a prebuilt llama.cpp binary was
installed and all five steps now pass - see *Local runtime* below. This section
stays as the record of what was present before.

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
   **Pinned since.** `test_every_scope_still_begins_with_the_cohort_token`
   asserts the prefix on all 60 packets. If the wording changes, a test breaks
   instead of a sentence silently disappearing.

### Future work: two structured packet fields

Both fixes belong in the packet builder, and each would rebuild the test
packets. They are deferred while the deterministic tier is frozen, not dropped.

| field | today | the fix | until then |
|---|---|---|---|
| `error_unit` | `count_or_ratio` for every item not in minutes | the unit the error is actually in: `fraction`, `per hour`, `count`, `ratio` | `_ERROR_DISPLAY` stays as it is. It is declared, exhaustive over the 60 packets, and raises `UndeclaredUnit` rather than guessing |
| `attribution_quality.scope` | prose beginning "COHORT, NOT THIS NIGHT..." | a structured field, e.g. `scope_level: "cohort"`, with the prose kept beside it | the prefix pin above |

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

## Local runtime: llama.cpp b10927, grammar verified (12 September 2026)

This supersedes the negative check above. On instruction, I installed a
prebuilt llama.cpp release binary outside the repository and drove it by
subprocess: not `llama-cpp-python`, and with no fallback to Transformers.
Nothing in `report/` imports it or calls it. The only callers are scratch
scripts outside the repository.

### Why llama.cpp for deployment measurements

The deployment experiment targets quantized GGUF inference, so deployment measurements are performed with the corresponding llama.cpp runtime rather than an fp16 Transformers configuration. Throughput figures are analytical bandwidth-bound estimates rather than on-device measurements; peak resident memory and quantized model size are measured directly.

### Install

- **Build:** `llama-b10927-bin-win-cpu-x64.zip`, from the `ggml-org/llama.cpp`
  GitHub release `b10927`. The release tagged "latest" is a nightly marker,
  not a build.
- **sha256:** `ec597a9ba17e48256138377c22a8004f38b9634d5feaacdb7269bab5d03bf6c1`
- **Location:** unpacked to `C:\Users\shahn\tools\llama.cpp\b10927\`
- **Download:**

  ```
  Invoke-WebRequest https://github.com/ggml-org/llama.cpp/releases/download/b10927/llama-b10927-bin-win-cpu-x64.zip
  ```

  followed by `Expand-Archive`.

### The five steps, in order - all pass

Generation uses `llama-completion.exe` from the same build. In this build
`llama-cli` has no `-no-cnv` option, whereas `llama-completion` has one, and
its own help gives it as the text-generation form. In the invocations below,
`$L` is `C:\Users\shahn\tools\llama.cpp\b10927\llama-completion.exe` and `$M`
is `C:\Users\shahn\models\qwen2.5-1.5b-instruct-q4_k_m.gguf`.

| step | invocation | result |
|---|---|---|
| 1. runtime version | `llama-cli.exe --version` | `version: 0.4.0-dev (build 10927, commit 718f7b417)`, `built with Clang 20.1.8 for Windows x86_64` |
| 2. trivial grammar `root ::= "yes" \| "no"` | `$L -m $M -p "Is the sky blue on a clear day? Answer:" --grammar-file yesno.gbnf -n 8 --temp 0 --seed 0 -no-cnv --no-display-prompt` | `yes`, then end of text; exit 0. **Control:** with `-p "The capital of France is"`, the output is `no` with the grammar and ` Paris. The capital of Italy` without it. The grammar constrains; it is not merely accepted. |
| 3. `claims.gbnf` loads | `$L -m $M -p "Output:" --grammar-file report/claims.gbnf -n 24 --temp 0 --seed 0 -no-cnv --no-display-prompt` | **no parse error**. Exit 0, and no stderr line mentions grammar, parse or error. The 24 tokens begin `[` `{"claim_id":  "c1", "claim_type": "population_association",`, truncated by `-n` as intended |
| 4. one model | Qwen2.5-1.5B-Instruct Q4_K_M, from `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | 1,065.6 MiB; sha256 matches Hugging Face (table below) |
| 5. constrained generation | `$L -m $M -f step5_prompt.txt --grammar-file report/claims.gbnf -n 3000 -c 12288 --temp 0 --seed 0 -no-cnv --no-display-prompt` | dev SC4111E0 (low tier), `build_prompt` at prompt version 2D.1: 3,856 characters, 1,007 tokens. Exit 0 after 60 s wall. Prompt eval ran at 109.9 tok/s, and generation produced 1,076 tokens at 22.75 tok/s, reaching end of text within the cap. **`json.loads` gives a list of 18 claims, and Layer 1 finds 0 violations.** |

This was a mechanical check only. Layer 2 was not run and output quality was
not evaluated. It used raw completion with no chat template, because the prompt
format is a 2F selection decision. One claim from the output, verbatim apart
from whitespace:

```json
{"claim_id": "c1", "claim_type": "value", "cites": ["arch.total_sleep_time"], "subject": "this_recording", "value": 433.5, "unit": "minutes"}
```

The output also contained what the grammar is documented *not* to prevent:
`text_key` / `cites` pairings that do not agree, which rule 7 exists to catch.
That is the grammar boundary stated in the README, observed, not a result.

### The five candidates, all Q4_K_M

The other four were fetched only after step 5 passed, and none has been run.
The three that had no ungated official GGUF come from bartowski's
quantizations. Every file's sha256 matches the value Hugging Face publishes for
it.

| model | repository | MiB | sha256 |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | 1,065.6 | `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| Llama-3.2-3B-Instruct | `bartowski/Llama-3.2-3B-Instruct-GGUF` | 1,925.8 | `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff` |
| SmolLM2-1.7B-Instruct | `HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF` | 1,006.7 | `decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33` |
| Gemma-2-2b-it | `bartowski/gemma-2-2b-it-GGUF` | 1,629.4 | `e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787` |
| Phi-3.5-mini-instruct | `bartowski/Phi-3.5-mini-instruct-GGUF` | 2,282.4 | `e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5` |

All five are in `C:\Users\shahn\models\`, outside the repository.

---

## The grammar/verifier boundary is deliberate (12 September 2026)

The runtime check showed two behaviours. **Neither is to be fixed.**

1. The grammar permits `population_association`, although no packet carries
   associative evidence. Every such claim is therefore guaranteed to fail
   rule 12.
2. The grammar permits `text_key` / `cites` pairings that do not agree. Rule 7
   catches them.

Making the grammar packet-conditional, for example by dropping
`population_association` when no item is associative, or constraining which
ids each key may cite, would stop a measurable model failure from being
measured. A model that emits a claim it had every means to know was
unsupported is showing a real behaviour: the prompt says the type is reserved,
the verifier catches the claim, and the per-rule count shows it. Constraining
it away would raise policy pass rates for reasons that have nothing to do with
the model.

The line, stated once:

- **The grammar enforces syntax:** claim types, field sets and order, enums,
  and the per-type id vocabulary. It is the same grammar for every packet.
- **The verifier enforces evidence policy:** anything that needs the packet,
  or agreement between two fields.
- **The line between them is where model behaviour becomes visible.** Whatever
  the grammar makes ungeneratable, the evaluator never sees; whatever is left
  to the verifier gets counted.

`claims.gbnf` stays packet-independent and generated from the schema
constants. Neither this step nor this decision changes it.

---

## EXPLORATORY, NOT A MEASUREMENT: the step-5 output, scored (12 September 2026)

> **One packet, one model, one seed.** This is not a measurement, and it must
> not enter any selection decision: not model choice, not prompt design, not
> K. It may inform only the *shape* of small-model failure under this contract,
> and through that, how 2F is framed. SC4111E0 is a dev packet, so the
> test-set access log does not apply.

**Input.** The step-5 output as saved to disk, not regenerated:
Qwen2.5-1.5B-Instruct Q4_K_M on dev SC4111E0-PSG (low tier), with
`claims.gbnf`, temp 0, seed 0 and raw completion. The only change was removing
llama.cpp's `[end of text]` display marker and the surrounding whitespace.

**Scoring.** The output went through `verify_report`, then `evaluate`. The
manifest was one row, copied from `phase2_dev_manifest.json`, so no other
night counts as missing.

| metric | result |
|---|---|
| claims | 18 |
| Layer 1 | pass, 0 violations |
| **Layer 2** | **fail: 16 violations**, one on each of 16 claims |
| overall pass | no |
| `mandatory_coverage` | **2/5** (0.4). Covered: `arch.total_sleep_time`, `arch.time_in_bed`. Missing: `arch.sleep_efficiency`, `model.n1_reliability_warning`, `night.confidence` |
| `discretionary_coverage` | 0/14 |
| `oracle_recovery` | 0.0; `unrecovered_available` 14 |
| `numeric_fidelity` | **11/11 exact** |
| `unsupported_claim_rate` | 7/18 = 0.389 |

**Per rule:**

| count | code | where |
|---|---|---|
| 9 | `L2.unsafe_item_not_hedged` | Nine `safe_to_assert: false` items emitted as bare `value` claims: sleep onset latency, WASO, REM latency, sustained REM latency, REM periods, stage transitions, transition rate, light-to-deep ratio, and wake interruptions per hour |
| 7 | `L2.text_key_dependency_missing` (unsupported) | Every key-bearing claim uses a key whose required item it does not cite. Six observations use `n1_reliability_is_low` on `arch.sleep_efficiency` and on each of the five stage fractions. The one review flag cites `night.confidence` with `reason_key: n1_low_reliability` |

**What would render.** Only c1 and c2 verify, so the rendered report is just
this:

```
Total sleep time: 433.5 minutes.
Time in bed: 464 minutes.

Explainability check (gate 3a): PASS, 3 of 5 pre-registered predictions met, across 31 validation recordings. It describes the model across that cohort, not this recording.
```

It has no REVIEW REQUIRED banner, although this is a low-confidence night.
That exposes the next finding.

### Found, not fixed: an invalid flag satisfies rule 10

`rule_low_night_needs_flag` is satisfied by **any** `review_flag` that cites
`night.confidence`, whether or not that flag itself verifies. Here, c18 cites
`night.confidence` with the wrong reason key. It fails rule 7 and is dropped
from the enriched set, yet its presence still suppresses
`L2.missing_review_flag`. A controlled check on the same packet:

| claims | codes | banner |
|---|---|---|
| a TST value only | `L2.missing_review_flag` | no |
| + a flag with `reason_key: n1_low_reliability` | `L2.text_key_dependency_missing` | no |
| + a flag with `reason_key: low_night_confidence` | none | yes |

The report still fails, so overall pass is unaffected. Two things are affected:

1. **The per-rule count misattributes the failure.** It shows zero low nights
   left unflagged and puts the failure under rule 7. That is exactly the shape
   this section exists to show, and rule 10 is the rule the canonical register
   was chosen to protect.
2. **Nothing gates rendering on a clean report.** If a failing report were
   rendered, a low night would reach its reader without the banner, and no
   rule-10 violation would say so.

The tier is frozen, so nothing was changed. The candidate fix is for rule 10 to
count only flags that verify, or only flags with `reason_key:
low_night_confidence`. Either is a rule change and needs a decision before 2F,
as does whether a report with any violation renders at all.

**Fixed the same day** as a documented correctness fix: rule 10 now reads the
claims that survive every other check. See *Rule 10 read the wrong claim set*
below. Whether a report with any violation renders at all was closed the same
day: it does not (see *Rendering is gated on a clean verification result*).

### The shape, for framing 2F only

- **Syntax is not the bottleneck.** The grammar made Layer 1 free, and every
  number was transcribed exactly (11/11).
- **The failures are policy failures:**
  - choosing the claim type by hedging status: 9 of 11 numeric claims needed
    `hedged_value`;
  - matching a key to its evidence: 7 of 7 key-bearing claims got it wrong.
- **Coverage collapses as a consequence.** Two verified claims leave
  mandatory coverage at 2/5 and discretionary coverage at 0/14.

So on this single output, 2F's question reads as whether a small model follows
the contract's *policy*, not its *format*. That is a framing, not a result.

### Re-read after the rule-10 fix and the render gate (12 September 2026)

**Still exploratory.** One packet, one model, one seed, raw completion. It
must not enter any selection decision. The output on disk was re-scored; it was
not regenerated.

| figure | value |
|---|---|
| violations | 17: 9 × `L2.unsafe_item_not_hedged`, 7 × `L2.text_key_dependency_missing`, and 1 × `L2.missing_review_flag` (report-level, new since the rule-10 fix) |
| claims verified | 2 of 18: c1 (total sleep time) and c2 (time in bed) |
| `mandatory_coverage` | **2/5**. Covered: `arch.total_sleep_time`, `arch.time_in_bed` |
| `discretionary_coverage` | 0/14 |
| `oracle_recovery` | 0.0 |
| `numeric_fidelity` | 11/11 |
| `unsupported_claim_rate` | 7/18 |
| rendered | no: `render_report` refuses, carrying all 17 violations |

**Where the three mandatory misses come from.** This is a read of the
verifier's output, not an evaluator metric. Each item was verified, cited but
rejected, or never cited:

| mandatory item | outcome |
|---|---|
| `arch.total_sleep_time` | verified |
| `arch.time_in_bed` | verified |
| `arch.sleep_efficiency` | cited, rejected: emitted as an `observation` with key `n1_reliability_is_low` (rule 7), where a `value` was needed |
| `night.confidence` | cited, rejected: a `review_flag` with reason `n1_low_reliability` (rule 7), where `low_night_confidence` was needed |
| `model.n1_reliability_warning` | never cited |

**The discretionary items.** All 14 were cited, and all 14 rejected:

- The nine `arch.*` items came as bare `value` claims where `hedged_value` is
  required, with every number exact. The model emitted no `hedged_value` claim
  at all.
- The five stage fractions came as observations carrying the N1 key.

**Overall.** Of the 19 reportable items, the model cited 18. The one it never
touched is `model.n1_reliability_warning`, which is the very item that the key
it applied everywhere depends on.

So the number is 2/5, but its anatomy is mostly claim shape:

- Of the three mandatory misses, two are items the model found and cited with
  the wrong claim type or key. Only one is a true omission.
- Every rejection among the 16 failing claims is one of two policy errors:
  hedging, or key/evidence agreement.
- None is a transcription error or a failure to locate the evidence.

**One confound,** recorded because it bears on exactly this question. Step 5
was a grammar-mechanics check, so it used raw completion with no chat
template. An instruct model outside its chat format may follow policy worse
than it would inside it. The prompt format is a 2F selection decision, so the
confound is named here, not tested.

---

## Rule 10 read the wrong claim set - fixed (12 September 2026)

This is a correctness fix inside the frozen tier, found by the exploratory
step-5 score above. It changes how rule 10 is evaluated and nothing else: no
other rule, claim type, coverage definition, schema or grammar.

**The defect.** Rule 10 requires a `review_flag` citing `night.confidence` on
a low-confidence night. It checked the **submitted** claims, but the banner
renders from the **verified** ones. A flag rejected by its own rules therefore
satisfied rule 10 while contributing no banner. The night rendered without its
safety signal, and no violation was recorded. It was exploitable in the wrong
direction: a malformed flag scored better than no flag, for an identical unsafe
report.

**Which flags got through.** Not every invalid flag did. Probed before the fix:

| flag | before the fix | why |
|---|---|---|
| unknown `reason_key` | caught | Layer 1 rejects it, and `verify_report` re-runs policy without the claim |
| `low_night_confidence` without its dependency | caught | without `night.confidence` in `cites` it never looked like a rule-10 flag |
| `n1_low_reliability` citing `night.confidence` (the step-5 flag) | **satisfied rule 10** | a Layer 2 per-claim failure |
| valid key, `subject: population` | **satisfied rule 10** | a Layer 2 per-claim failure, found while probing |

So the hole was any flag citing `night.confidence` that failed a Layer 2
per-claim rule.

**The fix,** in `report/verify_policy.py`. Rule 10 now reads
`surviving_claims()`, which uses the fixpoint pattern of `report/oracle.py`:
verify, drop what fails, re-verify until stable, then test rule 10 against what
survived.

- **Why a fixpoint at all.** Every other rule judges one claim on its own, so
  the first pass is already stable, and the surviving set equals the verified
  set exactly. The loop keeps that true if a rule that judges claims jointly
  is ever added.
- **The loop is bounded.** It runs at most `len(claims) + 1` passes, because
  each pass either drops a claim or returns.
- **One copy of the rule list.** The per-claim rules are now applied by a
  single helper, `_claim_violations()`, which both the main pass and the
  fixpoint use, so the two cannot apply different lists.
- **Scope.** The change is not generalised: rule 10 is the only set-level
  rule.

**Blast radius: nil, as expected.**

- All 271 existing tests passed unchanged with the fix in place, before any
  new test was added. There was nothing to stop and report.
- **Oracle, coverage and evaluator numbers.** A snapshot was taken on all 60
  packets before and after the fix, and the two are **byte-identical**. It
  covered:
  - every oracle witness and its id set;
  - the witness's coverage;
  - `score_output` for the witness, the empty array and a missing file;
  - the per-split strata.
- The oracle witness carries a valid flag. The empty array still violates
  rule 10 exactly once on each of the 23 low nights (`test_22d` is
  unchanged).
- The step-5 output now scores 17 violations: the same 16, plus
  `L2.missing_review_flag`. Its exploratory record above stays as it was
  scored at the time.

**Tests added: seven, taking the suite from 271 to 278.** All are in
`tests/test_adversarial.py`.

| test | asserts | against the pre-fix rule |
|---|---|---|
| 10a: unknown `reason_key` | the flag is rejected, `missing_review_flag` fires, and there is no banner | passes (a guard) |
| 10b: `low_night_confidence` with its dependency absent | the same | passes (a guard) |
| 10c: `n1_low_reliability` citing `night.confidence` | the same | **fails**: the hole |
| 10c2: valid key, wrong subject | the same | **fails**: the second route in |
| 10d: a valid flag plus failing claims | rule 10 is silent and the banner stays | passes (the overcorrection guard) |
| 10e: all 23 low nights, each with an invalid flag | exactly the flag's own code plus `missing_review_flag` on each | **fails** |
| banner if and only if rule 10 is silent | 12 claim sets on each of the 23 low nights (276 checks), in both directions | **fails** |

The "against the pre-fix rule" column comes from running the new tests with
only `verify_policy.py` reverted. They fail exactly where the fix matters, so
they test the fix rather than merely agreeing with it.

The last test couples the rule to the thing it protects. The rule and the
renderer read different code, so this test is what stops them drifting apart
again.

---

## Rendering is gated on a clean verification result (12 September 2026)

This is a gate-only change inside the frozen tier. Of the tier's modules, only
`report/render.py` changed. No rule, schema, coverage definition, oracle,
evaluator, grammar, packet, wording, ordering, whitespace or template changed.

**The rule.** A report renders only from a verification result with exactly
zero violations: `len(violations) == 0`. That does not mean schema-valid,
policy-pass, "some claims survived", or a coverage threshold. Anything else
raises `RenderRefused`, which carries the complete violation list in the
verifier's order. Failing claims are never filtered out so that the survivors
can be rendered.

**Why.** Mandatory coverage is a property of the claim set, not of individual
claims. Rendering a subset produces an apparently authoritative report that
silently omits facts the contract requires. This is the failure the rule-10 fix
closed, one level up: rule 10 stopped an invalid flag from hiding a missing
banner, and the gate stops a failing set from rendering as though it were
complete. Fail closed, consistent with the gate-3a HTTP 503 path.

A caveat on that last sentence: no HTTP 503 path exists anywhere under
`F:\Sleep_project`, across every text file. If the gate-3a serving path lives
in another repository, the consistency should be checked there.

### Signature: `render_report(result, packet)`

The explicit form consumes the result that was already computed. The renderer
never verifies, so there is one verification path.

- **The claims are not a separate argument.** The result already holds them,
  raw and enriched, and a second copy could disagree with what was verified.
  Rendering `result.enriched` makes "render what was verified" true by
  construction. That is the only reason this departs from the
  `render_report(claims, verify_result)` form in the brief.
- **`packet` stays an argument.** `VerifyReport` does not carry it, and the
  footer reads it. Adding it to `VerifyReport` would have changed the
  verification result object, which is outside a gate-only change. The
  residual risk: a caller could pair a result with another packet's footer,
  and nothing checks for that.
- **Only a `VerifyReport` is accepted.** A Layer 2 `PolicyResult` has the same
  two attributes but has skipped Layer 1. On a flag with a nonexistent
  `reason_key` it is clean while `verify_report` is not, so accepting it would
  have been a bypass. A bare list, the old signature, also raises `TypeError`.
- **Internal verification was not needed.** The existing API made the
  explicit form no worse.

### What the signature enforces, and what the guard can check

- **The signature enforces passing.** No caller can get a rendered report out
  of a result with a violation.
- **The AST guard answers only the syntactic question:** nothing under
  `report/` calls, imports or `getattr()`s `render_unverified`. A self-test
  shows it catches each of those forms. It does not check, and cannot, that
  `render_report` was handed an honestly produced result; that is dataflow. A
  hand-built `VerifyReport` with an empty violation list would pass the gate,
  and only tests build one.
- **`render_unverified(enriched_claims, packet)` is the named escape hatch.**
  It and `render_report` both call `_assemble`, the unchanged former body, so
  they are one renderer.

**`render_claim` is guarded too** (added the same day). It is public and
ungated, and it renders a single claim, so it is the same bypass as
`render_unverified` under another name. The guard now covers both names:

- Nothing under `report/` may use `render_unverified`. `render.py` stays
  checked for it, which is stricter than "outside `render.py`" and keeps the
  original promise.
- Nothing under `report/` outside `render.py` may use `render_claim`.
  `render.py` keeps that exemption because `_assemble` is built from
  `render_claim`. The test also asserts the exemption is still in use, so it
  cannot linger as an unused hole.

For each name, the self-test plants a direct call, an attribute call, an
import-as and a `getattr`.

### Known gap, not fixed: a result is not bound to its packet

Nothing checks that a `VerifyReport` was produced from the packet it is
rendered against. `render_report(result_for_A, packet_B)` would pair A's
verified claims with B's footer. Today the footer differs between packets only
in split and cohort size, so the visible damage would be small. The point is
that nothing would catch it.

The fix is a `recording_id` on the verification object, which `render_report`
would check against the packet. That changes the verification object, which is
outside the frozen tier, so it is deferred. **This is a candidate for the next
non-frozen window.**

### Callers migrated

- **`tests/golden/make_register_a.py` and `tests/test_no_leak.py`** go
  through `render_report` with a real, clean result.
- **`tests/test_render.py`:** 25 tests render.
  - 23 go through `render_report` only, each with a real, clean
    `VerifyReport`. That includes all 17 tests that existed before.
  - 2 also use `render_unverified`, both new and both by design. Test 4
    compares the two functions' bytes, which needs both. Test 4b deliberately
    renders a non-clean set to show what the gate stops.
  - No existing test was moved to the escape hatch.
- **`tests/test_adversarial.py`:** the 6 rule-10 tests deliberately inspect
  the verified subset of a non-clean set, which the gate now refuses. They
  read that subset through `render_unverified`, and they now also assert that
  `render_report` refuses.

### Tests added: ten, taking the suite from 278 to 288

| test | asserts |
|---|---|
| 1 | a set with one violation is refused although a claim survived; the exception carries `result.violations` |
| 1b | the refusal carries all four violations in order, not just the first |
| 2 | the oracle witness renders through the gate on all 60 packets, with a banner exactly on low nights |
| 3 | the empty array is refused on all 23 low nights, with codes `[missing_review_flag]` |
| 3b | the empty array renders on all 37 clean nights as a blank line then the footer, the same bytes as before |
| 4 | `render_unverified` and `render_report` produce the same bytes on 60 clean sets |
| 4b | the escape hatch is ungated: it renders the lone TST line that the gate refuses |
| type | a `PolicyResult` and a bare list are refused with `TypeError` |
| 5 | nothing under `report/` calls `render_unverified` |
| 5b | the guard catches a call, an attribute call, an import-as and a `getattr`, but not a definition |

### Nothing moved

- **Golden pins.** The golden script's dry run reports all six unchanged, and
  no golden file was touched.
- **Renders.** 180 renders were captured before the change: the witness, the
  witness plus the N1 flag, and the empty array, on all 60 packets.
  Recomputed afterwards, all 180 are byte-identical. 157 are clean and now
  render through the gate. The other 23, the empty array on a low night, are
  refused with exactly their one rule-10 violation.
- **Evaluator.** The 60-packet snapshot is byte-identical before and after.
  It covers the oracle, coverage, and `score_output` for the witness, the
  empty array and a missing file, plus the strata.
- **Test packets.** The md5, `050fffe46d035008d643435ee826dd92`, is
  unchanged.

---

## Reference model for 2F - setup (12 September 2026)

**Terminology.** "Reference model" is the term throughout the code and these
notes. The specific model is named in **one place only**: the
`reference model string` row below. A later change is then a one-line edit.

**Status: checked 12 September 2026, except the rate limits.** The key is in
the repository-root `.env` as `GEMINI_API_KEY`. `.env` is git-ignored
(`.gitignore` line 74) and has never been committed. Nothing in `report/` reads
it.

| fact | status |
|---|---|
| models the key reaches | `models.list` returns 55, of which 40 support `generateContent`. **Being listed does not mean the key can use it:** one of the two Pro models has a free-tier quota of zero, and the other is closed to new users (probe log below). **Flash-class only.** |
| active RPM / TPM / RPD | **The reference model: 5 RPM, 250K TPM, 20 RPD**, read from AI Studio's "Rate limits by model" page for this project on 12 September 2026 (full table below). This agrees with the API's own Pro 429, `generate_content_free_tier_requests, limit: 0, model: gemini-3.1-pro` |
| **reference model string** | **`gemini-3.8-flash`**: the highest-numbered Flash model listed, and one that answered. The response's `modelVersion` is `gemini-3.8-flash`. It is a thinking model, using 147 thought tokens on a trivial call. |
| structured output | **yes.** `responseMimeType: application/json` with a `responseSchema` returned `{"ok":true}`, which parses and matches the schema. This parallels GBNF, so the reference model and the local candidates are both constrained |
| connectivity check | `python reference_check.py call <model>` sends `POST https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent` with the key in the `x-goog-api-key` header. The body is the prompt "Reply with ok set to true.", temperature 0 and a one-field boolean schema. Result: HTTP 200, `finishReason: STOP`, 7 prompt tokens, 5 output tokens |
| Flash-class only? | **yes** |

Probe log, 12 September 2026. One trivial call each, strongest first. This is
a dated record of the calls made; it is not where the reference model is
named.

| call | HTTP | result |
|---|---|---|
| `gemini-3.1-pro-preview` | 429 | `RESOURCE_EXHAUSTED`: free-tier requests `limit: 0` for gemini-3.1-pro |
| `gemini-2.5-pro` | 404 | "no longer available to new users" |
| `gemini-3.8-flash` | 200 | `{"ok":true}`, schema-valid |

The `-latest` aliases were not considered. `gemini-pro-latest` reports itself
only as "Latest release of Gemini Pro", and a reference whose identity can
change under the same string cannot anchor a comparison.

**Active free-tier limits for this project**, as read from AI Studio's "Rate
limits by model" page on 12 September 2026. Text-generation models are shown.
Image, audio, TTS, live, embedding and agent models are omitted, because none
is a candidate for generating claims.

| model (AI Studio name) | RPM | TPM | RPD |
|---|---|---|---|
| Gemini 3.8 Flash | 5 | 250K | 20 |
| Gemini 3.7 Flash | 5 | 250K | 20 |
| Gemini 3.6 Flash | 5 | 250K | 20 |
| Gemini 3.5 Flash | 5 | 250K | 20 |
| Gemini 3 Flash | 5 | 250K | 20 |
| Gemini 2.5 Flash | 5 | 250K | 20 |
| Gemini 3.5 Flash Lite | 15 | 250K | 500 |
| Gemini 3.1 Flash Lite | 15 | 250K | 500 |
| Gemini 2.5 Flash Lite | 10 | 250K | 20 |
| Gemma 4 31B | 30 | 16K | 14.4K |
| Gemma 4 26B | 30 | 16K | 14.4K |
| Gemini 3.1 Pro | 0 | 0 | 0 |
| Gemini 2.5 Pro | 0 | 0 | 0 |
| Gemini 2 Flash, Gemini 2 Flash Lite | 0 | 0 | 0 |

**What 20 RPD means for 2F. This is recorded as a constraint, not a design.**
One pass over the 60 packets, at one request per packet, needs at least three
days of the reference model's quota. RPM and TPM do not bind at that rate. Any
design that samples each packet more than once multiplies the number of days.
The Flash Lite and Gemma rows have far more headroom, but they are weaker, so
choosing one would trade reference strength for throughput. That choice
belongs to 2F, and it would change the name in the reference-model row.

The AI Studio peak-usage column read 0 for every model over the last 28 days,
including the Flash model that answered the connectivity call above. The
dashboard appears to lag. The call itself is recorded above.

**Prepared, outside the repository.** A standard-library script,
`reference_check.py`:

- `list` enumerates the models the key can reach, with no generation;
- `call MODEL` makes one trivial call carrying a one-field `responseSchema`,
  which checks connectivity and structured output in the same call;
- the key is read from the environment and sent as the `x-goog-api-key`
  header, never in a URL.

**Framing, decided before any number exists.** If the key reaches only
Flash-class models, the comparison is "gap to a strong hosted model", not "gap
to a frontier model". A local 3B landing close to it would then be a finding,
not a disappointment. **This framing now applies: the key is Flash-class
only.**

**For the paper's methods section:**

> Reference claim sets were generated by a hosted model accessed through
> Google's free-tier Gemini API, whose terms permit the provider to use
> free-tier prompts and responses to improve its products. All evidence in
> those prompts was derived from the public Sleep-EDF Expanded dataset, so no
> participant privacy was at stake.

---

## `cited_coverage`: a diagnostic in the evaluator (12 September 2026)

```
cited_mandatory     = |distinct mandatory ids cited by ANY claim|     / 5
cited_discretionary = |distinct discretionary ids cited by ANY claim| / 14
```

- **What it counts.** Every claim counts, verified or not. An id cited twice
  counts once. A missing output cites nothing, so it scores 0.
- **Why it is split rather than pooled over 19.** Pooled, a high number could
  hide that the misses were the mandatory ones. Citing all 14 discretionary
  items and no mandatory one would read 14/19 (tested).
- **What it separates.** When cited coverage is high and verified coverage
  low, the model found the evidence and mis-shaped the claim. When both are
  low, it did not find the evidence. The failures differ, and so do the
  fixes. Until now the distinction existed only because one output had been
  traced by hand.
- **Where it appears:**
  - on each `OutputResult`;
  - in the per-recording JSON;
  - in `stratum_metrics`, overall and per tier, with n and the small-n flag;
  - in two table rows marked `(diag.)`, with a legend line.

**It is diagnostic only.** It does not alter policy pass, oracle recovery,
mandatory coverage, numeric fidelity, or any existing metric:

- The 60-packet evaluator snapshot, with the two new keys stripped, is
  **byte-identical** before and after the addition. Those two keys are the
  only ones added.
- `test_no_other_metric_reads_it` overwrites both fields on every result, and
  every other stratum figure stays the same.
- Cited coverage always bounds verified coverage from above. That is tested
  over five claim-set shapes.

**One existing test was made stale by the addition.** In
`test_the_worked_example`, a `SimpleNamespace` stand-in for an output result
lacked the new attributes, so `stratum_metrics` raised `AttributeError`. The
stand-in now carries `cited_mandatory=None, cited_discretionary=None`, meaning
no data. No assertion in the test changed.

**Tests added: eight, taking the suite from 288 to 296.** They are in
`tests/test_evaluate.py`, class `TestCitedCoverage`.

---

## EXPLORATORY, NOT A MEASUREMENT: runs A and B on SC4111E0 (12 September 2026)

> One packet, one model, one seed per run, with no API quota used. Three runs
> on one packet can direct which version of 2F gets written. They cannot
> settle a prompt format: adopting one would need confirmation across several
> dev packets first. Nothing here enters a selection decision.

### What was run

**Shared by all three runs:** Qwen2.5-1.5B-Instruct Q4_K_M, dev SC4111E0 (low
tier), the unchanged `claims.gbnf`, and `--temp 0 --seed 0 -n 3000 -c 12288`.

| run | what differs | prompt tokens | generated |
|---|---|---|---|
| raw | the step-5 output, raw completion (`-no-cnv`) | 1,007 | 1,076 |
| **A** | only change: `build_prompt`'s output placed inside the model's chat template | 1,036 | 353 |
| **B** | A plus two worked examples | 1,303 | 175 |

**Run A's invocation.** In llama.cpp b10927 the chat template comes from
`--jinja -cnv -st`, and that works alongside `--grammar-file`: the output is
constrained JSON and the process exits after one turn.

```
llama-completion -m <qwen> --jinja -cnv -st -f prompt.txt --grammar-file report/claims.gbnf -n 3000 -c 12288 --temp 0 --seed 0 --no-display-prompt
```

- `--jinja` applies the GGUF's embedded template, which is ChatML
  (`<|im_start|>` ... `<|im_end|>`).
- `-cnv` puts the prompt in a user turn.
- `-st` ends the run after that turn, without waiting for input.
- No `-sys` was given, so the template's own default applies.

**Run B's examples** come from a different dev packet, SC4081E0 (high tier).
They are a `value` claim on `arch.total_sleep_time` (410.0) and a
`hedged_value` on `arch.sleep_onset_latency` (28.5). Both passed full
verification against SC4081E0, singly and together, before use. They are
inserted between the rules and the target's evidence, under a heading saying
they show format and claim-type choice only.

**Where Run B departs from the brief, and why.** All 60 packets share one
evidence-ID set. That had two consequences:

1. The requested instruction, that "no evidence ID from the example packet may
   appear", would have forbidden the target's own items, including the
   mandatory `arch.total_sleep_time`. The prompt instead forbids any value or
   `text_key` from the examples, and explains that ids recur across
   recordings while their values differ.
2. An ID leak check is empty by construction, because `cited_id_not_in_packet`
   cannot fire for an example id. The leak check was run on values instead.
   Both example values differ from the target's (433.5 and 29.5), so a copied
   value would be caught.

### Results

| figure | raw | A | B |
|---|---|---|---|
| claims | 18 | 5 | 3 |
| claim types | value 11, observation 6, flag 1 | value 2, observation 2, flag 1 | value 1, **hedged_value 1**, flag 1 |
| `hedged_value` used | 0 | 0 | **1** |
| violations, per rule | 9 unsafe_item_not_hedged, 7 text_key_dependency_missing, 1 missing_review_flag | 2 text_key_dependency_missing, 1 rem_latency_double_count | **none** |
| claims verified | 2 / 18 | 3 / 5 | 3 / 3 |
| `mandatory_coverage` | 2/5: TST, TIB | **3/5**: TST, TIB, night.confidence | 2/5: TST, night.confidence |
| `cited_mandatory` | 4/5 | 4/5 | 2/5 |
| `discretionary_coverage` | 0/14 | 0/14 | 1/14 (SOL) |
| `cited_discretionary` | 14/14 | 8/14 | 1/14 |
| `oracle_recovery` | 0.0 | 0.0 | 0.071 |
| `numeric_fidelity` | 11/11 | 2/2 | 2/2 |
| `render_report` | refused, 17 violations | refused, 3 violations | **renders** |

Cited versus verified, per item. **V** = verified, **R** = cited but rejected,
**–** = never cited.

| item | raw | A | B |
|---|---|---|---|
| `arch.total_sleep_time` (mandatory) | V | V | V |
| `arch.time_in_bed` (mandatory) | V | V | – |
| `arch.sleep_efficiency` (mandatory) | R: observation, N1 key | R: observation, N1 key | – |
| `night.confidence` (mandatory) | R: flag, N1 reason | **V**: flag, `low_night_confidence` | V |
| `model.n1_reliability_warning` (mandatory) | – | – | – |
| `arch.sleep_onset_latency` | R: bare value | R: in c4 | **V**: hedged |
| `arch.waso` | R: bare value | – | – |
| 7 more `arch.*` items | R: bare values | R: all in c4 | – |
| 5 stage fractions | R: observation, N1 key | – | – |

"c4" in Run A is one observation (`tier_is_low`) citing eight `arch.*` items
at once. It fails rule 7, because it lacks `night.confidence`, and rule 8,
because it cites both REM latencies.

**Leakage in Run B:**

- **Example-packet ids not in the target:** none. This is impossible by
  construction, as explained above.
- **Example values:** none. No claim carries 410.0 or 28.5.
- **What did carry over is the examples' selection.** B's only two numeric
  claims cite exactly the two example ids, with the target's values, and
  nothing else numeric. Cited coverage fell to 2/5 and 1/14, both low, so B
  did not attempt the rest of the evidence.

### What this shows, for framing 2F only

- **The failure moves with the prompt.**
  - *Raw* is broad and mis-shaped: it cited 18 of 19 items and verified 2.
  - *A* got the review flag right, satisfying rule 10 and reaching 3/5
    mandatory. But it narrowed to 5 claims and lumped eight items into one
    observation.
  - *B* produced the first `hedged_value` of any run and a fully clean,
    renderable report. But it covered only the examples' two items plus the
    flag.
- **No run reached 5/5.** No run ever cited
  `model.n1_reliability_warning`.
- **The model can do each part, but no single prompt got all of them.** It can
  locate the evidence (raw, 18/19 cited). It can emit each correct claim shape
  (B's hedged value, A's and B's flag). On one packet at one seed, no single
  prompt did both at once. That says the prompt matters a great deal; it does
  not show the prompt is sufficient.
- **A candidate mechanism for B's narrowing, not tested:** an examples block
  of exactly two claims may anchor the output's scope.

---

## EXPLORATORY, NOT A MEASUREMENT: the N1 inspection, runs B′ and C (13 September 2026)

> The same terms as runs A and B apply: one packet (dev SC4111E0), one model,
> one seed per run, local only. With B′ and C there are five generations on
> this packet, which is already more than one packet supports. **No further
> prompt variants are run here.** The next information comes from breadth, the
> 31-packet dev population. Nothing here selects a prompt format, a model or K.

**What the five runs establish, stated no more strongly than this:**

- The failures are sensitive to how the prompt is worded.
- The model demonstrates each required behaviour somewhere across the runs.
- No run has shown those behaviours combining.

Whether the residual cause is prompt design, instruction-following, context
competition, decoding, model capacity or an interaction is **not yet
determined**.

### Item 1: the N1 item, inspected without generation

`model.n1_reliability_warning` was never cited in raw, A or B. It was the only
mandatory item that nothing had reached. The prompt inspected is byte-identical
to the one the model saw: `build_prompt(SC4111E0)`, prompt version 2D.1.

| question | finding |
|---|---|
| serialized line | `model.n1_reliability_warning \| N1 is low-reliability in this model \| "low" \| tier \| true` |
| fields as serialized | label "N1 is low-reliability in this model"; value `"low"`, a quoted string; unit `tier`; `safe_to_assert` true |
| position | 19 of 19: the last evidence line, directly after `night.confidence` |
| the other 18 items | 17 carry a numeric value; 1 carries a string (`night.confidence`, `"low"`) |
| is a non-numeric value still claimable? | **Yes.** The prompt names both tier items as the only non-numeric ones, and its coverage rule reads: "Every item with safe_to_assert true must be covered: numeric ones by a value claim, tier items by an observation or review_flag." |
| legal claim shapes, and are they reachable? | An `observation` with text_key `n1_reliability_is_low`, or a `review_flag` with reason_key `n1_low_reliability`, each citing this id. `value` and `hedged_value` are illegal, because they take numeric items only. The prompt's key tables spell out both keys as "cite model.n1_reliability_warning". **Reachable as written.** |

**Correct as it stands, so the serializer is left alone.** That moves the miss
from presentation to model behaviour, and the runs show which behaviour. The
model *uses* the key: `n1_reliability_is_low` appears in raw (six
observations, plus the flag reason `n1_low_reliability`), in A, in B′ and in C.
It is always attached to the wrong evidence and never to
`model.n1_reliability_warning`. The model has the key but does not bind it to
the item the key requires.

The inspection surfaced two related facts, recorded because they bear on C:

- The prompt already pairs `hedged_value` with `safe_to_assert: false` in its
  claim-type list, but only as a description ("use when"), not as a rule.
- It says an observation "cites 1 or more items", which licensed A's
  eight-item bundle.

### Item 2: run B′, with only the example IDs changed

**The hypothesis:** B's examples anchored *which items to report*, not only
how to shape a claim.

**How B′ differs from B.** Only the two example claims changed:

- **In B:** `arch.total_sleep_time` (410.0) and `arch.sleep_onset_latency`
  (28.5).
- **In B′:** `arch.time_in_bed` (537.0) and `arch.waso` (98.5).

Both new items are in minutes, like B's pair, so the ids and values are the
only change. The prompt builder asserts that B and B′ differ in exactly those
two lines. Everything else stayed the same:

- The source packet is still SC4081E0.
- Both examples verified clean against it, singly and together.
- Both values differ from the target's (464.0 and 1.0).
- B cited neither of the new ids.

**Verdict: selection anchoring confirmed, for this one generation.**

- B′'s only two numeric claims cite exactly the new example ids:
  `arch.time_in_bed` as a `value` and `arch.waso` as a `hedged_value`, both
  with the target's values.
- Neither of B's example ids appears in a numeric claim.
  `arch.total_sleep_time`, cited by every earlier run, appears only inside a
  mis-shaped observation.
- No example value leaked.

This is one generation per condition, so it is not over-read.

### Item 3: run C, with explicit claim-shape rules and no demonstrations

C is A plus the block below, placed where B put its examples, between the
rules and the evidence. It names no evidence id and no key (asserted). Its
second rule acknowledges the schema's "1 or more" for observations instead of
contradicting it silently.

```
CLAIM-SHAPE RULES
- Every evidence item whose safe_to_assert is false must be claimed with claim_type hedged_value, never value.
- Each claim cites exactly one evidence item. An observation may cite more than one item, but in this report every claim, observations included, cites exactly one.
```

**Result.** C reached the best mandatory coverage of any run, 4/5: every
mandatory item except the N1 warning. Neither rule held as intended:

- **No `hedged_value` at all.** No unsafe item went into a `value` claim, which
  is the first rule's letter. But instead of being hedged, 13 of the 14
  discretionary items were bundled into one observation.
- **That observation cites 13 items,** against the second rule. It also
  carries the N1 key and cites both REM latencies, which is 2 violations.

### Scoring: B′ and C

**Per-rule breakdown.**

| run | violations |
|---|---|
| B′ | 1 × `L2.text_key_dependency_missing`, on c3: an observation on total sleep time carrying the N1 key |
| C | 1 × `L2.text_key_dependency_missing` and 1 × `L2.rem_latency_double_count`, both on c4, the 13-item observation |

**Cited versus verified, per item.** V = verified, R = cited but rejected,
– = never cited.

| item | B′ | C |
|---|---|---|
| `arch.total_sleep_time` (mandatory) | R: observation, N1 key | V |
| `arch.time_in_bed` (mandatory) | V | V |
| `arch.sleep_efficiency` (mandatory) | – | **V**, the first verified claim on it in any run |
| `night.confidence` (mandatory) | V: low-night flag | V: low-night flag |
| `model.n1_reliability_warning` (mandatory) | – | – |
| `arch.waso` | V: hedged | – |
| the other 13 discretionary items | – | R: all in c4 |

### The five runs together

All five used dev SC4111E0, Qwen2.5-1.5B-Instruct Q4_K_M, `claims.gbnf`, temp
0 and seed 0. **Exploratory.**

| run | prompt | claims | violations | mandatory verified | cited mand. | disc. verified | cited disc. | `hedged_value` | numeric fidelity | renders |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| raw | raw completion | 18 | 17 | 2/5: TST, TIB | 4/5 | 0/14 | 14/14 | 0 | 11/11 | no |
| A | chat template | 5 | 3 | 3/5: TST, TIB, NC | 4/5 | 0/14 | 8/14 | 0 | 2/2 | no |
| B | A + two examples (TST, SOL) | 3 | 0 | 2/5: TST, NC | 2/5 | 1/14 | 1/14 | 1 | 2/2 | **yes** |
| B′ | A + two examples (TIB, WASO) | 4 | 1 | 2/5: TIB, NC | 3/5 | 1/14 | 1/14 | 1 | 2/2 | no |
| C | A + claim-shape rules | 5 | 2 | **4/5**: TST, TIB, SE, NC | 4/5 | 0/14 | 13/14 | 0 | 3/3 | no |

Key: TST = total sleep time, TIB = time in bed, SE = sleep efficiency, NC =
night confidence, SOL = sleep onset latency, WASO = wake after sleep onset.
Across all five runs, every number was transcribed exactly (19 of 19 numeric
claims). No run cited `model.n1_reliability_warning`.

### Reading B correctly

Operationally, a clean but incomplete report is the dangerous case: it reads as
complete and is not. B was exactly that. It had zero violations, `render_report`
rendered it, and it carried two of the five mandatory items.

Scientifically, B is a valuable result. **It demonstrates that a model can
produce a zero-violation subset while failing completeness.** That is exactly
why this architecture carries coverage alongside verification rather than
verification alone.

This bears on the gate. `render_report`'s condition is zero violations, as
specified, so a B-shaped output renders. Completeness is measured by the
evaluator's coverage; it is not enforced at render time. This is recorded, not
changed.

---

## 2F Item 0: structured-output preflight (13 September 2026)

**This is a gate. No request carrying packet content has been sent.**
Everything below either ran locally or sent a synthetic prompt: a packet built
from the schema's own vocabulary with invented values. None of its 19 evidence
lines matches a line from any real packet.

### 0a: the schema, generated rather than written by hand

`reference/schema.py` builds the reference model's `responseSchema` from the
same constants as `claims.gbnf`:

- from `claim_schema.py`: `CLAIM_TYPES` (field sets, `cites` arity, subject),
  `BASE_FIELDS` (field order), `CLAIM_ID_RE`, `TEXT_KEYS` and `REASON_KEYS`;
- from `gbnf.py`: `NUMERIC_IDS`, `TIER_IDS` and `NUMERIC_UNITS`.

**Which ids each claim type may cite** is the one mapping the constants do not
carry, so it lives in `CITE_SCOPE`, a five-row table mirroring the grammar's
three cite rules: numeric ids, tier ids, or any id. It is guarded two ways:

- it fails loudly if a claim type is added without a scope;
- it is checked against the grammar for all 95 (type, id) pairs.

The committed `reference/response_schema.json` is asserted to be
byte-identical to the generator's output.

**Its shape.** A top-level `ARRAY` whose items are `anyOf` five `OBJECT`
branches, one per claim type. Each branch has:

- `claim_type` as a one-value enum;
- its own `properties`, all of them `required`;
- a `propertyOrdering` in the grammar's field order.

**Nothing frozen was modified.** `claim_schema.py`, `claims.gbnf`, the
verifier, the evaluator and the packet format are untouched. The reference tier
lives in a new top-level `reference/` package, where the model string is named
in one constant (`reference/config.py`), and a test fails if anything under
`report/` imports it.

### 0b: the constraint mapping

| constraint | `claims.gbnf` (local candidates) | `responseSchema` (reference model) | bucket |
|---|---|---|---|
| top-level array | `root ::= "[" ... "]"`, empty allowed | `type: ARRAY`, no `minItems` | **matches** |
| allowed claim types | a union of 5 per-type productions | `anyOf` of 5 branches, each with a one-value `claim_type` enum | **matches** |
| per-type field set (presence) | each production's fixed field sequence | each branch's `properties` | **matches** |
| required vs optional | every field required | every field in `required` | **matches** |
| field order | fixed by the production | `propertyOrdering` | **matches** as generation order; no layer treats order as meaningful |
| `claim_id` syntax | `"c" digits` | `pattern: ^c[0-9]+$` | **matches** as declared, and accepted by the API; enforcement during decoding is the provider's claim, not verified here |
| `cites` cardinality per type | exactly 1 for value, hedged_value and review_flag; 1 or more for observation and population_association | `minItems` / `maxItems` from `cites_min` / `cites_max` | **matches** |
| `cites` vocabulary per type | 17 numeric / 2 tier / all 19 ids | per-branch `items.enum` from the same tuples | **matches** |
| `text_key` enum | 4 alternatives | `enum` of the same 4 | **matches** |
| `reason_key` enum | 2 alternatives | `enum` of the same 2 | **matches** |
| `subject` per type | the literal `this_recording` or `population` | a one-value `enum` | **matches** |
| `unit` enum | the 5 numeric units | `enum` of the same 5 | **matches** |
| `value` type | a number, with no exponent | `NUMBER`, exponent allowed | **weaker, harmless**: the same float after parsing |
| type-specific field exclusion (a `value` claim with no `text_key`) | ungeneratable: nothing outside the production can be emitted | **not expressible.** The dialect cannot say "no other properties". That the decoder emits only declared properties is provider behaviour, not a declared constraint | **weaker**: Layer 1's unknown-field check enforces it |
| `claim_id` uniqueness | not expressible | not expressible | **neither**: Layer 1 |
| a cited id is in *this* packet | – | – | **neither**: Layer 2, rule 1 |
| value equals the packet's value | – | – | **neither**: rule 2 |
| unit matches the cited item | – | – | **neither**: rule 3 |
| safe/unsafe decides value vs hedged_value | – | – | **neither**: rules 4 and 5 |
| a key's evidence dependency and predicate | – | – | **neither**: rule 7 |
| both REM latencies in one claim | – | – | **neither**: rule 8 |
| rules 6, 9, 10, 11 and 12 | – | – | **neither**: Layer 2 |

Rule 13 (subject matches type) and "no value claim on a tier item" are
enforced by both mechanisms and re-checked by the verifier anyway. No Layer 2
policy was put into the schema to close a gap: a `value` claim may cite any of
the 17 numeric ids, unsafe ones included, exactly as the grammar allows
(tested). The local GBNF was not weakened to match.

> The local candidates generated under GBNF constraints derived from the frozen
> claim grammar, while the reference model used the provider's structured-output
> schema. Where that mechanism could not express equivalent production-level
> constraints, semantic equivalence was enforced downstream by the common
> deterministic verifier.

### 0d: local validation, zero quota

`tests/test_reference_schema.py` holds 15 tests:

- the committed schema equals the generator's output;
- every constraint traces to a schema constant;
- no Layer 2 policy is encoded;
- `report/` never imports `reference/`;
- grammar and schema agree on all 95 (type, id) pairs and on every unit and
  key; on cites cardinality (including empty cites); on the closed-world
  rejections (unknown type, uncitable object, wrong subject, bad `claim_id`,
  string value, missing field); and on the empty array;
- each of the four documented asymmetries is asserted together with the check
  that catches it;
- every oracle witness on all 60 packets validates against the schema.

**The API accepted the full schema**, including `anyOf`, `pattern`,
`propertyOrdering` and `minItems`/`maxItems`: HTTP 200 on every completed
call. All 39 claims in the two complete responses conformed to it.

### 0c: thinking tokens

Three completed calls, all on the same synthetic prompt of 1,126 tokens (by
the provider's count), with `temperature` 0, `seed` 0 and `maxOutputTokens`
8,192:

| `thinkingConfig` | HTTP (attempt) | thinking tokens | output tokens | total | finish | output |
|---|---|---:|---:|---:|---|---|
| unset (the model's default) | 200 (after one 503) | **7,860** | 318 | 9,304 | **MAX_TOKENS** | truncated. The 5 claims before the cut were well-shaped; the JSON as a whole did not parse |
| `thinkingBudget: 0` | 200 (attempt 3) | none reported | 1,457 | 2,583 | STOP | 20 claims; schema-valid; Layer 1 clean |
| `thinkingBudget: 512` | 200 (attempt 4) | none reported | 1,387 | 2,513 | STOP | 19 claims; schema-valid; Layer 1 clean |

**1. Can thinking be disabled or budgeted?**

- **Disabled: yes.** `thinkingBudget: 0` is accepted, and no thinking tokens
  were used.
- **Budgeted: accepted, and respected.** Under a 512-token budget, thinking
  stayed within it, at zero. One call cannot tell whether a small budget acts
  as a ceiling the model chose not to reach or suppresses thinking outright.
- `thinkingLevel` was not tried, to save quota.

**2. Does structured output work with thinking on, off, or both?**

- **Off: yes.** Both budgeted calls produced complete, schema-valid,
  Layer-1-clean output.
- **On (the default): the structure held up to the cut.** But thinking
  consumed 7,860 of the 8,192 `maxOutputTokens`, which truncated the answer.
  **Thinking tokens count against `maxOutputTokens`.**

**3. What does thinking cost at realistic length?**

- At the default, about 7.9K thinking tokens per request: roughly 9.3K tokens
  in total, 3.6 times the cost without thinking.
- Latency was about 24 s, against 5–6 s with thinking off.
- At 5 RPM that is about 47K TPM, far under the 250K limit. **TPM does not
  bind; RPD does.**

**Availability.** 10 of the 13 attempts returned HTTP 503 UNAVAILABLE ("high
demand") over about 15 minutes, and the retry client absorbed all of them. It
is not known whether 503s count against the daily request quota. The calls ran
around 20:00 UTC, which is still 12 September on Pacific time, the day the
quota is counted on.

### What Item 2 needs decided first

These decisions are not taken here.

- **Thinking on or off.** The local candidates do not think: Qwen2.5 has no
  reasoning mode. Thinking *off* makes the reference model decode the way the
  candidates do. Thinking *on* measures the reference model at its default
  capability, which is arguably what "can a strong hosted model satisfy the
  contract" asks. Either is defensible, but it has to be fixed and recorded
  before the first packet is sent.
- **If thinking is on, `maxOutputTokens` must rise well above 8,192** (the
  model allows 65,536). Otherwise responses risk the default call's
  truncation. A truncated response is an unusable response, and under the
  retry policy it is not retried.
- **Temperature.** The preflight used 0, with seed 0, to mirror the local
  runs. Google's Gemini 3 guide recommends leaving temperature at its default
  of 1.0, warning that lower values can degrade output; whether that applies
  to this model is unverified.

### Does the asymmetry materially affect the comparison?

**No, for the policy and coverage comparison.** The two weaker constraints,
field exclusion and number format, can only produce one of two things:

- an output that Layer 1 rejects (an undeclared field), or
- an output that parses to the same value (an exponent).

So the asymmetry can lower the reference model's schema-validity rate. It
cannot raise any of its policy or coverage figures, and so it cannot flatter
the reference model. If the provider did not enforce `pattern` or `anyOf`
during decoding, the effect would run the same way: Layer 1 catches a bad
`claim_id` or a malformed branch.

Any effect would therefore show up in one direction only, in schema validity,
which is reported separately with its denominator. In the synthetic responses,
no undeclared field appeared.

---

## 2F Item 2: the runner is built, and failed attempts cost quota (13 September 2026)

**Nothing has been generated. No packet has been sent to the reference model.**

### 2a: failed attempts count against the daily quota

On 12 September (Pacific) the request log holds 13 attempts to the reference
model: 3 returned HTTP 200 and 10 returned HTTP 503. AI Studio's rate-limit
page shows **13–14 consumed** for that day. The 14th would be the connectivity
check, if it fell on the same Pacific day. **A 503 costs a request.**

**The consequence.** At the preflight's failure rate, 10 of 13, a day's 20
requests could yield as few as 5 responses. So the four-day plan does not hold.

**The plan now:**

- P1 alone across all 31 dev packets first, with P2 on later days.
- The runner is prompt-major, so that order is its default.
- Its daily budget counts every attempt, which is also its default.
  `--successes-only` exists only for the case 2a ruled out, and must not be
  used.
- A session stops after one packet exhausts its 4 attempts. A bad hour of
  provider availability therefore costs at most 4 requests before the run
  halts, rather than draining the day.

### 2b: generation settings, fixed before any packet is sent

| setting | value |
|---|---|
| thinking | off (`thinkingBudget: 0`) |
| `maxOutputTokens` | 16,384 |
| temperature | 0 |
| seed | 0 |
| structured output | on, using the Item 0 schema |

> The reference model was run with thinking disabled, matching the local
> candidates' generation regime. The reference therefore measures the ceiling
> for comparable single-pass generation rather than the provider's maximum
> capability.

> Unlike llama.cpp, the provider does not guarantee bit-identical output at
> temperature 0, so reference generations are not reproducible in the way the
> local runs are.

### 2c: the two prompts

Both wrap the frozen `build_prompt`. Each inserts one claim-shape block between
the rules and the evidence, as run C did. Neither contains a demonstration, an
evidence id or a key.

**P1 is run C's block, verbatim.**

**A premise correction on P2.** P2 was framed as "C with the obligation stated
positively". But C's hedging line already reads "must be claimed with
claim_type hedged_value, never value": the positive obligation plus the
prohibition. So a positive-versus-negative contrast was not available. What C
actually did was cover the obligation by bundling 13 items into one
observation. P2 therefore makes each unsafe item's own hedged claim explicit
(chosen 13 September 2026).

**The diff between P1 and P2** is one line, asserted by
`tests/test_reference_run.py` on the full prompt text:

```
P1: - Every evidence item whose safe_to_assert is false must be claimed with claim_type hedged_value, never value.
P2: - Every evidence item whose safe_to_assert is false must be claimed, each in its own hedged_value claim.
```

### 2d: the runner, built and tested before any live request

The runner is `reference/run.py`, and the scorer is `reference/score.py`.

- **Order.** Prompt-major: the 31 dev packets in manifest order under P1, then
  under P2. A manifest row that is not dev, or that appears in the test
  manifest, stops the run.
- **Cache.** One file per HTTP 200, keyed by `(recording_id, prompt_id,
  prompt_hash, model)` and written atomically. A cached key is never
  requested again. A changed prompt or model misses the cache.
- **Budget.** Before every attempt, retries included, the runner counts the
  attempts made on the current Pacific date across every request log. At the
  limit it stops before sending. A packet cut off mid-retry stays uncached
  and becomes the next session's first job.
- **Logging.** Every attempt records the timestamp, packet, prompt id, prompt
  hash, attempt number, retry flag, retry cause, HTTP status, token counts
  (prompt, output and thinking) and latency. The record is written before the
  response is processed.
- **Pacing and retries.** Attempts are at least 12.5 s apart (5 RPM). Only
  429, 5xx, timeouts and connection errors are retried, with exponential
  backoff and jitter, up to 4 attempts. Any other non-200 stops the run for
  inspection. A completed response is cached as it is and never retried.
- **Tests.** 19 offline tests, all against a fake transport, cover every
  point above, plus the Pacific-date boundaries across DST and the scoring
  path: 5/5 counts, and a reference set built from passing responses only.

**To run,** once per Pacific day (the quota resets at 07:00 UTC):
`python -m reference.run --go`. Then score with `python -m reference.score`.

---

## 2F: the unattended launch, the thinking check, the Flash-Lite pilot (13 September 2026)

### Step 1: what started the 21:07 UTC run

A `reference.run --go` began at 21:07:06 UTC on 12 September and spent the
day's remaining 7 attempts. Its seven log records carried request data only,
with no process id, working directory or caller. The other traces identify it:

- **Shell history.** PowerShell's interactive history (`PSReadLine`,
  `ConsoleHost_history.txt`) ends with `python -m reference.run --go`, and the
  file was last written at 21:07:02 UTC, four seconds before the first logged
  attempt. PSReadLine records only interactive consoles; this session's shell
  tools run non-interactively. The lines just before it are `pandoc` builds of
  the paper (`sn-article.tex`), in the same terminal.
- **The IDE.** Antigravity's extension log records `[Terminal] Command
  completed: python -m reference.run --go exit code 0` at 21:09:55 UTC. The
  command ran in **Antigravity's integrated PowerShell terminal**.
- **Not Antigravity's agent.** Its language server failed every second with
  "certificate has expired" TLS errors throughout the window, and the log
  holds no agent or tool-call entry.
- **Not a Claude session.** No transcript in this project has a tool call that
  executed `--go`. The Claude Code log shows this session running only the
  dry-run form, at 20:46:52 and 21:10:14 UTC.
- **Not scheduled or triggered.** No Task Scheduler entry, watcher process,
  IDE run configuration, task-runner file or git hook refers to it. No test
  reaches the live path: every `Runner` built in the tests is given a fake
  transport.

**Cause: a manual invocation, typed or recalled in Antigravity's integrated
terminal.** A cause was found, so per the instruction no live-run guard was
added. What changed is that every attempt now logs its caller (pid, parent
pid, argv, cwd), so the log alone identifies what started any future run.

### Step 2: the three cached responses ran with thinking off

| request | prompt tokens | output tokens | thinking tokens | total | prompt + output = total |
|---|---:|---:|---|---:|---|
| P1/SC4081E0 | 1,171 | 1,586 | none reported | 2,757 | yes |
| P1/SC4082E0 | 1,169 | 1,393 | none reported | 2,562 | yes |
| P1/SC4111E0 | 1,165 | 1,391 | none reported | 2,556 | yes |

All three are comparable to the rest of the run.

**The resolved request,** printed from the objects the runner sends rather
than from its intended settings (`python -m reference.run --show-request`):

- `POST .../models/gemini-3.8-flash:generateContent`
- `generationConfig`: `{"maxOutputTokens": 16384, "temperature": 0.0, "seed": 0,
  "responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 0}}`
- a `responseSchema` byte-identical to `reference/response_schema.json`

A test asserts the same body.

**The daily budget now counts per model,** because quotas are per model. A
candidate model's attempts can never spend the reference model's budget
(tested).

### Step 3: the Flash-Lite pilot (`gemini-3.5-flash-lite`)

The pilot is exploratory. It selects a reference model; it is not a
measurement, and nothing from it enters the reference set.

**The zero-cost check.** The metadata reports `thinking: true` and supports
`generateContent`. `countTokens` accepted the full configuration (HTTP 200).
That proved insufficient, because `countTokens` does not validate generation
settings.

**The pilot.** P1 on SC4111E0, at the fixed settings, returned **HTTP 400
INVALID_ARGUMENT** on its first attempt. It was not retried, which is correct
for a 400. Nothing was cached, so nothing was scored.

**Diagnosis,** from six synthetic calls with no packet content:

| configuration | HTTP |
|---|---|
| the full fixed configuration | 400 |
| without `thinkingConfig` | **200** |
| without `seed` | 400 |
| without `responseSchema` | 400 |
| `thinkingLevel: "minimal"` in place of `thinkingBudget: 0` | **200** |
| `thinkingLevel: "low"` in place of `thinkingBudget: 0` | **200** |

**The rejected setting is `thinkingBudget: 0`.**

- Flash-Lite **supports `responseSchema`.** The full Item 0 schema, with seed
  and temperature 0, was accepted.
- It **does not accept a zero thinking budget.** It does accept
  `thinkingLevel` "minimal" and "low".
- None of the accepted calls reported thinking tokens, but on a trivial
  synthetic prompt that says nothing about a real one.

**So the option is not dead, but it cannot run at the reference run's fixed
settings.** Running it at `thinkingLevel: "minimal"` would change a fixed
setting for one model, which is a decision not taken here.

**Availability.** 7 Flash-Lite attempts: 4 returned 400, 3 returned 200, and
**none returned 503**. Seven attempts, six of them trivial, are far too few to
estimate a failure rate.

---

## 2F: the plan after the pilot (13 September 2026)

**The reference model stays Gemini 3.8 Flash, at the fixed settings.**

**Flash-Lite was investigated and rejected.** It refuses `thinkingBudget: 0`
(HTTP 400). The nearest setting it accepts, `thinkingLevel: "minimal"`, is not
zero, so the reference model would reason where no local candidate does, and
the gap measurement would absorb that difference. It would also be a third
asymmetry, stacked on the weaker structured-output constraint and the
non-reproducible temperature 0, against candidates that have none of them.
Each is defensible alone; together they erode what the comparison means. This
is a finding about free-tier constraints in its own right: **the free tier's
higher-quota model cannot be run with reasoning fully off.**

**P1 only, stopping at 31.** P1 across the 31 dev packets answers the 2F
question. P2, the prompt-design comparison, is held in reserve. The decision
rule is fixed now, before any results:

- **25 or more of 31 at 5/5 mandatory:** the contract is satisfiable. The
  question is answered, P2 is unnecessary, and the work moves to 2G.
- **Near zero:** the contract is the ceiling. That is the finding, and P2
  would not change it.
- **Mid-range, around 12 of 31:** P2 earns its days, because prompt design
  plausibly explains the gap.

The runner's default schedule is now P1 only (`SCHEDULED_PROMPTS`), so it
sends nothing once the 31 are cached. 28 remain. At the observed success rate
of about 30%, that is roughly five days.

**Launching.** A person starts each day's session, after 07:00 UTC, with
`python -m reference.run --go`. A timer does not, because a scheduled launch
would be exactly the unattended start the 21:07 investigation was about. If a
session stops as unavailable (4 consecutive 503s on one packet), the rest of
that day's budget is intact, and a later session the same day resumes where it
stopped.

---

## Deployment numbers for the five local candidates (13 September 2026, target named)

Resource-constrained evaluation was performed on x86 under
Windows, with the Raspberry Pi 5 (8 GB, 4 cores, LPDDR4X-4267) as
the named analytical target. Throughput figures are analytical
bandwidth-bound estimates derived from quantized tensor sizes
against an assumed effective-bandwidth range, not on-device
measurements. Peak resident memory and quantized model size are
measured directly, on the x86 evaluation host.

The conclusion this section supports is that the reporting
workload is plausible within a Raspberry Pi 5-class resource
envelope — not that the system was demonstrated to run on a
Raspberry Pi 5.

**Nothing was run on a Raspberry Pi 5, and nothing will be.** Every figure
below belongs to one of these kinds:

| figure | kind | where it comes from |
|---|---|---|
| throughput ceilings (tokens/s) and predicted generation time (prefill excluded) | **ANALYTICAL**: predictions of a model | quantized tensor sizes against the target's bandwidth, which is either its theoretical 17 GB/s or the **assumed** 7–10 GB/s effective range. Not measured, not timed |
| peak resident memory (peak working set, peak private bytes) | **MEASURED on the x86 evaluation host** | an Intel i7-11800H (8 cores, 16 threads, 15.7 GiB) running 64-bit Windows 11, with llama.cpp b10927 (CPU build). A Pi 5's allocator, page size and runtime build differ, so these are **not target measurements** |
| quantized file size | **MEASURED**, platform-independent | the GGUF file on disk |
| layer and head counts, head dimensions; KV cache per position; weights read per token | **read from GGUF headers, then derived**; platform-independent | `deploy/gguf.py`, `deploy/throughput.py` |

### The target and its bandwidth

**The target is the Raspberry Pi 5: 8 GB, 4 cores, LPDDR4X-4267.** Its
**theoretical** memory bandwidth is about 17 GB/s: 4,267 MT/s times 4 bytes
over its 32-bit interface, which is 17.07 GB/s.

**The analytical model's effective bandwidth is an ASSUMED range of 7–10
GB/s**, roughly 40–60% of theoretical. This is an assumption of the
simulation. It is never a point estimate, and it is not a measured or typical
property of the hardware. It is labelled as an assumption wherever it appears,
including in `deploy/throughput.py` and `deploy/results/throughput.json`.

**Why the cited benchmark is not the basis of the range.** The tinymembench
figure below is cited, but it is not used as the basis for 7–10 GB/s. It is
*copy* bandwidth, on a *pre-release* board, not read-only bandwidth, and
decode is read-dominated. So it neither produced the range nor tests it.

**What the published benchmarks do and do not say.** The one citable
memory benchmark found is tinymembench on a *pre-release* 8 GB Pi 5
([geerlingguy/sbc-reviews, issue #21](https://github.com/geerlingguy/sbc-reviews/issues/21)).
It reports copy throughput of 4,793.9–5,688.0 MB/s across its copy variants
(standard memcpy 4,805.4 MB/s) and fill of about 13,700 MB/s (standard memset
13,676.9 MB/s). It gives no read-only figure. Decoding is dominated by reads,
so none of these is the quantity the model needs, and the board was
pre-release. The benchmark is cited as context. **The 7–10 GB/s range stays a
declared assumption; it is not derived from the benchmark.**

### KV cache: verified from the headers

`attention.head_count` and `attention.head_count_kv` were read from all five
GGUF headers. Equal values mean no grouped-query attention (GQA). The cache
per position is derived as layers × `head_count_kv` × (key length + value
length) × 2 bytes (f16).

| model | arch | layers | `head_count` | `head_count_kv` | GQA | head dim (K/V) | KV per position | vs Qwen | weights read per token |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | qwen2 | 28 | 12 | 2 | yes | 128/128 | 28,672 B (28 KiB) | 1.00× | 934.7 MiB |
| SmolLM2-1.7B-Instruct | llama | 24 | 32 | 32 | **no** | 64/64 | 196,608 B (192 KiB) | **6.86×** | 1,005.0 MiB |
| Gemma-2-2b-it | gemma2 | 26 | 8 | 4 | yes | 256/256 | 106,496 B (104 KiB) | 3.71× | 1,623.7 MiB |
| Llama-3.2-3B-Instruct | llama | 28 | 24 | 8 | yes | 128/128 | 114,688 B (112 KiB) | 4.00× | 1,918.4 MiB |
| Phi-3.5-mini-instruct | phi3 | 32 | 32 | 32 | **no** | 96/96 | 393,216 B (384 KiB) | **13.71×** | 2,228.8 MiB |

**The headers confirm that SmolLM2 and Phi-3.5 lack grouped-query
attention.** Each has as many KV heads as attention heads (32 and 32). The
other three share KV heads across groups of 6, 2 and 3 query heads.

### Memory, measured on the x86 evaluation host

**The workload** is dev SC4111E0 through the frozen `build_prompt`, in each
model's chat template, under `claims.gbnf`, with a context of 4,096 tokens and
up to 512 generated tokens at temperature 0. Weights are memory-mapped
(llama.cpp's default). The peak is read exactly from the finished process with
`GetProcessMemoryInfo`, over one run per model.

| model | quantized file size | peak working set (host) | peak private (host) | peak working set ÷ file | peak working set against the target's 8 GB (8,192 MiB) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 1,065.6 MiB | 1,812.1 MiB | 1,168.7 MiB | 1.70× | 22% |
| SmolLM2-1.7B-Instruct | 1,006.7 MiB | 2,615.1 MiB | 1,682.3 MiB | **2.60×** | 32% |
| Gemma-2-2b-it | 1,629.4 MiB | 3,150.7 MiB | 2,023.0 MiB | 1.93× | 38% |
| Llama-3.2-3B-Instruct | 1,925.8 MiB | 3,840.4 MiB | 2,162.2 MiB | 1.99× | 47% |
| Phi-3.5-mini-instruct | 2,282.4 MiB | 5,103.5 MiB | 2,935.2 MiB | 2.24× | **62%** |

**The envelope comparison is analytical, not a demonstrated fit.** The
comparison is between peak memory on the evaluation host and the target's
8 GB.

- Phi-3.5, at about 5.1 GB, is **tight** against that envelope. This is
  particularly so because the student staging model runs alongside it on the
  same device. The staging model's memory was not measured here, and neither
  is the operating system's share.
- The other four leave more headroom on this comparison.

None of this shows that any model runs within 8 GB on a Pi 5, whose allocator
and page size differ.

### The finding: quantized size gives the wrong ranking for this workload

**Peak working set runs 1.70–2.60× the quantized file size** across the five
models, so file size alone understates what a device must hold. (An earlier
note in this session said 1.7–2.2×, which omitted SmolLM2's 2.60×. The table
above is the computed range.)

**Quantized size is the metric usually reported for edge-deployment
feasibility, and it gives the wrong ranking for this workload.** The reason is
the KV cache:

- **By file size,** SmolLM2 is the lightest of the five (1,006.7 MiB), just
  under Qwen (1,065.6 MiB).
- **By per-position cache** from the headers, SmolLM2 is second-worst: 192
  KiB, against 28 KiB for Qwen and 384 KiB for Phi-3.5.
- **By share of per-token memory traffic** at 2,400 cached positions, it is
  the worst: KV reads are 30.9% of what SmolLM2 reads per decoded token,
  against 28.8% for Phi-3.5, 13.1% for Gemma, 12.0% for Llama and 6.6% for
  Qwen.
- **The consequence:** once the cache is included, Qwen moves ahead of
  SmolLM2 on both measured peak memory (1,812.1 against 2,615.1 MiB) and
  predicted decode rate, reversing the file-size order at the top. Below the
  top two, the orders agree.

The file-size ranking would have picked the model with the second-largest
cache, the highest peak-to-file ratio (2.60×) and the largest cache share of
decode traffic.

### Predicted throughput (analytical)

Decode tokens/s ≤ B / (W + K·L), taken at L = 2,400 cached positions (a
~1,200-token prompt plus a ~1,200-token report). Every entry is a
**prediction** of the model. The general form is kept, so the analysis
survives a change of target; the Pi 5 rows are the named target.

| bandwidth | Qwen2.5-1.5B | SmolLM2-1.7B | Gemma-2-2b | Llama-3.2-3B | Phi-3.5-mini |
|---|---:|---:|---:|---:|---:|
| 10 GB/s (general form) | 9.5 | 6.5 | 5.1 | 4.4 | 3.0 |
| 25 GB/s (general form) | 23.8 | 16.4 | 12.8 | 10.9 | 7.6 |
| 50 GB/s (general form) | 47.7 | 32.8 | 25.5 | 21.9 | 15.2 |
| 100 GB/s (general form) | 95.3 | 65.5 | 51.1 | 43.7 | 30.5 |
| **Pi 5, 17 GB/s theoretical (ceiling)** | **16.2** | **11.1** | **8.7** | **7.4** | **5.2** |
| **Pi 5, ASSUMED 7–10 GB/s effective** | **6.7–9.5** | **4.6–6.5** | **3.6–5.1** | **3.1–4.4** | **2.1–3.0** |

**Under the assumed 7–10 GB/s effective-bandwidth envelope, the model predicts
approximately:**

| model | predicted decode rate |
|---|---:|
| Qwen2.5-1.5B | 6.7–9.5 tok/s |
| SmolLM2-1.7B | 4.6–6.5 tok/s |
| Gemma-2-2b | 3.6–5.1 tok/s |
| Llama-3.2-3B | 3.1–4.4 tok/s |
| Phi-3.5-mini | 2.1–3.0 tok/s |

**Even at the 17 GB/s theoretical ceiling, the model predicts no more than
5.2–16.2 tok/s.**

### Predicted generation time per report, prefill excluded (analytical)

One complete report is a ~1,200-token prompt followed by ~1,200 generated
tokens, so about 2,400 tokens in context by the end. **Predicted generation
time** is the sum of (W + K·t) / B over the 1,200 generated tokens, as the
cache grows from 1,200 to 2,400 positions.

**These are floors, not report times.** Prefill, the prompt processing
before the first generated token, is compute-bound and not modelled. So
**end-to-end report time is strictly greater** than every figure in this
table.

| bandwidth | Qwen2.5-1.5B | SmolLM2-1.7B | Gemma-2-2b | Llama-3.2-3B | Phi-3.5-mini |
|---|---:|---:|---:|---:|---:|
| **Pi 5, 17 GB/s theoretical (ceiling): predicted generation time, prefill excluded** | 73 s | 99 s | 134 s | 157 s | 215 s |
| **Pi 5, ASSUMED 7–10 GB/s effective: predicted generation time, prefill excluded** | 124–177 s | 169–241 s | 227–325 s | 266–380 s | 365–522 s |

**Even under ideal bandwidth, the model predicts at least 73–215 s of
generation per report, with end-to-end time strictly greater once prefill is
added.** That is the stronger form of the argument,
since no effective bandwidth can beat the theoretical ceiling.

Under the assumed 7–10 GB/s envelope, it predicts approximately 2–9 minutes of
generation time per report (prefill excluded). None of these is a report
time.

**The consequence.** At these predicted rates the task is plausible for
**overnight batch reporting**, where one report per recorded night takes
minutes. It is **not plausible for interactive use.**

### Not done: a Pi 5-matched container

Earlier planning referred to a memory- and core-limited container matching the
Pi 5 envelope. **No such container exists, and none was used.**

**It is a candidate for future work:** the five candidates run inside a
container or job object limited to 4 cores and 8 GB. That would make the
memory-envelope comparison *demonstrated* rather than notional, because an
out-of-memory failure or swapping would show directly, and it would show
behaviour on 4 cores.

It would still not reproduce the Pi 5's ARM cores, NEON kernels or LPDDR4X
bandwidth. Throughput would remain analytical even then.

---

## 2F reference run: session log (P1, gemini-3.8-flash)

Each session is started by a person. Every attempt's log record carries its
caller (pid, ppid, argv), available from the 13 September sessions onward.
Quota is counted per Pacific day, and 503s count against it.

| Pacific date | session start (UTC) | started from | attempts | HTTP 200 | HTTP 503 | how it ended | P1 cached after |
|---|---|---|---:|---:|---:|---|---:|
| 12 Sep | 21:07:06 | Antigravity's integrated terminal, by hand (traced after the fact) | 7 | 3 | 4 | the day's budget (20 of 20, preflight included) | 3 / 31 |
| 13 Sep | 09:08:13 | Antigravity's integrated terminal (`pwsh`, pid 8680, child of Antigravity.exe) running `reference/run.py --go` as pid 6720 | 4 | 0 | 4 | unavailable: 4 consecutive 503s on SC4112E0, remaining allowance untouched (16 of 20 left) | 3 / 31 |
| 13 Sep | 09:50:14 (806 s wall clock) | Antigravity's integrated terminal, `pwsh` pid 8680, running the skip-and-breaker runner as pid 20012 | 16 | 5 | 11 | the day's budget (20 of 20), with 3 skips; the longest run of consecutive failures was 5, one short of the breaker | 8 / 31 |

**Run attempts so far: 27, of which 8 succeeded and 19 returned 503 (70%).**

The 09:50 session is the first to run to its budget. Its 16 attempts give a
503 rate of 68.8%, in line with the earlier partial samples. Per packet:

| packet | outcome |
|---|---|
| SC4112E0 | 503, 503, skipped (still pending) |
| SC4131E0 | 503, then 200 |
| SC4171E0 | 503, then 200 |
| SC4172E0 | 503, then 200 |
| SC4321E0 | 503, 503, skipped (still pending) |
| SC4322E0 | 503, 503, skipped (still pending) |
| SC4341F0 | 503, then 200 (the success came after 5 consecutive failures, one short of the breaker) |
| SC4342F0 | 200 |
| SC4351F0 | 503, then the budget ran out (still pending) |

The three skipped packets were not reached again before the budget ran out.
**23 P1 packets remain.** At about 5 responses per 20-attempt day, that is
roughly 4–5 more days.

**Interim pipeline check, 13 September 2026: 8 of 31. This is not a
result.** The 8 saved responses were scored with `reference.score` into a
scratch directory, to confirm the scoring path end to end. All 8 verify with
zero violations: 5/5 mandatory, 14/14 discretionary, 17/17 numbers exact,
14 `hedged_value` claims each, and every one renders. The reference set's
oracle recovery is 1.0 on each night. The 8 are six high, one medium and one
low tier (manifest order), so the harder tiers are under-sampled. **The
decision rule is applied at 31, not now.**

---

## 2F reference run: skip-and-requeue and a session breaker (13 September 2026)

This is an efficiency change to the runner, not to the experiment. The 13
September session spent 4 of its 20 attempts on four consecutive 503s against
one packet and produced no observation. **A 503 is a property of the service
at that moment, not of the packet**, so retrying the same packet straight away
is the least informative use of the next attempt. At a failure rate of about
73%, that pattern could consume a day's allowance on two or three packets.

**The rules** (`reference/run.py`):

- **Skip and requeue.** After 2 consecutive failures on one packet, the
  packet moves to the back of the session's queue and the next pending packet
  goes next. For example, `[A, B, C, D, E]` with A failing twice becomes
  `[B, C, D, E, A]`. The skipped packet stays pending; it is never marked
  failed, and it gets another turn later in the same session if attempts
  remain. The skip is written to the attempt log as its own event, after the
  two failures that caused it, so both can be recovered from the log alone.
- **The session breaker.** After 6 consecutive failures, the session stops
  and the rest of the day's allowance is left untouched. Any success, on any
  packet, resets the count. While two or more packets are pending, the 6 must
  span at least three packets, because skip fires at 2; that is what makes
  the breaker a service signal. **With exactly one packet pending**, the
  requeue returns that same packet, so the 6 fall on it across three visits,
  and the breaker still stops the session.
- **Backoff.** After a failure, the next attempt waits 20 s × 2^(n−1) plus
  jitter, capped at 160 s, whether it goes to the same packet or the next.
  Here n is the session's current run of consecutive failures.
- **Session summaries.** Each session appends one record to
  `reference/logs/sessions.jsonl` with `session_start`, `session_end`,
  `wall_clock_seconds`, `attempts`, `successes`, `503s`, `503_rate`,
  `stop_reason` (complete, budget, service_unavailable or api_error) and its
  skip count. Every attempt record carries its `session_id`. The wall-clock
  span tells a busy hour from a busy day across sessions, which a per-day
  count cannot.
- **Events are never counted.** Skip and session records carry an `event`
  field. The daily budget and the 5 RPM pacing count requests only.
- **Defence in depth.** The `Runner` itself refuses any packet not in the dev
  manifest, however its job list was built.

**Unchanged:**

- every generation setting: thinking off, `maxOutputTokens` 16,384,
  temperature 0, seed 0, the Item 0 schema;
- both prompts, and the schema;
- the per-day attempt budget, and the fact that failures count against it;
- the frozen tier, the test packets, and the three cached responses, which
  are byte-identical to their committed versions.

**Tests.** 10 were added and 3 existing tests were updated for the new retry
behaviour, taking the suite from 339 to 349. (The commit that introduced them
said 11 and 350; the determinism and requeue-order checks are a single test.) All run against a fake transport. They cover:

- skipped, not failed;
- a skipped packet retried later in the session;
- the requeue order, deterministic across runs;
- a success resetting the cross-packet count;
- 6 failures across packets stopping the session with the allowance intact;
- 6 on one packet being unreachable with 2, 3 or 5 packets pending;
- the one-packet case;
- the budget counting attempts but no events;
- session summaries;
- a test packet refused at the queue.

### The runner now says what it decided (13 September 2026)

A `--go` session used to print the status header and then nothing until the
session ended. While it waited out a 20–160 s backoff, it looked exactly like
a dry run that had returned. That is how the 09:50 UTC session was taken for
one that had sent nothing, when in fact it was running and had already made
three attempts.

The command line now announces:

- whether it is starting a session (how many packets are pending and how many
  attempts are left), or not starting and why (the budget is used, or nothing
  is pending);
- every attempt's result;
- every wait, for backoff or for pacing;
- every skip;
- why it stopped, and a closing summary.

A session that declines to start still writes its summary, marked
`"started": false`. Used as a library, the runner stays silent, and a test
asserts it.

5 tests were added, taking the suite from 349 to 354.

---

## Local candidates on the dev population: Qwen2.5-1.5B under P1 (13 September 2026)

This is the first local candidate measured across all 31 dev packets. It
replaces a one-night diagnostic with a dev-population measurement. The other
four candidates have not been started.

**Settings.** They are identical to the diagnostics and, where they apply, to
the reference run:

- P1, which is run C's shape rules, unchanged, on the frozen `build_prompt`
  body;
- the model's own chat template (`llama-completion --jinja -cnv -st`);
- `report/claims.gbnf`;
- temperature 0, seed 0, up to 3,000 tokens, context 12,288.

There was one generation per packet, with no retries and no resampling.

**The harness** is `candidates/run.py`. Its cache is keyed by
`(recording_id, model, model_file, prompt_id, prompt_hash)`, and it accepts dev
packets only. All 31 ran. None was stopped early, and P1 was not modified.

**Determinism check.** P1 on SC4111E0 is byte-identical to diagnostic run C's
prompt. The new output is identical to run C's once Windows line endings are
normalised, and identical as parsed JSON. The harness keeps llama.cpp's raw
CRLF output, which does not affect parsing.

### Scoring, through the frozen verifier and evaluator

| metric | overall (31) | high (11) | medium (10) | low (10) |
|---|---:|---:|---:|---:|
| schema validity | 30/31 | 11/11 | 9/10 | 10/10 |
| overall pass | 0/31 | 0/11 | 0/10 | 0/10 |
| policy pass (of schema-valid) | 0/30 | 0/11 | 0/9 | 0/10 |
| mandatory coverage (mean) | 0.497 | 0.473 | 0.380 | 0.640 |
| **packets at 5/5 mandatory** | **0** | **0** | **0** | **0** |
| discretionary coverage | 0.000 | 0.000 | 0.000 | 0.000 |
| cited mandatory (diagnostic) | 0.774 | 0.800 | 0.720 | 0.800 |
| cited discretionary (diagnostic) | 0.555 | 0.578 | 0.514 | 0.571 |
| oracle recovery | 0.000 | 0.000 | 0.000 | 0.000 |
| unrecovered available (mean) | 14 | 14 | 14 | 14 |
| numeric fidelity | 1.000 | 1.000 | 1.000 | 1.000 |
| reports rendered | 0/31 | 0/11 | 0/10 | 0/10 |
| violations | 106 | 43 | 42 | 21 |

| rule (count) | overall | high | medium | low |
|---|---:|---:|---:|---:|
| `L2.text_key_dependency_missing` | 53 | 22 | 18 | 13 |
| `L2.rem_latency_double_count` | 26 | 11 | 7 | 8 |
| `L2.text_key_predicate_false` | 17 | 10 | 7 | 0 |
| `L2.unsafe_item_not_hedged` | 9 | 0 | 9 | 0 |
| `L1.malformed_json` | 1 | 0 | 1 | 0 |

**`hedged_value`, as two numbers:**

| | overall | high | medium | low |
|---|---:|---:|---:|---:|
| packets using it at least once | **0 of 31** | 0 of 11 | 0 of 10 | 0 of 10 |
| total `hedged_value` claims | **0 of 434** | 0 of 154 | 0 of 140 | 0 of 140 |

The model never uses the form at all. This is the first of the two failure
modes: never learning the form, as opposed to using it once and stopping.
Run C's zero on one night holds across the population.

**The distribution of mandatory coverage:**

| mandatory verified | nights |
|---|---:|
| 4/5 | 2: SC4111E0 and SC4352F0, both low tier |
| 3/5 | 13 |
| 2/5 | 15 |
| 0/5 | 1: SC4171E0 (see below) |

The median is 2/5.

**Per mandatory item, verified and cited, of 31 nights:**

| item | verified | cited |
|---|---:|---:|
| total sleep time | 30 | 30 |
| time in bed | 30 | 30 |
| sleep efficiency | 7 | 30 |
| night confidence | 10 | 30 |
| N1 reliability warning | **0** | **0** |

**What the claims look like.** There were 156 claims in all: 76 values, 50
observations, 30 review flags, and no `hedged_value`.

- **The modal output,** on 9 nights: two values (total sleep time and time in
  bed), two two-item observations under `n1_reliability_is_low`, and one
  review flag.
- **The low-night flag is emitted on almost every night,** 27 times in all.
  It verifies only on the 10 low nights, which accounts for the 17
  `text_key_predicate_false` violations on high and medium nights. So the low
  tier's higher mandatory mean (0.640) is a constant behaviour meeting the one
  tier where it happens to be right. It is not the model reading the tier.
- **Every discretionary item appears only inside multi-item N1-key
  observations.** That produces the missing-dependency violations, and the
  REM double-counts wherever both REM latencies land in one observation.
- **`model.n1_reliability_warning` was never cited,** on any of the 31
  nights.

### What differed from the single-packet diagnostic

1. **SC4111E0 reproduced exactly.** The pipeline is deterministic at these
   settings.
2. **The diagnostic's 4/5 was the top of the distribution,** shared with one
   other low night. The median is 2/5.
3. **A failure one night could not show:** the low-night flag on nights that
   are not low (17 predicate failures).
4. **A degenerate loop.** SC4171E0 generated 44 observation claims in a
   repeating pattern until the 3,000-token limit, which left malformed JSON.
   It is the run's only Layer 1 failure and its one timing outlier.
5. **The `hedged_value` result generalises:** 0 of 31 packets and 0 of 434
   claims.

### Host figures: the x86 evaluation host, run log only

These are not deployment figures.

- **Wall clock:** 588.3 s in total. Per packet: minimum 13.22 s, median
  14.85 s, maximum 99.65 s (the SC4171E0 loop).
- **Peak RSS.** Each packet ran as a fresh llama.cpp process, and its peak
  working set was read from Windows after it exited. The peak was 2,037 MiB on
  every packet (median 2,037.3, maximum 2,037.6) at a context of 12,288.
- **This is not comparable with the deployment table's 1,812 MiB,** which was
  taken at a context of 4,096. The ~225 MiB difference matches the larger
  KV-cache reservation: 28 KiB × 8,192 more positions ≈ 224 MiB.

### The comparison this run answers

- **Single-night diagnostic** (run C, SC4111E0): 4/5 mandatory, 0 hedged.
- **31-night P1 result:**
  - 0 of 31 nights at 5/5 mandatory;
  - mandatory mean 0.497, median 2/5, with 4/5 as the best, on 2 nights;
  - `hedged_value` in 0 of 31 packets and 0 of 434 claims;
  - 0 of 31 reports render.
- **For contrast,** the reference model's interim result is 8 of 8 at 5/5,
  with 14 hedged values per night.

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
