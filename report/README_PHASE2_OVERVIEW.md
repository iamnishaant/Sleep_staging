# Phase 2 in plain terms: what we are doing, why, and what happens next

**Explainable Deep Learning for Sleep Disorder Detection · Team 40 · Project 48**
**Written 17 September 2026.**

This is the plain-language guide to Phase 2. It assumes no knowledge of the
project beyond the title, and it defines the terms it uses. The other
documents in this folder are the detailed records; this one is the map.

---

## 1. The project so far

**Phase 1 built the scorer.** A small neural network reads one night of a
sleep recording and labels every 30-second slice as one of five stages:
Wake, N1, N2, N3 or REM. From that sequence it computes the numbers a sleep
clinic cares about — how long the person slept, how efficiently, how long it
took them to fall asleep, how much of each stage.

It also reports **how much it trusts itself**, both overall for the night and
stage by stage. That matters, because it is much better at some stages than
others. It recognises Wake well, and N1 — the lightest stage, the one humans
also disagree about — poorly.

**Everything that comes out of Phase 1 is numbers.** A hypnogram, a table of
statistics, and a set of confidence figures.

**Phase 2 turns those numbers into a written report.** Not because prose is
prettier, but because "explainable" is the point of the project. A table of
numbers is not an explanation. A paragraph that says what the model found,
how confident it is, and which parts to distrust — that is.

---

## 2. Why this is hard

A language model would write that paragraph instantly. It would also,
sooner or later, make something up. That is not a bug that gets patched; it
is how these models work. They produce fluent, plausible text, and fluent
plausible text is exactly what a wrong number looks like.

**In this setting, a confident wrong number is worse than no report at all.**
If the report says "sleep efficiency 84%" and the real figure is 76%,
nobody can tell by reading it. The error is invisible precisely because the
writing is good.

So the design of Phase 2 starts from a decision: **we do not trust the
language model, and we do not ask anyone else to.** We check everything it
says, mechanically, against the source data, and we throw away anything that
does not check out.

---

## 3. How we made it safe

This machinery was built in Phase 1 and is now **frozen** — it does not
change while we test models against it, or the test means nothing.

A useful way to picture it: the language model is a junior writer. The
verifier is a fact-checker sitting with the source data. The renderer is the
printer, which only prints what the fact-checker approved.

**The packet.** Everything known about one night, as a structured file. It
lists **19 evidence items** — the only things a report is allowed to talk
about. Anything not in the packet cannot be mentioned, and anything in the
packet but not on the list of 19 cannot be cited.

**Claims, not prose.** The model does not write paragraphs. It writes small
structured statements, each one citing exactly which piece of evidence it
rests on:

```json
{"claim_id": "c17", "claim_type": "hedged_value", "cites": ["arch.rem_latency"],
 "subject": "this_recording", "value": 119.0, "unit": "minutes"}
```

The English comes later, from fixed templates, once the claim has passed.
The model never writes the final wording.

**The contract.** Of the 19 items, the report must cover:

- **5 items stated outright.** Total sleep time, time in bed, sleep
  efficiency, the night's confidence tier, and the warning about N1
  reliability. These are the facts the pipeline is reliable enough to assert
  plainly.
- **14 items hedged.** Everything the model is *not* reliable enough to
  assert — these must be written in the cautious form, which renders as an
  approximate value carrying a caveat, never as a bare fact.

**Getting this backwards is a failure in both directions.** Hedging a fact
that is solid makes the report useless. Stating a shaky number plainly makes
it dangerous.

**The verifier.** Two layers and 30 distinct ways a claim can be rejected.
The first layer checks shape without looking at the data; the second checks
every claim against the packet. Its governing rule:

> **The verifier never infers missing evidence.** If the packet does not
> explicitly contain what a claim needs, the claim fails. No interpolation,
> no rounding allowance, no benefit of the doubt.

**The grammar.** The model is constrained as it writes, so that many kinds
of bad claim are not rejected but **impossible to express**. A whitelist can
be shown to be complete; a blocklist cannot.

**The render gate.** A report is produced only when the whole set of claims
passes with zero violations. There is no partial credit.

---

## 4. What Phase 2 is actually testing

Two questions, in order:

1. **Can a language model fill this contract at all?**
2. **Can a small one — small enough to run on a Raspberry Pi with no
   internet — do it?**

The target is a Raspberry Pi 5 producing reports overnight, in batch. That
is the deployment the project is aiming at: no cloud, no data leaving the
device, no per-report cost.

So we test two tiers against the identical packets, prompt and verifier:

- **Local small models:** five of them, 1.5 to 3.8 billion parameters,
  compressed to run on a CPU.
- **A reference model:** Gemini 3.8 Flash, hosted, far larger. Not a
  candidate for deployment. It is there to tell us what is achievable.

Everything is measured on **31 "dev" nights**. A separate set of **29 test
nights is locked away** and has not been looked at once — it is checked by
checksum before and after every session. It gets used once, at the very end.
That is what stops us from tuning until something passes.

---

## 5. What we have found

### The local models: zero

Five models, 31 nights each, 155 reports. **Not one report both passed the
verifier and covered all five mandatory facts.**

The interesting part is *how* they fail:

| model | what it does |
|---|---|
| **Qwen2.5-1.5B** | States facts well, finishes cleanly — but never once uses the cautious form, and attaches its citations to the wrong evidence. |
| **SmolLM2-1.7B** | Falls into a loop on 30 of 31 nights: lists values, then repeats them until it is cut off mid-sentence. |
| **Gemma-2-2b** | Always well-formed, never loops — but says almost nothing, usually one fact a night. |
| **Llama-3.2-3B** | When it finishes, it covers nearly everything and hedges correctly — but it also repeats each hedged item as a plain fact, and it loops on 20 of 31 nights. |
| **Phi-3.5-mini** | Writes one fixed template per confidence tier. Clean on easy nights, but on the others it hedges the very facts that are safe to state. |

**They are not failing at arithmetic.** Across all 155 reports there is
exactly **one wrong number**. They read the packet correctly and copy the
figures exactly.

**They are failing at the rules** — which kind of statement goes with which
item, and which evidence a statement must cite. And each model fails
differently, which tells us this is not one model's quirk.

### The reference model: so far, perfect

Every night it has answered has come back with all five mandatory facts, all
14 hedged items, every number exact, zero violations, and a report that
renders.

**Two honest cautions.** Only 8 of its responses have been scored so far,
and most of those are the easier, high-confidence nights, because the run
works through the list in order. The figures should be expected to fall as
the harder nights arrive. That is why no conclusion is being drawn yet.

---

## 6. What this means

The local models' total failure has two possible explanations, and they point
in opposite directions:

1. **The contract is the problem.** Too strict, or badly designed, so that
   nothing can satisfy it. If so, training a small model is wasted effort,
   and the honest finding is that the contract is the ceiling.
2. **The small models are the problem.** The task is doable; they are simply
   too weak when merely *asked* to do it. If so, training them is exactly the
   right response.

**The reference run separates these two.** Same packets, same prompt, same
verifier — the only change is a much stronger model. If it fills the
contract and the small ones score zero, the contract is fine and the gap is
capability.

The evidence so far points that way, and it suggests a specific fix.
The small models can already do the hard parts — read the packet, find the
right evidence, copy numbers exactly. What they lack is the mapping: which
form of statement to use, and which evidence to cite. That is a narrow,
learnable skill, and the standard way to teach it is **distillation**:
instead of describing the rules in a prompt, show the small model a few
hundred worked examples of the task done correctly, and train it on them.

---

## 7. The rule we wrote down in advance

Whether we train anything is decided by a rule fixed **before any results
existed**, applied when all 31 reference nights are in:

| result | what it means | what we do |
|---|---|---|
| **25 or more of 31** complete, clean reports | the contract is satisfiable | train a small model to close the gap |
| **around 12 of 31** | the way we phrased the task may be the problem | try the reserve prompt, P2, before anything else |
| **near zero** | the contract itself is the ceiling | stop and report that finding |

**Why fix it in advance?** Because when the results arrive there will be a
strong temptation to read them as whatever is most convenient. A rule written
beforehand removes that. The third outcome is not a failure of the project —
"this cannot be done safely at this size" is a real result, and reporting it
honestly is worth more than a number we massaged.

---

## 8. Where each outcome leads

**If the rule routes to training** — the expected case on current evidence:

1. Run the grammar ablation (2G), which measures what the grammar
   constraint is actually contributing, including whether it is causing the
   loops.
2. Train **Qwen2.5-1.5B**, the recommended student, on 123 worked examples
   with 14 held back for validation. These come from nights that share no
   subject with the dev or test sets.
3. Train **Llama-3.2-3B** alongside it, purely to answer whether the extra
   size is worth roughly double the memory.
4. Measure both on the same 31 dev nights, with the same harness.
5. Only then, once, on the 29 locked test nights.

**If the rule routes to P2:** re-run the reference with the reserve prompt
before spending effort on training.

**If the rule routes to "the contract is the ceiling":** write up the
finding, with the five-model comparison as the supporting evidence.

---

## 9. How to move forward

**Right now — finish the reference run.** 19 of 31 nights are saved, so 12
remain, which is about three more days.

```powershell
python -m reference.run          # shows the plan, sends nothing
python -m reference.run --go     # runs one session
```

- **Once a day, after 07:00 UTC,** when the free tier's 20-attempt daily
  allowance resets.
- **Change nothing** — not the prompt, the model, the settings or the
  packets. All 31 responses must come from identical conditions, or they
  cannot be compared with each other or with the local models.
- **If it stops early, run it again.** The service returns "high demand"
  errors most of the time, and the runner gives up after six in a row. It
  does not spend the rest of the day's allowance when it does. On 16
  September the first session saved nothing and stopped with 14 attempts
  left; a relaunch five minutes later used them and saved five responses.
- **Re-running is always safe.** A night already saved is never requested
  again.

**Then:** score all 31, apply the rule, and take whichever branch it
selects.

**The constraints we hold to throughout,** because they are what makes the
result trustworthy:

- the deterministic tier stays frozen;
- the 29 test nights stay untouched until the very end;
- one generation per night, no retries, no best-of-N;
- the decision rule is applied as written, not renegotiated.

---

## 10. Glossary

| term | meaning |
|---|---|
| **epoch** | one 30-second slice of the night, the unit the scorer labels |
| **hypnogram** | the sequence of stage labels across the night |
| **packet** | the structured file holding everything known about one night |
| **evidence item** | one of the 19 things a report is allowed to talk about |
| **claim** | one structured statement by the model, citing its evidence |
| **hedge** | the cautious claim form, rendering as an approximate value with a caveat |
| **mandatory / discretionary** | the 5 items that must be stated plainly; the 14 that must be hedged |
| **5/5** | a report covering all five mandatory facts |
| **verifier** | the deterministic checker; 30 rejection codes, two layers |
| **render gate** | the rule that only a report with zero violations is produced |
| **dev / test** | the 31 nights we measure on; the 29 locked away for the end |
| **the contract** | the full requirement: pass the verifier *and* cover all five mandatory facts |
| **reference model** | the strong hosted model, used to find the ceiling, not to deploy |
| **distillation** | training a small model on worked examples of the task |
| **quantisation / GGUF** | the compressed format that lets these models run on a CPU |
| **503** | the "high demand" error the free tier returns most of the time |

---

## 11. Where everything lives

| document | what it covers |
|---|---|
| `report/README_PHASE2_OVERVIEW.md` | this document |
| `report/PHASE2_STATUS.md` | the current state, with every headline figure |
| `report/README_SLM.md` | the small-model work in full |
| `report/README_FIVE_MODELS.md` | the five-model comparison, table by table |
| `report/README_DISTILLATION.md` | what training would involve, and why |
| `report/PHASE2_NOTES.md` | the lab record: every run, every session, every correction |
| `report/PHASE1_REPORT.md` | how the deterministic tier was built |
| `report/README.md` | the schema, grammar, verifier and renderer in detail |

| directory | contents |
|---|---|
| `reference/` | the reference runner, its cache of saved responses, and its logs |
| `candidates/` | the local-model harness, its cached generations and its scores |
| `student/` | the training set and the training kit, built and waiting on the rule |
| `deploy/` | the Raspberry Pi memory and speed figures |
| `distillation/results/` | the packets: 137 for training, 31 dev, 29 test |
