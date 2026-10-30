# R2: Verifier Ablation, Results

**24 October 2026.** The definitions were fixed in [PREREGISTRATION.md](PREREGISTRATION.md) before arm C
was generated. 31 dev nights; the test set was not read.

- **Reproduce:**
  - `python "ROADMAP R2 - VERIFIER ABLATION/run_arm_c.py" --go` (arm C generation; cached in `cache/`);
  - `python "ROADMAP R2 - VERIFIER ABLATION/score_ablation.py"`.
- **Per-night data:** `results/ablation.json`.
- **Scorer self-check:** `test_score_ablation.py` (4 tests).

## Arms

| Arm | Grammar | Verifier gate | Generations |
|---|---|---|---|
| A | on | on | cached P1 run |
| B | on | **off** | the same generations as A |
| C | **off** | **off** | new for Llama-Student (this task); cached 2G run for the base model |

## Results

| Model | Arm | Reports released | **Released with a flaw** | Malformed | Claims released | Wrong number | Wrong / missing evidence | Unreliable stated as fact | Reliable hedged | Other | Missing a mandatory item | Numeric fidelity | Token cap hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Llama-Student** | A | 30/31 | **0** | 0 | 582 | 0 | 0 | 0 | 0 | 0 | 0 | 522/522 | 0 |
| | B | 31/31 | **1** | 0 | 600 | 0 | 0 | 0 | 3 | 0 | 0 | 538/538 | 0 |
| | C | 31/31 | **2** | 0 | 632 | 0 | 0 | 1 | 3 | 0 | 1 | 571/571 | 0 |
| **Llama-3.2-3B base** | A | 0/31 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | 20 |
| | B | 31/31 | **31** | 20 | 359 | 0 | 0 | 145 | 6 | 9 | 20 | 337/337 | 20 |
| | C | 31/31 | **31** | 31 | 0 | — | — | — | — | — | 31 | — | 31 |

- **Column meanings:**
  - "Released with a flaw": a report reaching the reader with at least one incorrect claim, a malformed
    output, or a missing mandatory item.
  - Claim columns count claims.
  - "Missing a mandatory item" counts reports.
- **In arm A,** a night that isn't released goes to the deterministic fallback summary, so the reader
  still gets something correct.

### The flawed Llama-Student releases

| Night | Arm | What a reader would have seen |
|---|---|---|
| SC4582G0 | B, C | Total sleep time, time in bed and sleep efficiency, the three items safe to state plainly, given as approximations with caveats (3 × `L2.safe_item_hedged`). This is the 4 Oct cached output; today's re-execution of arm A passes (D1, reference 2) |
| SC4111E0 | C only | One unreliable metric stated as fact (`L2.unsafe_item_not_hedged`), and one mandatory item missing |

SC4111E0 is also the worst night in the T1 time-in-bed analysis: the model scores 3 h of morning wake
largely as REM.

## What this shows

**1. The gate released nothing incorrect.**
- **Primary model, arm A:** 0 flawed releases out of 31. Every flawed output (1 in arm B's
  generations) was caught and replaced by the fallback.
- **Base model, arm A:** the gate blocked all 31 flawed outputs.

**2. For the trained Student, the gate's measured value on dev is small but real.**
- **Without the gate:** 1 of 31 reports (3 %) would reach a reader with a policy error. With the
  grammar off as well, 2 of 31 (6 %).
- **What kind of errors:** they're about **hedging and completeness**, not wrong numbers. Numeric
  fidelity is 100 % in every Student arm.
- **What we can't claim:** we said in advance we'd report this plainly. On this model and these
  nights, the gate is a safety net that fires rarely. It is not shown to prevent large-scale
  hallucination.

**3. For an untrained model, the gate is the difference between a usable and an unusable system.**
- **Ungated, the base model is flawed on all 31 nights.** It states unreliable metrics as fact 145
  times and loops into the token cap on 20 nights even with the grammar.
- **Gated, it releases nothing incorrect.** Every night falls back.
- **So the gate makes correctness independent of model quality:** a weaker or changed report model
  degrades **coverage** (more fallbacks), not **correctness**.

**4. The grammar matters less than expected for the trained Student.**
- **Formatting comes from training:** without the grammar the Student still wrote valid JSON on 31/31
  nights, and never looped.
- **The grammar's measured effect** is one extra policy error (SC4111E0).
- **For the base model,** the grammar is necessary but not enough. With it, 11/31 outputs parse; without
  it, none do.

**5. No arm produced a wrong number.**
- When the Student states a value, it copies it exactly.
- This matches the base models' Phase 2 finding: they read numbers correctly and fail on the rules.

## Limits

- 31 dev nights, already reused; one generation per night. Counts of 1–2 can't separate the B and C
  rates statistically.
- Arms A and B share generations by design, since the verifier doesn't change what the model writes.
  So B − A isolates the gate's effect exactly.
- Base arm C is scored strictly, as pre-registered: no fence stripping. All 31 hit the 3,000-token cap,
  so they couldn't be parsed under any reading.
- One model family (Llama 3.2 3B). A different or smaller report model would likely show larger gate
  value, as the base model does.

## How to state it in the report

> On 31 development nights, the deterministic verifier released no incorrect claim. Without it, the
> trained 3B model would have released 1–2 reports (3–6 %) with hedging or completeness errors, and
> no numeric errors. The untrained model would have released 31/31 flawed reports. The gate turns
> report-model weakness into fallbacks rather than errors.
