# R2b: Verifier Ablation on Release v2 — Results

**2 Nov 2026.** Run as pre-registered (`PREREGISTRATION.md`), on the 31 v2 dev packets; the test set
was not touched. One clean single-runner pass, after an earlier duplicate-runner fault whose 13 outputs
were discarded unread (`RUN_NOTES.md`).

## Result

| Arm | Released | Flawed released | Malformed | Claims | Wrong number | Wrong/missing evidence | Unreliable as fact | Missing mandatory | Fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A (grammar on, gated) | 31/31 | **0** | 0 | 589 | 0 | 0 | 0 | 0 | 527/527 |
| B (grammar on, no gate) | 31/31 | **0** | 0 | 589 | 0 | 0 | 0 | 0 | 527/527 |
| C (grammar off, no gate) | 31/31 | **0** | 0 | 589 | 0 | 0 | 0 | 0 | 527/527 |

**All 31 arm C outputs are byte-identical to arm A's.**
- They were generated fresh: arm C averaged 89.5 s per night, against arm A's 123 s in R3.
- The arm C command was checked to be release v2's invocation with `--grammar-file` removed and nothing
  else changed.

At temperature 0 this means the grammar never changed a single token choice. On these nights, the
fine-tuned v2 model already writes valid, correct claims without it.

## Interpretation (the pre-registered rule)

On dev, **neither the grammar nor the verifier was needed for v2**. There was nothing for them to stop.
- **v1 for comparison (R2):** removing the gate released 1 flawed report, and removing the grammar as
  well released 2.
- **The verifier's value for v2 is therefore as a safety net, and the test set shows it working:** in
  P8, it blocked the one flawed v2 generation on held-out data. That was an invented REM-latency value on
  a night with almost no REM sleep (`ROADMAP P8 - V2 TEST EVALUATION/RESULTS.md`).
- **On an untrained model the verifier does decisive work:** without the gate, 31 of 31 reports released
  a flaw (R2 secondary arms). That result was measured on **v1** packets and was not regenerated on v2
  packets (pre-registered).

**Claim to make:** "Fine-tuning made v2's reports correct by construction on dev (0 flawed with the
grammar and verifier removed). The verifier is a safety net for the residual failures seen on held-out
data (1 of 29 blocked), and the decisive control for weaker models (31 of 31 flawed releases prevented
for the base model)."

**Claim to avoid:** "The verifier prevents v2's errors on dev." There were none to prevent.

Files: `results/ablation_v2.json` (per night), `results/ABLATION_TABLE_V2.md`, `cache_arm_c/` (raw outputs).
