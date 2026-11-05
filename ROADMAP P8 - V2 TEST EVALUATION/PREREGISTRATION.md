# P8: Release v2 on the locked test set: pre-registration

**Written 2 Nov 2026, 01:55 IST, before the run.** Nothing below changes after the run starts.

This is the single, logged final test evaluation of the frozen release (SYSTEM_ACCEPTANCE G2 /
agent P8). It closes one gap: v1's report result on test is 29/29, while v2's report model has dev
evidence only (31/31). The user decided on 2 Nov that v2 is the final release for the paper.

## What is frozen

| Item | Value |
|---|---|
| Release | `agentic/config/releases/v2.yaml`, execution manifest `4973f498b099fd9f…` (pins checked 2 Nov, all match) |
| Report model | `student/slm_models/v2/llama-3.2-3b-instruct-student-v2-q4_k_m.gguf` (sha256 `55730aae…`) |
| Runtime | llama-completion b10927, grammar `report/claims.gbnf`, pinned chat template, temperature 0, seed 0 |
| Verifier, renderer, serializer | Frozen files pinned in the release |
| Inputs | The 29 v2 test packets in `ADDING EOG VERSION - PIPELINE INTEGRATION/work_n6adv_packets/packets_test/`, built earlier by the v2 staging model with ground truth withheld |
| Policy | `config/policy.yaml` with **one** change: `resources.min_free_ram_gb` 6 → 5.0 (`policy_ram5.yaml`). This host has about 5.9 GB free, and the report model peaks at about 4.9 GB. It is the same override, for the same reason, as the v2 dev scorecard (PENDING V7, `agentic/reports/P2_STATUS.md`) |

## How it runs

```
python -m agentic.cli run-packets "ADDING EOG VERSION - PIPELINE INTEGRATION/work_n6adv_packets/packets_test" \
    --release v2 --backend llama --policy "ROADMAP P8 - V2 TEST EVALUATION/policy_ram5.yaml" \
    --runtime "ROADMAP P8 - V2 TEST EVALUATION/runtime"
```

One generation per night. The agent's own crash retry (`report.retries` = 1) is part of the
release behaviour and is kept.

**Re-runs.** A night may be re-run once **only** if its outcome comes from infrastructure, not the
model:
- `F01`: the resource wait timed out;
- `F02`: a process crash or timeout after the agent's own retry.

Both attempts are reported. A night is never re-run because of what the model wrote or what the
verifier decided.

## What is reported: all 29 nights, whatever the result

**System level** (the agent's delivery records):
- the count of each outcome (VERIFIED_REPORT, REVIEW_ONLY, FALLBACK, REJECT), with reason codes;
- **flawed reports released**: VERIFIED_REPORT nights whose raw output fails the frozen scorer. This
  should be 0 by construction.

**Report model** (each night's `raw.txt`, scored with `report.evaluate.score_output`, the scorer
behind v1's 29/29 and v2's dev 31/31):
- verifier pass count;
- the 5/5 count (pass plus all mandatory items covered);
- total violations, with codes;
- numeric fidelity (matched / total).

**Uncertainty:** a 95% Clopper–Pearson lower bound on the verified count and on the 5/5 count.

## Comparison and its limits

- **Reference:** v1's frozen test result, 29/29 at 5/5, 0 violations, fidelity 1.000
  (`deploy/frozen.json`, 5 Oct 2026).
- **The inputs differ.** v1 was scored on N4kd packets and v2 on N6adv packets. This is a
  release-to-release comparison, not the same model on the same inputs.
- **Fixed in advance:** no part of the release, the packets, the verifier, the thresholds or the
  scorer is changed in response to this result. The result goes into `RESULTS.md` as it comes.
