# P8: Release v2 on the locked test set — results

**2 Nov 2026.** The single, pre-registered final evaluation of release v2 (`PREREGISTRATION.md`).
Nothing below was changed, re-run, or tuned after seeing these numbers. One delivery per night,
29/29, under the frozen v2 execution manifest (pins checked before the run).

## Headline

| | v2 (this run, N6adv packets) | v1 reference (`deploy/frozen.json`, N4kd packets) |
|---|---:|---:|
| Verified reports | **28/29** | 29/29 |
| 95% lower bound (Clopper–Pearson) | **82.2%** | 88.1% (quoted project-wide) |
| 5/5 (pass + all mandatory items) | 28/29 | 29/29 |
| Flawed reports released | **0** | 0 |
| Violations (on the 1 blocked night) | 2 | 0 |
| Numeric fidelity | 492/493 | 1.000 |

**The inputs differ** (v2 is scored on N6adv/EEG+EOG packets, v1 on N4kd/EEG-only packets), so this
is a release-to-release comparison, not the same model on the same data. The project's 29/29 bound of
88.1% remains v1's own, measured result.

## System-level outcome

| Outcome | Count | Reason codes |
|---|---:|---|
| VERIFIED_REPORT | 28 | — |
| FALLBACK_SUMMARY | 1 | F03 (verifier rejected) |

No REVIEW_ONLY, no REJECT, no crash or timeout fallback (F01/F02), no re-run was needed.

## The one blocked night: SC4611E0

The model wrote a REM-latency claim citing `arch.rem_latency_sustained` with value `-1.0`. The
packet's actual value for that field is `None` — this night has almost no REM sleep (REM is 0.12%
of the night, `stage.REM.fraction = 0.0012`), so there is no sustained REM period to report a
latency for. The model appears to have substituted a sentinel value instead of omitting the claim.

The verifier caught both the value mismatch and the uncited-quantity violation (`L2.value_mismatch`,
`L2.uncited_quantity`), the generation was never released, and the agent delivered the fallback
summary instead. **This is the system working as designed**: on this cohort it demonstrates the
same guarantee the paper's main claim rests on — a flawed generation gets blocked, not shipped.

## What this adds to the journal draft

- v2's report model now has **test-set evidence**, closing the gap noted in `FOR JOURNAL/README.md`
  (it previously had dev-only evidence, 31/31).
- A **second real example of the verifier blocking a hallucinated value** on a held-out night,
  for the same reason the dev/test narrative already uses (fabricated or uncited numbers).
- Do not merge this 28/29 with v1's 29/29 into one combined figure — they're different packet sets;
  report them side by side, as above.

## Recorded, not re-run

- **RAM floor override**: `resources.min_free_ram_gb` 6 → 5.0 for this run only (`policy_ram5.yaml`),
  same justification and same value as the v2 dev scorecard (`PENDING.md` V7). Changes the policy
  hash, so these runs carry new run IDs, not the frozen-policy identities — consistent with how V7
  is already handled project-wide.
- Raw generations, verify records and delivery records for all 29 nights are under `runtime/`.
  `results.json` holds the per-night detail this file summarises.
