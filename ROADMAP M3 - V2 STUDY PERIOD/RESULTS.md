# M3: study-period estimator on v2 predictions, results

**25 October 2026.** Pre-registration: `PREREGISTRATION.md` (written before any v2 prediction existed).
Scripts: `m3_refit.py`. Data: `results/frozen_params.json`, `results/dev_eval.json`.

## Outcome: keep v0.3 for v2. The refit is not adopted.

## Train (137 nights, v2 pass 1, frozen before dev was read)

| Parameters | start median / p90 | end median / p90 |
|---|---:|---:|
| v0.3 (run 5, gap 8, merge 240) on v2 | 0 / 5.4 | 1 / 45.0 |
| **Refit** (run 5, gap 2, no merge), frozen | 0 / 3.0 | 1 / 45.0 |
| For context: v0.3 on N4kd's own predictions | 0 / 6.4 | 1 / 41.4 |

Selection used the same grid and rule as v0.3. The refit differs only in start p90.

## Dev (31 nights, one evaluation; fourth use of dev for an estimator, a development acceptance set)

| | v0.3 params on v2 | Refit on v2 |
|---|---:|---:|
| Start error median / p90 (epochs) | 0 / 2 | 0 / 2 |
| End error median / p90 (epochs) | 2 / **41** | 3 / **41** |
| TST abs error median (min) | 0.0 | 0.0 |
| SE abs error median (points) | 0.18 | 0.20 |
| Mean κ change | −0.0012 | −0.0040 |
| Nights with end error > 30 epochs | 5 | 6 |
| W08 flags | 1 | 1 |
| Bars met | start, TST, SE, κ | start, TST, SE, κ |
| End bar (p90 ≤ 30) | **not met** | **not met** |

**Decision rule (fixed in advance):** adopt the refit only if it meets every bar v0.3 meets and lowers end
p90 by at least 5 epochs. It lowers it by 0 and has one more large-end-error night, so **v0.3 stays**.

## What this shows

- **v0.3 already works for v2.** On v2's predictions it meets the same four bars it met on N4kd (start,
  TST, SE, κ), with TST median 0 min and SE median 0.18 points. This was the open question in M3.
- **The end-point deviation is unchanged,** and slightly better: p90 41 epochs against the bar of 30
  (it was 49 for N4kd on dev). The deviation accepted on 17 Oct (X1) carries over to v2, no worse.
- **The deviation is not a tuning problem.** Train p90 is 45 for both parameter sets on v2, and no
  setting in the grid reduces it. It comes from what happens at the morning end (wake after the last
  sleep), so a different staging model doesn't remove it.
- **Quality and line-noise thresholds** are EEG-only and depend on the expert reference spans, not the
  model, so they carry over unchanged. They weren't refitted.

## Effect on the release

None. `agentic/config/policy.yaml` (policy 0.2) is used unchanged for v2. M3 is closed.
