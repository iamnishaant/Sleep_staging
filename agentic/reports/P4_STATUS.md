# Phase 4 status (17 Oct 2026): whole-workflow evaluation on the PC

## Fault scenarios (`python -m agentic.tools.p4_faults`, `p4_faults.json`)

Signal faults are real EDF files made from dev night SC4081E0 (cropped at lights-off, 12 h), so they go
through the real ingest path. Report-model faults are injected backends. The oracle backend stands in for
the model where the scenario tests the workflow.

| # | Scenario | Expected → got | Result |
|---|---|---|---|
| S01 | Clean night, no hypnogram | VERIFIED_REPORT | PASS |
| S02 | Prebuilt packet (Path A) | VERIFIED_REPORT | PASS |
| S03 | EEG flat for 2 h | REVIEW_ONLY `Q01` | PASS |
| S04 | EOG missing (v1 uses EEG only) | VERIFIED_REPORT | PASS |
| S05 | Fpz-Cz missing (Pz-Oz present) | REVIEW_ONLY `W05` | PASS |
| S06 | Recorded at 128 Hz (resampled to 100) | VERIFIED_REPORT | PASS (also covers PENDING V2 for this night) |
| S07 | Only 2 h recorded | REJECTED `R03` | PASS |
| S08 | Corrupt header | REJECTED `R01` | PASS |
| S09 | Report model crashes once | VERIFIED_REPORT (retry) | PASS |
| S10 | Report model hangs | FALLBACK `F02` | PASS |
| S11 | Report model writes invalid claims | FALLBACK `F03` | PASS |
| S12 | Device too hot (82 °C) | FALLBACK `F01` after 12 waits | PASS |
| S13 | Power cut during generation (kill, restart) | VERIFIED_REPORT, exactly 1 delivery | PASS |
| S14 | Strong 49 Hz line noise | VERIFIED_REPORT | PASS (no line-noise check exists; staging and the report still run) |
| S15 | Two nights of one subject (trend note) | — | DEFERRED: needs the history store (Phase 6) |

**14/14 pass, 1 deferred.**

## Invariants

Runtime checks run on every scenario; static checks run on the agent code.

| Invariant | Check | Result |
|---|---|---|
| I1 no unverified text | A VERIFIED page ⇒ verifier ok and V-HTML clean | holds on all 14 |
| I2 numbers have provenance | Every delivered report or fallback page has 0 unexplained numbers | holds |
| I3 frozen components | Run's execution manifest = the current release manifest | holds |
| I4 no network | No network imports in agent code (static) | holds |
| I5 test data / expert hypnograms never read by the workflow | No hypnogram file or annotation reads in agent code (static; only `tools/` evaluation scripts do) | holds |
| I6 always ends | Every run has a DELIVERED marker | holds |
| I7 every step traced | Every step directory has a "done" trace entry | holds |
| I8 failure visible | Every non-verified outcome, or W/Q warning, has a review item | holds |

## Added in this phase
- **`wait_resources` step** before every model launch, in both paths: SoC temperature (Linux sensor) and
  free RAM against the policy. Waits up to 12 × 5 min, then FALLBACK `F01`. 2 tests.
- **48 tests pass.**

## Both scorecards (Phases 1–2, still current)
- **Path A, system:** dev 30/31 byte-identical to frozen; SC4582G0 differs (decision D1). Replay 31/31.
- **Path B, system:** 31/31 verified reports, review rate 16%, median 81 s per night on x86.
- **Path B, model:** κ 0.682, accuracy 76.7%, N1 F1 51.8 (on the estimated span vs expert labels).

## Notes for later (also in /PENDING.md)
- **No line-noise check:** the policy lists one, but `signals.py` only checks flat/stuck. S14 passes
  because nothing flags it, not because it's handled.
- **RAM floor is close on this PC:** 6 GB against about 7 GB free. A busy PC pushes runs into the resource
  wait (seen while testing S13).
- **The report window** (01:00–07:30) belongs to the scheduler (Phase 6), not to `wait_resources`.
