# 06 — Second review adopted: six clarifications, three decisions, Phase 0 started

**16 October 2026.** The second review approved the plan to proceed to Phase 0, subject to six
clarifications. All are adopted. The precedence order is now **06 → 05 → 01–04**.

| # | Issue | Resolution | Where |
|---|---|---|---|
| 1 | The run ID ignored model/software versions, so v1 outputs could be reused under v2 | `run_id = sha256(input + policy + execution_manifest)`. The execution manifest hashes 20 artefacts, from the staging checkpoint and tables to the GGUF and the `llama-completion` binary. Built: `execution_manifest_sha256 bf59afa1…` | `phase0/ACCEPTANCE.md §1`, `results/golden/execution_manifest_v1.json` |
| 2 | P0 blurred the reference paths | Three labelled references. **(1)** Frozen packet evaluation: built, 31 nights. **(2)** Legacy raw-EDF path: diagnostic only. **(3)** Corrected raw-EDF path: built in P2. Path A reproduces the original outcomes **including SC4582G0's failure** | `phase0/CONTRACTS.md §10`, `results/golden/` |
| 3 | Study-period estimator: coordinate mapping and repeated tuning | All indices are in original-recording epochs. Tune on train/val only, freeze a version, evaluate it on the 31 dev nights; repeated use is labelled a *development acceptance set*. `R04` instead of silently using the whole recording | `ACCEPTANCE.md §4.2`, `policy.yaml study_period` |
| 4 | Cohort-specific quality thresholds can't be used on a new device | The default policy is profile-agnostic. Cohort is used only to stratify evaluation. Device-profile overrides only when declared and validated. Thresholds must separate worse-staging nights, not make ST pass | `ACCEPTANCE.md §4.3` |
| 5 | Older documents still describe `llama-server`, `/health`, grammar-in-request and server restarts | 02, 03 and 04 were **edited in place**: the REPORT step, policy, software stack and Phase 3 now describe `llama-completion`. The history stays in 05 §6 | 02 §2–4, 03 §4, 04 Phase 0 and Phase 3 |
| 6 | V-HTML needs an explicit provenance model | Four sources: packet field, quality-sidecar field, **registered derivation** (with source IDs and rule), whitelisted constant. Applies to the report and fallback pages | `ACCEPTANCE.md §5`, `policy.yaml vhtml` |

**Three decisions made explicit**

| | Decision | Where |
|---|---|---|
| A | Four outcome categories: VERIFIED_REPORT, FALLBACK_SUMMARY, REVIEW_ONLY, REJECTED. Only the first counts as a verified report. The headline metric is verified-report coverage over all nights | `ACCEPTANCE.md §2` |
| B | The test set stays out of the decision loop. A P8 discrepancy is a recorded failure, fixed on dev data in a new version | `ACCEPTANCE.md §8` |
| C | **v1 is the only initial release** (P0–P7). v2 is introduced only after its own dev bar and a frozen manifest | `policy.yaml routing`, `ACCEPTANCE.md §8` |

---

## What Phase 0 found (16 Oct)

1. **Reference 1 (frozen packets, v1, 31 dev nights):**
   - every cached output is present;
   - every prompt hash matches today's prompt builder;
   - **30/31 at 5/5**, numeric fidelity 538/538.
   - **SC4582G0** fails, blocked by the verifier for 3 × `L2.safe_item_hedged`. An earlier summary
     line said "0 violations" for dev; that is corrected in 05 §11.
2. **Reference 2 (legacy raw-EDF path) can't stage a new recording:**
   - **9** spectral features instead of the **34** the model was trained on. `infer` crashes, confirmed on SC4081E0.
   - The hypnogram lookup misses Sleep-EDF file names, so the full 23.3-h recording is used.
   - Plus the trim-rule, sampling-rate and montage defects already known.

   So `pipeline.py` works only through its fast path on the project's own pre-built tensors.
   **There's no legacy raw output to preserve.** Path B is built new.
3. **The training spectral features need `pywt`,** which isn't installed on this PC. That is a P2
   dependency, to be pinned in the environment.

## Phase 0 deliverables

| Deliverable | Status |
|---|---|
| `phase0/CONTRACTS.md` | done |
| `phase0/ACCEPTANCE.md` | done (v1, frozen before code) |
| `phase0/policy.yaml` v0.1 | done (quality values are `to_calibrate` in P2 by design) |
| `phase0/build_golden.py` → `results/golden/` | done: manifest, reference 1, 30 rendered pages, legacy probe |
| Canonical baseline table | 05 §11 (corrected) |

**Next: P1.**
1. Agent skeleton: `state.py`, `policy.py`, `runner.py`, a `llama-completion` wrapper.
2. Re-generate all 31 dev reports through the wrapper and check them byte-identical against reference 1.
   That is about 45–60 min of CPU on this PC.
3. Then the 8 crash tests and the version-isolation test.
