# Phase 0: acceptance criteria (frozen 16 October 2026, before any agent code)

Every bar below is fixed **before** implementation. Changing one needs a new version of this file
and a dated entry in §9. Thresholds live in `policy.yaml` with their versions.

---

## 1. Run identity

```
run_id = sha256( input_sha256 + policy_sha256 + execution_manifest_sha256 )
```

- **`input_sha256`:** the EDF bytes (Path B) or the packet bytes (Path A).
- **`execution_manifest_sha256`:** from `results/golden/execution_manifest_v1.json`. It covers the staging
  checkpoint, every fitted table, the packet and prompt builders, serializer, grammar, verifier,
  renderer, report-model GGUF, the `llama-completion` binary and its arguments.
- **A model or software change makes a new execution identity**, so outputs are never reused across
  versions.
- **The policy hash is also recorded on its own in every trace entry,** so a policy change can be told
  apart from a model or software change.

## 2. Outcome categories (every night ends in exactly one)

| Outcome | Definition | Counts as a verified report? |
|---|---|---|
| **VERIFIED_REPORT** | The report model's claims passed the verifier **and** the final page passed V-HTML | yes |
| **FALLBACK_SUMMARY** | Deterministic template from packet values, labelled as such, plus a review item | no |
| **REVIEW_ONLY** | Processing couldn't safely produce a report; a notice plus a review item | no |
| **REJECTED** | The input failed the minimum requirements (unreadable, no EEG, too short) | no |

**Headline system metric:** verified-report coverage = VERIFIED_REPORT nights / **all** nights.

## 3. P1: integration parity (Path A) and recovery

| Check | Bar |
|---|---|
| Prompt | sha256 equal to the golden set for **31/31** nights |
| Report text | **Byte-identical** for 31/31 to a reference generated **in the same environment** through the agent's `llama-completion` wrapper (v3: `reference2_x86_reexec_v1.json` on this x86 host). Every difference from the frozen 4 Oct evaluation (`reference1`) is recorded, and each differing output must be checked by the verifier |
| Verifier outcome | Identical to the same-environment reference for 31/31 |
| Rendered HTML | sha256 identical to the same-environment reference for every passing night |
| Outcome category | As in the same-environment reference (reference 2: 31 VERIFIED_REPORT). Replay of the frozen cache is checked against reference 1 (30 VERIFIED_REPORT + 1 FALLBACK_SUMMARY, SC4582G0) |
| Trace | One trace entry per transition; the runner refuses a transition without one |
| Crash recovery | At each of the 8 boundaries (05 §7), `kill -9` → restart → **exactly one** delivered artefact, byte-identical to the uninterrupted run |
| Version isolation | A changed execution manifest (simulated by editing a copy of the manifest) produces a **new** run ID and **no** reuse of earlier outputs |

## 4. P2: corrected raw-EDF path (Path B)

### 4.1 Preprocessing parity on known data
Path B's spectral features, computed from the stored training tensors, must equal the stored
`processed_sleepedf/spectral/*.pt` (max |Δ| ≤ 1e-5) on all 31 dev recordings. This requires `pywt`.
Path B's EEG epochs, cut from the raw EDF with the training-rule span taken from the hypnogram,
must equal the stored tensors on all dev recordings. This proves the new preprocessing is the training
preprocessing before any estimator is involved.

### 4.2 Study-period estimator
- **Development:** tune on **train + validation** recordings only (raw EDF + hypnogram, never test).
  Then **freeze a version**, e.g. `period_estimator v0.1`, with its parameters in `policy.yaml`.
- **Acceptance set:** the 31 dev nights, evaluated with the **frozen** version.
  - If it fails, record the failure, revise using train/validation evidence only, and register a new
    version before re-evaluating.
  - If the dev set is used more than once, it is called a *development acceptance set* in every report,
    never independent evidence.
- **Coordinates:** every start/end index is expressed in **original-recording epochs**. Pass 2 works on the
  trimmed span, and its indices are mapped back (original = trimmed + trim start) before computing errors,
  durations or the stability flag.
- **No qualifying sleep run:** return `R04 period_not_found` and go to REVIEW_ONLY. The whole recording is
  **never** used silently.

| Quantity (31 dev nights, vs the hypnogram-based training rule) | Bar |
|---|---|
| Start error | median ≤ 10 epochs; 90th percentile ≤ 30 epochs |
| End error | median ≤ 10 epochs; 90th percentile ≤ 30 epochs |
| Total sleep time, absolute error | median ≤ 10 min |
| Sleep efficiency, absolute error | median ≤ 2 points |
| Staging κ on the reported span vs the hypnogram-span run | within 0.01 |
| Unstable nights (W08) and R04 nights | counted and listed |

### 4.3 Signal quality
- **Calibrated on train + validation only.** For EEG and EOG separately: unusable fraction and longest
  unusable run, per night.
- **Default policy is profile-agnostic.** It must not rely on knowing whether a night is "SC" or "ST",
  because a new device's recording carries no such label.
- **Cohort stratification is for evaluation and reporting only.** It shows how often each cohort is warned
  or rejected.
- **Acquisition-profile thresholds** are allowed only when a device profile is declared in its
  configuration **and** validated on that profile's data.
- **The goal isn't to make every ST night pass.** It is that a rejected or warned night must be measurably
  worse for staging than an accepted one: on train/val, nights flagged W02/Q01 must show lower κ than
  unflagged nights. Report the difference with a CI.

### 4.4 Path B scorecards
- **Model scorecard** on the dev nights: accuracy, κ, macro-F1, per-class P/R/F1 and ECE, by night tier and by quality band.
- **System scorecard**: §2 outcomes and coverage.
- **Differences from Path A are documented, not required to be zero.**

## 5. P3: report hardening

| Check | Bar |
|---|---|
| Runtime | `llama-completion` with exactly the §5 arguments of `CONTRACTS.md`; executable, GGUF and grammar sha256 checked before every launch |
| Timeout | Subprocess timeout (`policy.yaml`); a hung process is terminated, with no orphans left (checked) |
| Retry | At most `report.retries` attempts, only for a crash, timeout or missing end marker. **Never** for a verifier failure (that is deterministic at temperature 0) |
| Fallback | Every non-verified path ends in FALLBACK_SUMMARY or REVIEW_ONLY; 0 unverified claims shown |
| V-HTML | Every number on the delivered page comes from one of four sources: (1) a direct packet field; (2) a direct quality-sidecar field; (3) a **registered derivation** (`policy.yaml` `vhtml.derivations`: source field IDs + rule, e.g. fraction × 100 to 1 decimal + %, a fraction error in percentage points, `_num` formatting); (4) a **whitelisted constant** (dates, the run ID, units). Anything else blocks delivery and goes to REVIEW_ONLY. This applies to the report page **and** the fallback page |
| Packet integrity | The packet sha256 before generation = after verification = the one referenced on the page |

## 6. P4: whole-workflow evaluation on the PC
- Path A: §3 again, end to end. Path B: §4 on all 31 dev nights.
- The 15 fault scenarios (03 §5 L3), each ending in its expected outcome and reason codes.
- Both scorecards reported. Verified-report coverage is the headline, beside acceptance, review and fallback rates.

## 7. P5–P7: packaging, simulation, hardware (provisional targets, in the threshold register)

| Check | Bar |
|---|---|
| ARM64 staging parity | Probability max \|Δ\| ≤ 1e-4; identical hypnograms on dev nights |
| ARM64 report | 3 dev packets: same verifier outcome; output differences recorded |
| Replay | 7 nights in a row, 0 hangs, 0 duplicate deliveries; crash tests repeated |
| Pi 5 | P95 night ≤ 15 min; peak memory ≤ 10 GB; no throttling at 25 °C with the Active Cooler; cold and warm start, prompt and generation split, energy per night all **measured** |
| Pi report parity | Dev 5/5 count ≥ the x86 golden set (30/31); every Pi failure blocked by the verifier |

## 8. P8: freeze and the one test evaluation
- `frozen_edge.json` = execution manifest + policy hash + Pi build hashes.
- **One** logged evaluation on the locked test packets, to confirm the frozen Pi deployment reproduces the frozen system.
- **The test set stays out of the decision loop.** No model, threshold, routing or policy is chosen or
  changed because of it. A discrepancy is recorded as a failure, investigated on dev data, and fixed in a
  new version under this protocol.
- **v1 is the only initial release.** v2 (EEG + EOG + retrained report model) follows only after its own
  dev criteria pass and its execution manifest is frozen.

## 9. Change log
| Date | Change |
|---|---|
| 2026-10-16 | v1 of this file. Adopts reviews 1 and 2 (05, 06) |
| 2026-10-17 | **v2, user decision "A + C" after the estimator failed twice** (v0.1, v0.2; `agentic/reports/P2_STATUS.md`). **§4.2 input:** Path B is evaluated on **device-like input**: each recording cropped to [lights-off, lights-off + 12 h], with lights-off from Sleep-EDF's SC/ST-subjects.xls. The reference rule and the bar are **unchanged**, applied to the expert labels inside the crop. **New ingest guard:** recordings > 16 h → REVIEW_ONLY, code `R05 not_a_single_night`. Dev has now been used three times for this estimator, so it is a *development acceptance set* |
| 2026-10-17 | **User decision: estimator v0.3 accepted with a documented deviation.** On dev (development acceptance set, third use) it passes start (median 0 / p90 2), TST (median 0 min), SE (0.27 pts) and κ (−0.001), and **fails the end-point p90 (49 epochs vs ≤ 30)**: 6 nights have morning end errors of 34–125 epochs. Frozen as `study_period v0.3` (run 5, gap 8, merge 240, pad 60). Quality thresholds frozen from the train crop: warn 0.046, review 0.110 |
| 2026-10-24 | **v3, user decision D1 option A.** §3's parity bar becomes "byte-identical to a same-environment reference", the rule §7 already uses for the Pi. Built `reference2_x86_reexec_v1.json` from the real-model parity run of 17 Oct (`agentic.tools.rebaseline_reference`): 30 nights identical to reference 1; **SC4582G0 differs**, and its new output passes the verifier (0 codes, 5/5, fidelity 17/17), where the 4 Oct output failed with 3 × `L2.safe_item_hedged`. Reference 1 is untouched and stays the historical record: **v1 dev 30/31 and test 29/29 are not restated**. The parity tool checks real runs against reference 2 and replay against reference 1. Reference 2 still needs one fresh-runtime parity run to confirm same-environment reproducibility |
| 2026-10-24 | **The fresh-runtime parity run fails §3 v3: 29/31.** Generation isn't reproducible run to run on this host: SC4582G0 reproduced its 4 Oct output, and SC4111E0 changed. Both were caught by the verifier. **The bar is not changed;** the decision is reopened as D1b (`agentic/reports/P1_STATUS.md`) |
| 2026-10-24 | **Policy 0.2: line-noise check (pending item V5).** New quality codes **W10** (night median Welch P(44–50 Hz)/P(0.5–30 Hz) > 0.081, train p95, delivered with a review flag) and **Q03** (> 1.0, REVIEW_ONLY). Calibrated on train only; one dev evaluation: Q03 on SC4111E0 (κ 0.44, the lowest dev night) and W10 on SC4671G0 (`ROADMAP V5 - LINE NOISE/NOTES.md`). The policy hash changes, so every run ID changes; Path A outputs are unaffected |
| 2026-10-24 | **D1b cause found and fixed by user decision: release v1.1.** The Student's embedded chat template writes **today's date** into the prompt (`strftime_now`); v1 passed no pinned template, although training pinned "26 Jul 2024". Same-day runs are 12/12 identical, and setting the date reproduces 6/6 historical outputs byte for byte (`ROADMAP D1 - DETERMINISM PROBE/FINDINGS.md`). **v1.1** = v1 + `report.chat_template: student/templates/Llama-3.2-3B-Instruct.pinned.jinja`, hash-pinned, passed as `--chat-template-file` and re-checked before every launch. v1's identity is unchanged (`d68f96a6…`). §3's parity bar applies to v1.1 against **reference 3**, built from its first full dev run, and must hold on a later day. v1's historical results are not restated |
