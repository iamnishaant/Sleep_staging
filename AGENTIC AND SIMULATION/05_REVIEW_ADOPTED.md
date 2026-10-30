# 05 — External review adopted: amendments to the agentic and simulation plan

**16 October 2026.** An external review of documents 01–04 was received and is **adopted as written**.
Where it conflicts with 01–04, **this document wins.**

Some points needed updating with evidence measured *after* 01–04 were written (12–14 Oct, in
`ADDING EOG VERSION - PIPELINE INTEGRATION/`). Those updates are marked **[NEW EVIDENCE]**.

The review's verdict: approve with modifications. It found the deterministic design, verifier gate and
simulation ladder strong. It named the data path (study period + signal quality) as the main risk and
hardware performance as unproven. All of that is accepted.

---

## 1. Decisions taken

| # | Review point | Decision | Where it lands |
|---|---|---|---|
| D1 | Keep option A (deterministic state machine); no LLM routing or multi-LLM agents in the report path | **Adopted, unchanged** | 01 §4 stands |
| D2 | Study-period detection is the critical dependency; give it a formal acceptance threshold | **Adopted.** Definition and pass bar in §3 | Gates P2 |
| D3 | Quality thresholds are unvalidated; calibrate them; separate EEG/EOG; separate isolated from continuous artefacts; separate *warn* from *reject* | **Adopted.** Method in §4 | Gates P2 |
| D4 | Two kinds of parity: **integration parity** (identical inputs → identical outputs) vs **corrected-preprocessing evaluation** (new raw-EDF path judged on its own merits) | **Adopted.** Path A / Path B in §5 | P1, P2, P4 |
| D5 | Pi figures are hypotheses; benchmark prompt processing and generation separately; cold vs warm start; thermals; energy | **Adopted** | P7 protocol, §9 |
| D6 | A verified report isn't a clinically valid report; scope it as an offline sleep-wellness and reporting prototype | **Adopted.** Scope in §2 | All documents and the UI wording |
| D7 | Route only to a model validated for the exact channel configuration and preprocessing | **Adopted.** Routing table in §6 | P2 |
| D8 | Transactional recovery: stable run IDs, atomic writes, completion markers, idempotent delivery; crash tests at every boundary | **Adopted.** Design in §7 | P1, P6 |
| D9 | Validate evidence IDs, packet hashes, numeric provenance and the **final rendered HTML** | **Adopted** (new check V-HTML, §8) | P3 |
| D10 | Keep signal quality, model confidence, night confidence and report verification as **four separate signals** | **Adopted.** Quality travels in a sidecar, not in the frozen packet (§8) | P2, P3 |
| D11 | P3 spike: confirm the grammar-in-request mechanism on the pinned build | **Superseded [NEW EVIDENCE]:** the report runtime becomes `llama-completion`, not `llama-server` (§6) | P3 |
| D12 | Coverage is a first-class metric; don't optimise PASS rate alone | **Adopted** (§10) | P4, P7 |
| D13 | Separate model scorecard and system scorecard | **Adopted** (§10) | P4, P7, P8 |
| D14 | One canonical baseline table before P0 | **Adopted and filled in** (§11) | P0 |
| D15 | Every threshold is a versioned engineering decision with rationale and consequence | **Adopted.** Register in §12; `policy.yaml` hash changes with any threshold | All |
| D16 | No accelerator HAT; no speculative decoding or model swap before the unoptimised baseline is measured | **Adopted.** O3 (speculative decoding) is deferred until after P7's baseline | 03 §7 |
| D17 | Model changes during deployment work are controlled version changes, never silent swaps | **Adopted.** Model versions in §6 | P0, P8 |
| D18 | Start with a small, verifiable milestone, not the whole agent | **Adopted.** Checklist in §13 | Now |

---

## 2. Scope boundary (first deployment milestone)

**In scope:**
- offline sleep staging from supported recordings: EEG Fpz-Cz, with EOG when the EOG model version is enabled;
- confidence-aware, evidence-grounded reports;
- agentic workflow, audit and fault recovery;
- Raspberry Pi 5 deployment with **measured** resource use.

**Out of scope:**
- a conversational assistant;
- streaming (in-night) staging; in-night QC alerts are optional and later;
- apnea or other disorder risk prediction;
- any diagnostic claim.

**Wording rule** for the UI and documents: this is an **offline sleep-wellness and reporting prototype**.
A verified report means *every claim matches the evidence packet and its confidence rules*. It does not
mean the staging is correct, and it does not rule out a sleep disorder. Every report and fallback says so.

---

## 3. Study-period detection: definition and pass bar (gates P2)

**Training rule** (reference): first sleep epoch − 60 epochs (30 min) to last sleep epoch + 60 epochs,
clipped to the recording. "Sleep" means any non-W stage.

**Estimator v0.1 (two-pass baseline, not assumed correct):**
1. **Pass 1:** stage the whole recording with the routed model.
2. **Sleep onset:** the first run of ≥ 10 consecutive predicted non-W epochs (5 min). Inside the run, W
   gaps of ≤ 2 epochs are tolerated.
3. **Final awakening:** the same rule, applied backwards from the end.
4. **Pad** ±60 epochs and clip.
5. **Pass 2:** stage the trimmed span. Its predictions are the ones reported.
6. **If no qualifying run exists:** `R04 period_not_found`, the night goes to REVIEW_ONLY.

**Pass bar** (31 dev nights, against the hypnogram-based training rule; fixed before implementation):

| Quantity | Bar |
|---|---|
| Start-point error | median ≤ 10 epochs (5 min); 90th percentile ≤ 30 epochs (15 min) |
| End-point error | same |
| Total sleep time, absolute error | median ≤ 10 min |
| Sleep efficiency, absolute error | median ≤ 2 points |
| Staging κ on the reported span vs the hypnogram-trim run | within 0.01 |
| Nights that fail the per-night rule below | reported, not hidden |

**Per-night safety rule:** if either end-point moves by more than 30 epochs between pass 1 and pass 2
(the estimate is unstable), the night still goes out, but with code `W08 period_uncertain` and REVIEW.
The derived metrics are labelled "period estimated".

If the pass bar fails, the estimator is revised and re-tested on dev. Nothing downstream proceeds on a
failed estimator.

---

## 4. Signal quality: how thresholds are set (gates P2)

**Measured per channel** (EEG and EOG separately), per epoch:
- flat (std < 1 µV);
- stuck (≥ 1 s of identical samples, the existing `quality_mask` rule);
- clipping (≥ 0.5 s at the channel's digital rail);
- line-noise ratio (50/60 Hz power over 0.5–30 Hz power);
- amplitude out of range.

**Per night**, two numbers per channel: the **unusable fraction** and the **longest continuous unusable
run (min)**. This keeps isolated short artefacts apart from long failures.

**Calibration** (train + validation recordings, never test):

| Code | Meaning | Rule |
|---|---|---|
| W02 | warn | Unusable fraction above the 95th percentile of train/val nights, **or** a longest run ≥ 30 min. The report goes out with a quality note |
| Q01 | review | Unusable fraction above the maximum seen in train/val nights plus a margin, **or** a longest EEG run ≥ 120 min |
| Q02 | EOG unusable | EOG-specific. The night is routed to the EEG-only mode (§6), not rejected |

The draft 5% / 20% values in 02 §4 are replaced by these calibrated values. They are recorded in the
threshold register with the calibration data.

**Known fact to respect:** 29 of 30 ST training recordings contain stuck epochs; no SC recording does.
Thresholds must be read per cohort, so that ST nights aren't rejected wholesale.

**Signal quality is not trust.** A night that passes QC can still have low model confidence. Both are
reported, separately (§8).

---

## 5. Two kinds of parity (D4)

| Path | Input | What must hold | Used in |
|---|---|---|---|
| **A: integration parity** | Existing prebuilt dev packets, through the agent wrapper | **Byte-identical** prompts, claims, verifier results and rendered HTML vs the golden set | P1 exit, every later phase (regression) |
| **B: corrected preprocessing** | Raw dev EDFs, **no hypnogram given**, through QC → study period → resample → staging → packet | Judged on its own: the §3 study-period bar; staging κ vs expert labels on the reported span; report scorecard (§10). **Differences from Path A are expected and documented, not forced to zero** | P2 exit, P4 |

Path B intentionally differs from today's `pipeline.py` fresh path. That path trims to first-to-last
*annotated* stage, wake included, which doesn't match training (01 §3, G1). Matching a flawed baseline isn't a goal.

---

## 6. Model versions, routing, and the report runtime

### Model versions (D17)
| Version | Staging | Report model | Status | Evidence |
|---|---|---|---|---|
| **v1** | N4kd (EEG) | Llama-3.2-3B-Student (frozen) | **Deployment baseline for P0–P7** | Test 29/29 at 5/5, 0 violations; dev 30/31 |
| **v2** | Stage 3 EEG + EOG student | Llama-Student **retrained on v2 packets** | Planned **controlled version change**, after v2 passes its own dev bar | Staging test κ 0.756. With the *un*-retrained report model: dev 30/31, test 27/29 (1 blocked) |

The agent is built and proven on v1. v2 enters through the manifest, with its own dev evaluation. No
silent swap.

### Routing table (D7): only validated configurations
| Available input | Route | Validated? |
|---|---|---|
| EEG Fpz-Cz at 100 Hz (after resampling) | v1 N4kd, or v2 with **EOG off** | v1: yes. v2 EOG-off: measured on test, κ 0.744 [M], and in CV ablations |
| EEG Fpz-Cz + EOG horizontal, EOG passes QC | v2 (EOG on) | Yes, when v2 is enabled |
| EOG present but fails QC (Q02) | v2 with EOG off | Yes (as above) |
| Fpz-Cz absent (any other EEG lead) | **REVIEW_ONLY** (`W05`) | No model is validated on another lead |
| Sampling rate other than 100 Hz | Resample, then route | The resampling path must pass the Path B bar |

### Report runtime: **[NEW EVIDENCE]** replaces the grammar-in-request spike (D11)
On 13 Oct the same frozen model, run two ways, was compared on the same dev packets:

| Runtime | Dev at 5/5, 0 violations | Outputs identical to the frozen evaluation |
|---|---:|---:|
| `llama-completion`, one process per packet (the frozen evaluation's own path) | **30/31** | **yes, byte for byte** |
| `llama-server`, default template (what `deploy/server.py` uses) | 20/31 | no (8 of 31 differ) |
| `llama-server`, pinned date template | 23/31 | no |

**Decision:** the agent's REPORT step uses `llama-completion` exactly as in `candidates/run.py`.
- The grammar is passed per process, so G6 ("is the grammar active?") is solved by construction.
- `llama-server` may only be reintroduced after it reproduces the frozen dev outputs.
- The cost is a model load per night: about 3 s on x86 from the page cache [M]. On the Pi from NVMe it
  is to be measured; this is the "cold start" number in D5.

---

## 7. Transactional run design (D8)

- **Run ID** — *superseded by 06*: `sha256(input + policy + execution_manifest)`, so a model or software
  change can never reuse earlier outputs. The same recording under the same policy and manifest is the same
  run, so a repeated trigger never produces a second report.
- **Layout:** `runs/<run_id>/step_<n>_<name>/` holds the outputs plus a `DONE` marker.
- **Atomic writes:** every output is written to `*.tmp`, flushed and fsynced, then renamed with `os.replace`.
  The `DONE` marker is written last, the same way.
- **Resume:** skip every step that has a `DONE` marker. Re-run the first step without one, from scratch,
  deleting any partial `*.tmp` files.
- **Idempotent delivery:** `outbox/<run_id>.html` is written atomically, then `runs/<run_id>/DELIVERED`.
  If `DELIVERED` exists, delivery is a no-op. History-store appends are keyed by run ID (upsert, not append).
- **Crash tests** (`kill -9` injected at each boundary; expected: exactly one delivered artefact, with the
  same content as an uninterrupted run):
  1. after INGEST;
  2. mid-PREPROCESS;
  3. after STAGE outputs, before `DONE`;
  4. during REPORT generation;
  5. after the LLM output, before verification;
  6. after verification, before rendering;
  7. after rendering, before delivery;
  8. after delivery, before `DELIVERED`.

---

## 8. Four separate signals, and final-output validation (D9, D10)

| Signal | Question | Where it lives |
|---|---|---|
| **Signal quality** | Is the recording technically usable? | Agent QC → `runs/<id>/quality.json` (sidecar) |
| **Model confidence** | How sure is the model, epoch by epoch? | Packet: calibrated probabilities, per-stage reliability, N1 flag |
| **Night confidence** | How reliable is this night's staging overall? | Packet: `night_confidence` tier |
| **Report verification** | Does every claim pass evidence and policy checks? | Verifier result → trace |

**Why quality is a sidecar.** The frozen serializer builds the prompt only from `evidence_items` and the
night tier, with a fixed evidence-ID vocabulary. So adding quality to the packet would mean a new schema
version and new validation of the report model. Instead, the agent renders quality in a wrapper section
of the delivered page, outside the verified claims, clearly labelled.

**V-HTML (new final check).** The delivered HTML is parsed. Every number in it must be either a packet
evidence value (verified claims), a sidecar value (quality section), or a whitelisted constant (dates,
units). Its packet hash must equal the verified packet's hash. Any mismatch blocks delivery and goes to review.

---

## 9. Pi measurements (adds to 03 §6)
- Prompt processing and generation timed separately (llama.cpp perf lines).
- Cold start (first run after boot) vs warm start (page cache primed).
- Thermal: temperature trace and throttle flags through each night.
- Energy per night (USB-C meter).
- The budget is recomputed if the report model, context length or output length changes.
- The 15-min P95 limit is a **provisional engineering target**, recorded in the threshold register.

---

## 10. Two scorecards and coverage (D12, D13)

**Model scorecard** (staging):
- accuracy, κ, macro-F1;
- per-class P/R/F1 and calibration (ECE);
- each of these by night-confidence tier and by quality band.

**System scorecard** (per night set):

| Metric | Definition |
|---|---|
| Recording acceptance rate | Nights not REJECTed / all nights |
| Usable-epoch coverage | Usable epochs / epochs in the reported span (EEG; EOG separately) |
| Report delivery coverage | Nights that delivered a verified report **or** a fallback summary / all nights |
| **Verified-report coverage** | Nights that delivered a verified report / all nights (the honest headline, not the PASS rate among accepted nights) |
| Review rate | Nights with any review item / all nights |
| Fallback rate, crash count, recovery success | — |
| Latency (P50/P95), peak memory, energy | — |

---

## 11. Canonical baseline table (D14): what each existing number means

| Result | Unit | Split, n | Model (checkpoint) | Input | Hypnogram used in preprocessing? | Hardware / runtime | Value |
|---|---|---|---|---|---|---|---|
| Staging, stored labels | epochs | test, 29 rec | N4kd | prebuilt tensors | yes | x86, PyTorch | acc 77.73%, κ 0.700 (decoded 77.69%, 0.699) |
| Staging, corrected labels | epochs | test, 29 rec | N4kd | prebuilt tensors | yes | x86, PyTorch | acc 79.00%, κ 0.717 (decoded) |
| Staging, corrected labels | epochs | test, 29 rec | Stage 3 `full` s1 (`61e94bfc…`) | prebuilt tensors | yes | x86, PyTorch | acc 81.96%, κ 0.756 |
| Report | nights | dev, 31 | v1 packets + Llama-Student (`c1962812…`) | prebuilt packets | yes | x86, `llama-completion` b10927 | 30/31 at 5/5; the other night (SC4582G0) blocked by the verifier, 3 × `safe_item_hedged` |
| Report | nights | test, 29 | v1 | prebuilt packets | yes | x86, `llama-completion` | 29/29, 0 violations, fidelity 1.000 |
| Report | nights | dev, 31 | v1 | prebuilt packets | yes | x86, `llama-server` | 20/31 (default template), 23/31 (pinned) |
| Report | nights | dev / test | Stage 3 packets + un-retrained Llama-Student | prebuilt packets | yes | x86, `llama-completion` | dev 30/31; test 27/29, 1 blocked |
| Pi 5 anything | — | — | — | — | — | — | **estimate only** (6–9 min per night, 6–8 GB) |

**No raw-EDF, no-hypnogram (Path B) result exists yet.** That's P2's job.

---

## 12. Threshold register (D15): every value is a versioned decision

| Threshold | Draft value | Rationale | Calibration source | Consequence of crossing | Status |
|---|---|---|---|---|---|
| Sustained-sleep run | ≥ 10 epochs, gaps ≤ 2 | Ignores brief transitional sleep at lights-off | Dev-night pass bar (§3) | Defines the study period | provisional |
| Period instability | > 30 epochs | Unstable two-pass estimate | Dev | W08 + review | provisional |
| Quality warn / review | p95 / max + margin of train-val; runs 30 / 120 min | Data-driven, per channel | Train + validation | W02 / Q01 | to calibrate |
| SoC temperature | 70 °C | Below the Pi 5's throttle onset, with margin | Pi measurements (P7) | Wait | provisional |
| Free RAM | 6 GB | Report model peak (~4.7 GB on x86) plus margin | Pi measurements | Wait | provisional |
| Report window | 01:00–07:30 | Overnight batch | User need | Fallback after the window | provisional |
| LLM timeout | 1,200 s | About 2× the estimated Pi report time | P7 | Restart once → fallback | provisional |
| P95 night latency | 15 min | Morning-ready | P7 | Optimisation trigger | provisional target |
| Peak memory | 10 GB | Leaves 6 GB headroom on 16 GB | P7 | Fail P7 | provisional target |

`policy.yaml` carries each value **with its version**. Changing any value changes the policy hash,
which is recorded in every trace entry.

---

## 13. Revised exit criteria and the immediate checklist

| Phase | Exit (amended) |
|---|---|
| **P0** | Golden set for v1 on 31 dev nights; canonical baseline table (§11); every pass bar in §3, §4, §7, §9 and §12 written down **before code**; no existing file changed |
| **P1** | Path A byte-identical on all 31 dev nights; the 8 crash points (§7) each give exactly one delivery; every transition traced |
| **P2** | §3 study-period bar passed; quality thresholds calibrated and registered; routing table enforced; Path B scorecards produced |
| **P3** | `llama-completion` runtime; every server/LLM/verifier failure reaches a defined state; V-HTML passes; 0 unverified claims shown |
| **P4** | Path A and Path B on all dev nights; the 15 fault scenarios; both scorecards; coverage reported, not only PASS rate |
| **P5–6** | x86 and ARM64 images; staging parity on ARM; crash tests repeated in the replay; 7 replayed nights, no hangs, no duplicates |
| **P7** | Measured cold/warm latency, prompt/generation split, thermals, energy, memory; every §12 hardware target checked; optimisations only after the baseline |
| **P8** | `frozen_edge.json`. **One** logged test evaluation of the frozen Pi deployment (deployment parity on the locked test packets). Nothing is changed afterwards on the basis of it |

**A note on test use.** The locked test packets have already been evaluated for v1 (29/29) and for the
Stage 3 packets (27/29). The P8 run checks that the *Pi build* reproduces a frozen system, and is logged.
It must not feed back into any model, threshold or policy choice.

### Immediate checklist (all CPU, no Kaggle)
1. Document the input/output contracts of preprocess, stage, packet, report and verify (as-is).
2. Build the v1 golden set on 31 dev nights. The frozen evaluation's cached outputs already exist
   (`candidates/cache/Llama-Student/P1/`), so this is mostly collection and hashing.
3. Write the pass bars and the threshold register into `policy.yaml` v0.1 plus an `ACCEPTANCE.md`.
4. Agent skeleton: `state.py`, `policy.py`, `runner.py`, thin wrappers, separate dev audit log.
5. Path A parity and the 8 crash-point tests, before any new preprocessing.
6. QC + study-period estimator; calibrate on train/val; run the §3 bar on dev.
7. Report hardening: `llama-completion` runtime, timeouts, fallback, V-HTML, atomic delivery.

The 24–36 working-day envelope in 04 stays **provisional** until P2 (study period) and P7 (real board)
have been measured.
