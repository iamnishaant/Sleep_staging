# Phase 2 status (17 Oct 2026)

## Done and passing

| Check | Result |
|---|---|
| §4.1 preprocessing parity on the 31 dev recordings, raw EDF → EEG epochs → 34 features → N4kd probabilities | **31/31 exact** (EEG, features and probabilities all with max \|Δ\| = 0). `p2_parity.json` |
| N4kd staging adapter (`staging/n4kd.py`), signals (`signals.py`), study-period functions (`study_period.py`) | built and unit-tested |
| Quality thresholds calibrated on **train** reference spans | warn (W02) = unusable fraction > **0.046** (train 95th percentile); review (Q01) = > **0.145** (train max + 0.05) |
| Quality flags on dev (descriptive) | W02 on 3 ST nights, κ 0.61 / 0.54 / 0.49. Q01 on 2 nights. The formal flagged-vs-unflagged CI test is still to do |

**Process note.** The 31 "dev" nights *are* the validation split, so all tuning used the 137 **train**
recordings only.

## Study-period estimator: FAILED its bar twice (ACCEPTANCE §4.2)

| Version (tuned on train only) | Start err median / p90 | End err median / p90 | TST err median | SE err median | κ change | Pass? |
|---|---:|---:|---:|---:|---:|---|
| Bar | ≤ 10 / ≤ 30 | ≤ 10 / ≤ 30 | ≤ 10 min | ≤ 2 pts | within 0.01 | |
| v0.1: first-to-last sustained run (run 10, gap 4) | 2 / **186** | 3 / **529** | 1.5 | 0.4 | **−0.020** | **no** (`p2_calibration_v0.1_FAILED.json`) |
| v0.2: main sleep episode (run 5, gap 1, merge 240) | 8 / **569** | 2 / **81** | 1.5 | 0.6 | **−0.022** | **no** (`p2_calibration.json`) |

The medians are excellent; the failures are a tail of a few nights that are hours off. Dev has now been
used twice, so it is a *development acceptance set*, not independent evidence.

### Why: diagnosed on TRAIN data
- Sleep-EDF cassette (SC) recordings last about **20 hours**, including the daytime.
- Of the 54 train nights with an error > 30 epochs, **30 contain expert-scored sleep hours away from the main
  night** (naps; sometimes only 2–8 epochs). The reference rule ("first to last expert sleep ± 30 min")
  then spans the nap, the day between, and the night.
- The main-episode estimator finds the actual night. The *reference* it's scored against includes the naps.
- The other ~24 large-error nights are genuine estimator or model errors: the model calls daytime wake
  sleep, or a night broken by long wake.
- A real device records **one night**, so these 20-hour inputs aren't device-like. ST recordings (single
  nights) mostly have 0-epoch errors.

## Decision needed (nothing changed yet)

| Option | What changes | Notes |
|---|---|---|
| **A. Device-like input** (recommended) | Crop each recording to the night, using the **lights-off time from `SC-subjects.xls` / `ST-subjects.xls`** (shipped with Sleep-EDF), as a device starting at bedtime would. Re-tune on train, evaluate on dev | The bar and the reference rule are unchanged; only the input becomes realistic. Needs a dated ACCEPTANCE change |
| B. Change the reference | Reference = the expert **main** sleep episode ± 30 min, on the full 20-h files | Moves the target after seeing failures. Weaker |
| C. Ingest guard only | Recordings longer than about 16 h go to REVIEW_ONLY ("not a single night"). Evaluate the estimator only on night-length recordings | Can be combined with A |

## Update 17 Oct: option A + C applied, estimator v0.3

**Input:** each recording cropped to [lights-off, lights-off + 12 h] (ACCEPTANCE v2). **Tuned on train only:**
run 5, gap 8, merge 240. Dev used for the **third** time, so it is a development acceptance set.

| v0.3 on dev | Value | Bar | |
|---|---:|---:|---|
| Start err median / p90 | 0 / 2 | ≤ 10 / ≤ 30 | pass |
| End err median / p90 | 3 / **49** | ≤ 10 / ≤ 30 | **fail (p90)** |
| TST abs err median | 0.0 min | ≤ 10 | pass |
| SE abs err median | 0.27 pts | ≤ 2 | pass |
| Mean κ change | −0.001 | within 0.01 | pass |
| R04 / W08 | 0 / 1 | reported | |

Six nights have an end error of 34–125 epochs (SC4351F0, SC4561F0, SC4562F0, SC4671G0, SC4672G0,
ST7151J0). That's the morning end, where wake after final awakening gets added or cut. It barely moves
the reported quantities (TST, SE, κ). The W08 instability rule caught only 1 of the 6.

Quality thresholds (train, cropped): warn = 0.046, review = 0.110. Dev W02 nights: 4 (all ST).

## 17 Oct: v0.3 accepted with a documented deviation; Path B built; scorecards (real model)

**Decision (user):** accept v0.3 with the end-point p90 deviation (ACCEPTANCE change log). Policy frozen:
estimator v0.3 (run 5, gap 8, merge 240, pad 60), quality warn 0.046 / review 0.110, ingest guard 16 h (R05).

**Path B workflow** (`workflows/edf_report.py`):
ingest → preprocess → stage_pass1 → study_period → quality → stage_pass2 → build_packet → (Path A report steps)
→ annotate (quality note) → vhtml → deliver. 42 tests pass. The 20-h file goes to R05; an unreadable file to R01.

### System scorecard: 31 dev nights, device-like input, no hypnogram, real llama-completion

| Outcome | Nights |
|---|---:|
| VERIFIED_REPORT | **31 / 31** (verified-report coverage 100%) |
| FALLBACK / REVIEW_ONLY / REJECTED | 0 / 0 / 0 |
| Delivered with a review flag (W02 × 4, W08 × 1) | 5 (review rate 16%) |
| Latency per night (x86, end to end) | median 81 s, P95 114 s |

### Model scorecard: staging on the estimated span vs expert labels

| | Accuracy | κ | Macro F1 | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pooled, 30,805 epochs | 76.7% | 0.682 | 73.8 | 81.7 | 51.8 | 81.6 | 75.0 | 79.1 |
| SC | 77.1% | 0.690 | 74.7 | | | | | |
| ST | 75.2% | 0.648 | 71.0 | | | | | |

- **The same model on the expert-rule spans** gives a per-night mean κ of 0.674. On the estimated spans
  it's 0.673. **The estimator costs essentially nothing.**
- **The 0.722 headline (N4kd, validation, corrected labels)** was measured on the *stored training
  spans*: 34,275 epochs against 30,805 here, mostly extra wake. Different spans, so the figures aren't
  directly comparable.
- **Flagged vs unflagged nights:** mean κ 0.587 for the 5 flagged nights (W02/W08) vs 0.689 for the rest.
  The flags do pick out worse-staged nights. n = 5, so no CI is claimed.

**P2 exit:** study period (accepted with deviation) ✓ · quality thresholds calibrated and registered ✓ ·
routing (v1: Fpz-Cz only, other leads → W05) ✓ · Path B scorecards ✓.
**Open:** real-model crash tests (user's reminder) and the P1 parity decision (`P1_STATUS.md`).

---

## 28 Oct: release v2 Path B scorecards on the complete dev set (31/31)

**Why this run.** The 27 Oct v2 scorecard covered 23/31 nights: seven raw EDFs were absent locally and
SC4081E0 failed with `ValueError: The length of the input vector x must be greater than padlen` (a
2.1 MB truncated copy of a 22 MB recording). The full Sleep-EDFx download is now at
**`D:\sleep-edfx\sleep-edfx`** (153 SC + 44 ST PSGs, plus `SC-subjects.xls` / `ST-subjects.xls`); all 31
dev PSGs are present at 22.1–52.1 MB. Both scorecards were re-run from that root.

**Commands**

```bash
python -m agentic.tools.p2_scorecard_staging --release v2 --raw "D:/sleep-edfx/sleep-edfx"
python -m agentic.tools.p2_scorecards --release v2 --backend llama \
    --raw "D:/sleep-edfx/sleep-edfx" --policy <scratch policy, see below>
```

### System scorecard (real `llama-completion`, 31 nights, device-like crop, no hypnogram)

| Outcome / metric | **v2 (28 Oct)** | v1 (16 Oct, reference) |
|---|---:|---:|
| VERIFIED_REPORT | **30 / 31 (96.8%)** | 31 / 31 (100%) |
| FALLBACK_SUMMARY / REJECTED | 0 / 0 | 0 / 0 |
| REVIEW_ONLY | **1** (SC4111E0, `Q03` line noise) | 0 |
| Delivered with a review flag | 6 (W08 ×1, W10 ×1, W02 ×4) | 5 (W02 ×4, W08 ×1) |
| Review rate | 22.6% (7/31) | 16.1% (5/31) |
| Latency, median / p95 | 79.4 s / 95.3 s | 80.9 s / 114.2 s |
| Absent locally | **0** | 0 |

SC4111E0 is the **lowest-scoring night in v2 (κ 0.495)** and is the only `Q03` night. The 24 Oct line-noise
check (`ROADMAP V5 - LINE NOISE/NOTES.md`) routes the worst night to review without human input — the
first time that has been observed on the complete dev set.

### Model scorecard — system run (estimated span vs expert labels)

v2 is scored on the 30 nights that reach pass 2; SC4111E0 stops at the quality gate.

| | **v2 (30 nights, 29,624 epochs)** | v1 (31 nights, 30,805 epochs) |
|---|---:|---:|
| Accuracy | **0.7945** | 0.7667 |
| κ | **0.7187** | 0.6822 |
| Macro-F1 | **0.7685** | 0.7383 |
| W / N1 / N2 / N3 / REM F1 | 0.844 / 0.551 / 0.835 / 0.756 / 0.857 | 0.817 / 0.518 / 0.816 / 0.750 / 0.791 |
| SC κ | 0.7250 | 0.6896 |
| ST κ | 0.6908 | 0.6482 |

### Model scorecard — staging only, all 31 nights, same estimator (release comparison)

Running both releases through `p2_scorecard_staging` on identical inputs and spans isolates the model
change from the quality-gate routing difference above. Values are per-night means.

| | v1 | **v2** | Δ |
|---|---:|---:|---:|
| κ (mean per night) | 0.6728 | **0.7014** | **+0.0286** |
| Accuracy | 0.7696 | **0.7900** | +0.0204 |
| Macro-F1 | 0.6897 | **0.7174** | +0.0277 |
| SC κ (23 nights) | 0.6851 | **0.7088** | +0.0237 |
| ST κ (8 nights) | 0.6377 | **0.6803** | **+0.0426** |
| W / N1 / N2 / N3 / REM F1 | 0.776 / 0.496 / 0.809 / 0.565 / 0.802 | 0.821 / 0.545 / 0.820 / 0.567 / 0.834 | |

The EOG input helps most where the single-channel model was weakest: **W +0.046, N1 +0.049, REM +0.032,
ST κ +0.043**. N3 is flat (+0.002), consistent with delta power already being available in the 34 features.

### Quality band

| Flag set | flagged nights | κ flagged | κ unflagged |
|---|---:|---:|---:|
| W02 / W08 (scorecard definition) | 5 | 0.6079 | 0.7284 |
| + W10 (policy 0.2) | 6 | 0.6206 | 0.7302 |

The flags still pick out worse nights, now with n = 6 rather than n = 5. **Gap:** the scorecard's flagged
set predates `W10`, so the automated number excludes it; the W10-inclusive row above is computed from the
row data, not by the tool.

### Reproducibility

The 23 nights scored on 27 Oct **reproduce exactly** on 28 Oct — same outcome, same reason codes, κ equal
to 1e-12 — from a different dataset copy and under a different policy hash. Staging and report content are
therefore stable across days and hosts; this is corroborating evidence for V4, whose dedicated check is
already `reports/v4_reproducibility.json`.

### Run conditions (recorded, not default)

- **RAM floor overridden for this run.** The host had **5.65 GB** available against the provisional 6 GB
  `resources.min_free_ram_gb` floor, so the report model would have waited and then fallen back `F01`.
  The run used a scratch copy of `config/policy.yaml` with **only** that value lowered to **5.0 GB**
  (report-model peak is ~4.9 GB). The policy hash changes, so these runs have **new run IDs** and are not
  the frozen-policy identities. This is PENDING V7: the 6 GB floor is too high for the 16 GB dev PC.
- **Tool additions:** `p2_scorecards` gained `--raw` and `--policy`; `p2_scorecard_staging` gained `--raw`
  (both default to the old hard-coded path, so existing invocations are unchanged). `p2_parity.lights_off_epoch`
  takes an optional raw root. Fixed a subset-run crash in `p2_scorecards` (`by_cohort` called
  `np.concatenate` on an empty cohort list) and a self-inconsistent retention test (it stamped the
  "recent" run with the same 10-day-old mtime it expected to survive). Test suite: 60 passed, 4 skipped.

