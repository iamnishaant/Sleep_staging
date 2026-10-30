# V5: Line-Noise Quality Check

**24 October 2026.** Pending item V5. Fault scenario S14 (strong 49 Hz noise) passed only because
nothing checked for it.

## 1. Does line noise matter? Yes (`noise_effect.py`, 8 train nights)

**Where noise can enter.** Input is staged at 100 Hz. Other rates are resampled with
`mne.filter.resample` (FFT, anti-aliased), which removes 60 Hz. So only noise at or near **50 Hz** (the
Nyquist frequency) in native 100 Hz recordings reaches the model.

| Added noise (× EEG RMS) | 49 Hz: mean κ | 50 Hz (Nyquist): mean κ |
|---|---:|---:|
| none | 0.775 | 0.775 |
| 0.25 | 0.703 | 0.684 |
| 0.5 | 0.685 | 0.672 |
| 1.0 | 0.340 | 0.623 |
| 2.0 | 0.072 | 0.470 |

**Why this matters.** The frozen verifier can't catch this, since it checks claims against the packet,
not the packet against the recording. S14 produced a VERIFIED_REPORT from badly staged input. Input QC
is the only defence.

## 2. Metric, calibrated on train (`calibrate_metric.py`, `results/metric_train.json`)

- **Per epoch:** Welch power in 44–50 Hz / power in 0.5–30 Hz.
- **Per night:** the median over the span.

| Train, 137 real nights | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| Night median ratio | 0.004 | 0.030 | 0.081 | 0.789 | 13.3 |

| Injected on SC4001E0 | ×0.1 | ×0.25 | ×0.5 | ×1.0 |
|---|---:|---:|---:|---:|
| 49 Hz | 0.025 | 0.142 | 0.555 | 2.21 |
| 50 Hz | 0.047 | 0.279 | 1.106 | 4.41 |

**On real recordings the metric tracks staging damage.**
- Spearman ρ = −0.27 against κ (p = 0.002).
- Nights above 0.08: mean κ 0.657 (n = 7), against 0.755 for the rest.
- Nights above 0.79: mean κ 0.513 (n = 2).

## 3. Rule, FROZEN on train (24 Oct, before any dev read for V5)

| Code | Fires when the night median ratio is | Action | Basis |
|---|---|---|---|
| **W10** "line noise near 50 Hz" | > **0.081** | Report delivered with a review flag (like W02) | Train p95, the existing quality-warn convention |
| **Q02** "line noise too strong to stage" | > **1.0** | REVIEW_ONLY (like Q01) | Above all train nights but one (SC4762E0, κ 0.46); injected noise at this level costs 0.15–0.43 κ |

- **Computed on** the estimated span, as the quality step does.
- **Release impact:** a new policy version, so Path B run IDs change. Path A is unaffected.

## 4. Dev evaluation of the frozen rule (`dev_eval.py`, `results/dev_eval.json`; one run, 24 Oct)

| Night | Ratio | Code | Path B κ |
|---|---:|---|---:|
| **SC4111E0** | 3.06 | **Q02**, REVIEW_ONLY | **0.442**, the lowest on dev |
| SC4671G0 | 0.149 | W10, delivered with a review flag | 0.653 |
| The other 29 | ≤ 0.08 | — | mean 0.682 |

**Flagged nights:** mean κ 0.547, against 0.682 unflagged.

**SC4111E0 explains several earlier findings.** It's the night where:
- the model scored 3 h of expert wake as REM (T1: TIB +158 min);
- arm C of R2 released an unreliable value as fact;
- the 24 Oct parity run's output flipped and failed.

Strong 50 Hz-band contamination is a likely common cause. Under the new rule this night is withheld for
review instead of being reported.

**Decision:** adopt W10/Q02 as a new policy version for Path B.
