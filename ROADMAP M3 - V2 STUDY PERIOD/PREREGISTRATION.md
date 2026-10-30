# M3: refit the study-period estimator on v2 predictions: pre-registration

**25 October 2026, written before any v2 prediction was computed.** Pending item M3.

**Question.** The v2 release (EEG + EOG staging) uses estimator v0.3 (run 5, gap 8, merge 240), tuned on
N4kd's predictions. Is it still right for v2, and does a refit help?

**What can change.**
- **Estimator (run, gap, merge):** depends on the staging model, so it is re-tuned here.
- **Quality warn/review fractions and the line-noise thresholds:** computed from the EEG signal on the
  expert reference spans of train nights. They don't depend on the staging model, so they are not
  refitted and carry over unchanged.

**Protocol (same as P2, `agentic/tools/p2_calibrate.py`).**
1. Pass 1 of the v2 adapter on the 137 **train** nights (device-like crop at lights-off + 12 h).
2. Tune on train with the same grid and the same selection rule as v0.3 (no sleep not-found, then
   minimal start-p90 + end-p90, then medians). **Freeze to `results/frozen_params.json`.**
3. Only then evaluate **once on dev**: the frozen refit (v0.4-candidate) and the current v0.3 parameters,
   both on v2 predictions, against the existing bars (ACCEPTANCE §4.2: start/end median ≤ 10 and p90 ≤ 30
   epochs, TST median ≤ 10 min, SE median ≤ 2 points, mean κ change within 0.01).
4. Test split never read. Dev is a reused development acceptance set (4th use for an estimator).

**Decision rule, fixed now.**
- **Adopt the refit** only if on dev it meets every bar v0.3 meets on v2 AND its end p90 is lower than
  v0.3's on v2 by at least 5 epochs.
- **Otherwise keep v0.3.** No further tuning on dev either way.
- **Release effect.** If adopted, it's a policy for release v2 only (v1/v1.1 and their references are
  untouched).
