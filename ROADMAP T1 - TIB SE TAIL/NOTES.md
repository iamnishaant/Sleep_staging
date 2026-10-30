# T1: Time-in-Bed and Sleep-Efficiency Tail on Path B

**Started 24 October 2026.** Roadmap task T1 (`COMPARISON WITH EXISTING SYSTEMS/ROADMAP.md`).

## Step 1 result: the SC4111E0 discrepancy (resolved)

**Question.** `agentic/reports/p2_calibration.json` gives SC4111E0 an end error of 316 epochs, but
`agentic/reports/P2_STATUS.md` lists only "six nights 34–125 epochs off". Is the reference wrong (for
example a nap inside it), or is this a real error?

**Check.** From the cached crop (`agentic/runtime/p2_cache_crop/SC4111E0-PSG.npz`):

| Period | Expert labels | Model (pass 1) |
|---|---|---|
| Epochs 962–1338: after the expert's final sleep, to the estimated end | **376 W** | 169 W, **139 REM**, 48 N2, 20 N1 |
| Epochs 1338–1440: to the end of the 12-h crop | 102 W | — |

- Estimated span [92, 1338]; reference span [93, 1022].
- TST: 503.5 min estimated vs 434.0 min reference, so +69.5 min.

**Conclusion.**
- **It's a genuine staging error,** not a nap and not a reference artefact. The model scores over 3 hours
  of expert-scored morning wake as largely REM. That's the EEG-only wake-vs-REM confusion.
- **The span grows with it.** The estimator then extends the night, so on this one night TIB is +158.5 min,
  TST +69.5 min and SE −12.6 points.
- **Nothing flagged it.** W08 didn't fire.
- **P2_STATUS's list is incomplete.** Seven dev nights have end errors above 30 epochs:
  SC4111E0 (316), SC4351F0 (125), ST7151J0 (105), SC4562F0 (−49), SC4561F0 (−46), SC4671G0 (−34)
  and SC4672G0 (−34). The p90 figure (49) is unaffected.

**Implications for T1.**
- **Two error sources:** the tail mixes boundary error with staging error. SC4111E0 is mainly staging
  (wake called REM); the five negative-error nights are boundary truncation. The boundary-uncertainty
  signal should target the boundary cases, and staging uncertainty is a separate signal.
- **A possible v2 benefit:** v2 (EEG + EOG) is the natural fix for SC4111E0-type errors, since eye
  movements separate REM from wake. Check this on train under T1, not by tuning on dev.

## Correction to the starting figures

The "TIB error" figures first quoted in the README (median 1.5, p90 24.5, max 158.5 min) were computed
as TST / SE from `p2_calibration.json`. There, SE = TST / span length, so the figures are
**study-period length** errors, an upper bound on TIB error.

The packet's TIB (`physiological_features.extract_features`, `lights_off=0`) runs from the span start
to the **last predicted sleep epoch** + 1, so trailing wake doesn't count. Step 2 computes that
definition directly, from pass-2 predictions on the raw EDFs (available at
`D:/sleep-edfx/sleep-edfx`).

## Steps 2–3 result: train, packet values (`tib_on_train.py`, `results/tib_train.json`)

- **Coverage:** 137 train nights, 136 scored. SC4761E0 failed the frozen builder's rounding self-check
  (`PENDING.md` K4; the agent now ends such nights as REVIEW_ONLY R12).
- **Values** come from the agent's own packet builder, so these are the packet's TIB, TST and SE.

| Absolute error | TIB, min | TST, min | SE, points |
|---|---|---|---|
| **Boundary** (est − ref, same model) median / p90 / max | 0 / 12.5 / 372.5 | 0 / 4.3 / 187 | 0 / 1.5 / 31.5 |
| **Staging** (ref − expert) median / p90 / max | 0.5 / 22.5 / 173 | 13 / 38.5 / 75.5 | 2.5 / 7.9 / 18.8 |
| **Total** (est − expert) median / p90 / max | 1.5 / 39.8 / 381.5 | 13.5 / 44.5 / 194.5 | 2.8 / 8.5 / 31.1 |
| Total mean relative error | **3.5 %** | **5.1 %** | **5.2 %** |

**Step 3, the tiers.**
- **All three stay "robust" under Path B** by the existing rule (mean relative error ≤ 10 %). No tier
  changes, so TIB, TST and SE remain mandatory items stated plainly, justified by the project's own rule.

**The tail.**
- **16 of 136 nights (12 %) have a total TIB error above 30 min.** The tier rule is a mean, so it doesn't
  see them. 9 come from the boundary (5 truncations, 4 extensions) and 7 from staging.
- **The staging part exists in v1 Path A too.** The validation tier was measured with the same kind of
  errors; it isn't a Path B problem.
- **Worst boundary case:** SC4601E0, 263 vs 645 min. The estimator kept only part of a broken night.

## Step 4: boundary-uncertainty signal, FROZEN on train (24 Oct, before any dev read for T1)

**Candidates (computed from pass 1, so no ground truth is needed at run time):** predicted sleep outside
the estimated span, and the longest predicted sleep run outside the span.

| Rule on train (136 nights) | Flags | Catches total TIB > 30 min | Catches boundary > 30 | Flagged but TIB within 30 min |
|---|---:|---:|---:|---:|
| Longest predicted sleep run outside the span > 10 min | 4 | 3 / 16 | 3 / 9 (3 of 5 truncations) | 1 |

**Frozen rule:** *W09, "sleep predicted outside the estimated night"*, fires when the longest run of
pass-1-predicted sleep outside the estimated span exceeds **10 minutes**. 10 min is the smallest threshold
on train with at most one flagged night whose TIB was fine.

**What it is for:** **truncation** of a broken night. It doesn't detect extensions or staging errors,
and isn't claimed to.

**Action when it fires:** the same as W02 and W08. The report is delivered with a review flag, and the
packet and verifier are unchanged. TIB is still stated plainly, because the tier rule supports it on
average.

**Effects:** W09 needs a policy version bump (a new code and threshold), so Path B run IDs change. It
has no effect on Path A.

## Step 5 result: dev, one evaluation of the frozen rule (24 Oct; `results/tib_dev.json`)

| Dev, 31 nights, total error | TIB, min | TST, min | SE, points |
|---|---|---|---|
| Median / p90 / max | 1.5 / 40.5 / 163.5 | 13.0 / 51.5 / 98.5 | 3.4 / 7.8 / 11.8 |
| Mean relative error | **3.5 %** | **5.5 %** | **4.5 %** |
| Nights > 30 min | 5 | 8 | — |

**Tiers.** All three stay "robust" on dev as well. TIB, TST and SE remain mandatory items stated plainly.

**W09.** Fired on 1 night (SC4562F0, whose TIB was within 30 min), so 1 false flag.
- Caught **0 of the 5** nights with TIB error > 30 min.
- Caught 0 of the 1 truncation.
- The dev tail is mostly **extensions**: SC4111E0 +163.5 min (wake scored as REM), SC4351F0 +62.5,
  ST7151J0 +67. One is a **staging** error: ST7131J0 +31.5, with a boundary contribution of 1.5.

**Decision, applying the pre-set rule:**
- **W09 is NOT adopted.** It doesn't generalise from train (3/5 truncations) to dev.
- **It is not re-tuned on dev.** That would make dev a tuning set.
- **No policy change, so no new release version.**

## T1 conclusion

1. **The concern was partly unfounded.** TIB, TST and SE keep their "robust" tier under Path B on
   train and on dev. Stating them plainly is justified by the project's own reliability rule.
2. **A real tail remains.** About 12–16 % of nights have TIB more than 30 min off. It mixes boundary
   extensions and staging errors. Pass-1 features alone don't detect it (W09 failed on dev).
3. **This is now a documented limitation, not a fixed gap.** A mean-based tier hides per-night tails;
   that's true of Path A too.
4. **Next levers,** each a separate, versioned piece of work:
   - **v2 staging (EEG + EOG).** It targets the wake-vs-REM confusion behind the largest extension
     (SC4111E0). Re-run `tib_on_train.py` on train with the v2 adapter once R3 exists.
   - **A per-night error model.** Predict |TIB error| from staging uncertainty plus boundary features,
     trained on train, and hedge TIB on nights it flags. This changes the packet contract (per-night
     tiers), so it would need verifier and report-model re-validation. Not started.
5. **Findings along the way:**
   - the builder rounding fault (K4, now R12);
   - P2_STATUS's list of large end errors was incomplete (seven nights, not six);
   - the study-period length figures first quoted were an upper bound, not TIB.

## Next steps (from the roadmap)

1. ~~Resolve SC4111E0~~ (above).
2. **Recompute on train.** Path B estimated periods on the 137 train nights; TIB, TST and SE errors
   against the reference periods.
3. **Recalibrate the TIB and SE tiers on train** with the existing rule.
4. **Design a boundary-uncertainty signal on train,** with its threshold fixed on train.
5. **Freeze it,** then evaluate once on dev: detection of TIB errors > 30 min, false-review rate,
   coverage, SE reliability.
6. **Act through the packet only.**
7. **Release it as its own version.**

Step 2 needs the staging model run on 137 train crops. It runs after the R2 arm C generation finishes,
so the two don't compete for CPU and RAM.
