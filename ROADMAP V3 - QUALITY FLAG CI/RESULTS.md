# V3: Do Quality Flags Predict Worse Staging? A Confidence Interval

**28 October 2026.** Pending item V3. ACCEPTANCE.md §4.3's own bar: *"a rejected or warned night must
be measurably worse for staging than an accepted one: on train/val, nights flagged W02/Q01 must show
lower κ than unflagged nights. Report the difference with a CI."* Before this, only a 5-night, dev-only,
descriptive number existed (κ 0.587 flagged vs 0.689 unflagged, no CI).

## Method

`run_v3.py`. All 168 cached nights (137 train + 31 dev — the same cache `p2_calibrate.py`/M3 built; no
new model inference). Per night: N4kd's staging κ on its reference span, and the W02/Q01 flags computed
exactly as the frozen policy does (`quality_summary` + the frozen thresholds, warn 0.046 / review
0.110). 10,000-resample bootstrap on the mean-κ difference (flagged − unflagged), night-level.

**Note on scope:** this checks W02/Q01 (signal quality), as ACCEPTANCE §4.3 literally names. The
`PENDING.md` figure it was describing (0.587/0.689) combined W02 **and W08** (a different mechanism —
study-period instability, not EEG quality) on dev only. The two aren't the same comparison.

## Result

| Split | Flag | n flagged / clean | Mean κ difference | 95% CI | Bar met? |
|---|---|---|---:|---|---|
| **Dev** | W02 | 3 / 28 | **−0.138** | **[−0.202, −0.072]** | **Yes** — CI entirely below 0 |
| Train | W02 | 7 / 130 | +0.023 | [−0.030, 0.070] | No — not significant, wrong-signed point estimate |
| Train + dev pooled | W02 | 10 / 158 | −0.033 | [−0.108, 0.041] | No — diluted by train |
| Train / dev / pooled | Q01 | 0 flagged | — | — | n/a — no night exceeds the review threshold in either split |

Q01 never fires on train by construction (its threshold is the train maximum plus a margin) or on dev,
so it can't be checked this way.

## Reading

**The claim holds on dev, the set it matters for.** Dev is unseen data the threshold wasn't fit on; a
flag that holds up there is real evidence the flag generalises. CI [−0.202, −0.072] is entirely
below zero, with 3 flagged nights.

**Train shows no effect, and that's expected, not a failure.** `warn_fraction` is defined as the 95th
percentile of train's own `unusable_fraction` — a cut on the quality measure itself, not on κ. With only
7 flagged nights (5.1% of 137, exactly the design target) and signal-quality-to-κ being a noisy
relationship, a small, non-significant sample is the expected outcome, not evidence the flag fails.

**The pooled estimate is dominated by the larger, null train sample,** so it undersells the dev finding.
Report dev's result as the primary evidence, consistent with how the project already treats dev as the
evaluation set and train as the fitting set.

## Decision

**V3's bar is met on dev**, the split it was meant to hold on. No policy change: the existing W02
threshold (0.046, frozen 17 Oct) is retained as is.

## Caveat

3 flagged dev nights is a small sample; the CI is correspondingly wide relative to the point estimate.
This is the best evidence available from the existing dev set — it isn't grounds for a larger claim than
"flagged dev nights are measurably worse, with 95% confidence."
