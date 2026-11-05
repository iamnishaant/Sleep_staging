# Plain-language report: data layer

**Release v2 (updated 2 Nov).** Everything here runs on the v2 packets (`ADDING EOG VERSION - PIPELINE INTEGRATION/work_n6adv_packets/`). They cover exactly the same epoch windows as the v1 packets for all 197 nights, so the clock offsets and expert labels apply unchanged. Figures measured earlier on v1 are noted where they differ.

**2 Nov 2026.** Groundwork for a report a non-expert can read. The verified clinical report is
written for clinicians; this folder adds the facts a lay summary needs: age-specific reference ranges
with citations, the person's age, wall-clock times and one missing sleep measure. It's deterministic
(no language model), and no existing file was changed.

| File | What it is |
|---|---|
| `reference_ranges.json` | Good and poor thresholds by age group, each citing NSF 2017, Watson 2015 or Ohayon 2004 |
| `build_tables.py` | Builds the two tables below from the raw Sleep-EDF files |
| `subjects.json` | Age, sex and cohort for all 197 SC and ST recordings, plus placebo or temazepam for ST nights |
| `clock_offsets.json` | Wall-clock start and epoch count of each of the 168 dev and train research packets. An offset is used only for a packet with the same epoch count, so a Path B packet of the same recording (cut at lights-off) gets no clock times |
| `lay_metrics.py` | Grades one packet against the ranges and checks the night window; the self-check runs with no arguments |
| `validate_awakenings.py` | Scores "awakenings longer than 5 min" against expert labels on dev and writes `awakenings_validation.json` |
| `plain_text.py` | The simple view: fixed sentence templates, plus a self-check that every number traces to its source |

The UI serves it at `/api/plain/{split}/{recording}` and shows it under **Simple view** on each recording page.

```bash
python "PLAIN LANGUAGE REPORT/build_tables.py" --data D:/sleep-edfx/sleep-edfx   # once
python "PLAIN LANGUAGE REPORT/lay_metrics.py"                                    # self-check
python "PLAIN LANGUAGE REPORT/lay_metrics.py" SC4081E0-PSG                        # one night
```

## What was verified

- **Clock offsets: 168/168.** Preprocessing starts the signal 30 min before the expert's first sleep
  epoch (`code/Phase1_48/Phase1reworked/preprocess.py`). Recomputing that start for every dev and
  train night reproduces each packet's exact epoch count.
- **Sleep period: 168/168.** The sleep period used for "awakenings longer than 5 minutes"
  reproduces each packet's own WASO and sleep-onset latency exactly.
- **Grading rules.** Boundary cases are asserted in `_selfcheck()`, including the ones that differ
  by age.

## Grades only say what the source says

| Grade | Meaning |
|---|---|
| good / poor | The value meets the source's rule for this age group |
| in_between | The source sets both a good and a poor level, and the value meets neither |
| outside_good_range | Not good, and the source sets no poor level for this group (e.g. WASO at 65 or over) |
| not_flagged | Not poor, and the source sets no good level for this group (e.g. deep sleep under 26) |
| no_rule | The source has no threshold for this group (e.g. sleep duration over 60) |

## Findings to carry into the report

1. **Two kinds of percentage.** The verified report gives stages as a share of the whole
   recording ("N3 35.9% of the night"). Clinical norms use share of total sleep time (49.6% for the
   same night). Comparing a norm against the first is wrong, so this folder uses only the second.
2. **Not every number can be shown to a layperson.** The packet's own validation tiers rate time to
   fall asleep and deep sleep as **unreliable** and WASO and the stage shares as **fragile**. Only
   total sleep and sleep efficiency are **robust**. A plain report should lead with the robust
   numbers and hedge or fold away the rest.
3. **"Awakenings longer than 5 minutes" is unreliable on v2, as it was on v1.** On the 31 dev nights,
   against the corrected expert labels, v2 has a 47% relative error. That is "unreliable" under the rule
   the project applies to every count built from stage runs (`distillation/metric_reliability.py`:
   fragile up to 40%). v2 undercounts by about 0.26 per night (v1: 0.52) and is within one awakening on
   84% of nights. The same script reproduces the packet's own WASO error (35.1% on v2), which checks the
   method. The measure appears only under "Details for your doctor", ungraded.
4. **Test split excluded.** Clock offsets need the expert hypnogram, and the test split is locked,
   so test packets have no clock time. A real device (agentic Path B) needs no hypnogram: its clock
   is EDF start + ingest crop + study-period start.
5. **The lights-off times are right; our night windows aren't always.**
   - The hypnogram and the recording share the same start time in all 197 files, so this isn't a
     clock error.
   - The cause is the window: preprocessing starts each packet 30 min before the first sleep anywhere
     in the ~20 h recording, and on some SC nights that first sleep is an evening nap.
   - Scale: in 29 of 168 dev and train nights on v2 (3 dev; 30 on v1), the model's first sleep is more than 30 min before
     lights-off.
   - Example: SC4172E0 has a nap at 18:28 and lights off at 00:37, and its v2 packet reports 288 min
     awake after falling asleep.
   - The fix in this layer: lights-off is never shown, it only gates grading (`window_check`), and
     such nights are explained and left ungraded.
   - See `PENDING.md` K5 and `FOR JOURNAL/README.md` F14.
6. **The source has little consensus on sleep stages.** NSF 2017 found little or no consensus that
   stage percentages indicate sleep quality. The four continuity measures (time to fall asleep,
   awakenings over 5 min, WASO, efficiency) are the strong ones.

## What the simple view shows

| Model reliability for the metric | Where it goes |
|---|---|
| robust (total sleep, efficiency) | Main section, compared with the norm |
| fragile (WASO, REM, N1, N2) | Main section, compared with the norm and with "could be off by about X" |
| unreliable (time to fall asleep, awakenings, deep sleep) | "Details for your doctor" only, never compared |

It always shows what the test cannot detect (breathing, oxygen, apnea, leg movements), when to talk
to a doctor, and a statement that it's not a diagnosis.

The simple view is **not** checked by the frozen verifier. It is safe by construction instead:
fixed templates, and a self-check over all 168 rendered nights confirms that every number in the
text came from the packet, the assessment or `reference_ranges.json`.

## Not done yet

- A reader study with non-experts (SYSTEM_ACCEPTANCE H1/H2). The wording is unvalidated until then.
- Test-split and newly uploaded nights stay ungraded unless their bedtime can be established without
  expert labels (on a device, from its clock and the lights-off crop).
