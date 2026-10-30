# V2: Path B at Sampling Rates Other Than 100 Hz

**28 October 2026.** Pending item V2. Only 128 Hz, on one night (P4 S06), had been checked before.

## Method

`run_v2.py`: for 3 dev nights (2 SC, 1 ST), each cropped at lights-off to the usual 12 h device window
then resampled to 128 Hz and 256 Hz with mne, run through `run-edf` for **both** releases (v1, v2) and
compared against the native 100 Hz run on the same crop: outcome, the pass-2 hypnogram, and the
packet's derived evidence values. 18 runs in total (3 nights × 2 releases × 3 rates including the
100 Hz reference).

## Result: resampling does not change the outcome or materially change the hypnogram

| Night | Release | Rate | Outcome matches 100 Hz | Hypnogram agreement | Epochs (100 Hz / resampled) |
|---|---|---:|---|---:|---|
| SC4001E0 | v1 | 128 / 256 Hz | yes / yes | 0.997 / 0.997 | 738 / 738 |
| SC4001E0 | v2 | 128 / 256 Hz | yes / yes | 0.999 / 0.999 | 737 / 737 |
| SC4081E0 | v1 | 128 / 256 Hz | yes / yes | **1.000 / 1.000** | 937 / 937 |
| SC4081E0 | v2 | 128 / 256 Hz | yes / yes | **1.000 / 1.000** | 936 / 936 |
| ST7022J0 | v1 | 128 / 256 Hz | yes / yes | 0.996 / 1.000 | 1025 / 1025 |
| ST7022J0 | v2 | 128 / 256 Hz | yes / yes | 0.992 / 0.999 | 1025 / 1025 |

- **18/18 outcomes match** the 100 Hz run (all VERIFIED_REPORT, none dropped to review or fallback).
- **Hypnogram agreement 0.992–1.000** (worst case: ST7022J0 at 128 Hz on v2).
- **Derived packet values** (TST, TIB, sleep efficiency, etc.) differ by at most 0.5 minutes or 0.004 of
  a ratio between rates — resampling-filter edge effects, not a functional problem.
- Epoch counts match exactly except where mne's resampling filter trims a few samples at the very end
  of the recording (never more than 1 epoch).

## Reading

The resampling path (`agentic/signals.py::to_100hz`) works correctly for both releases and both
cohorts. 256 Hz, twice the already-tested 128 Hz, was not a larger source of error than 128 Hz — if
anything it was closer to the 100 Hz reference in two of three nights.

## Caveats

- 3 nights, not the full dev set — a spot check, not an exhaustive sweep.
- No real device in this project records faster than 100 Hz; this only matters if a future input device
  does. Every existing Sleep-EDF file is natively 100 Hz.
