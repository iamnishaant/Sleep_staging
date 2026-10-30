# R1: Same-Night Comparison with YASA

**28 October 2026.** Roadmap task R1 (`COMPARISON WITH EXISTING SYSTEMS/ROADMAP.md`). U-Sleep is an
online service (account + data upload) and was excluded by the project's offline constraint; YASA, fully
local once installed, is the alternative. Dev only — no test-set night is touched.

## Setup

- **Separate environment**, `F:\Sleep_project\venv_yasa` (independent of the project's torch
  environment, as the protocol requires): Python 3.14, YASA 0.8.1, mne 1.13.2, pandas 3.1.0,
  scikit-learn, python-calamine (for the `.xls` lights-off sheets).
- **Script:** `agentic/tools/r1_yasa.py`. Pretrained YASA, no tuning. Same device-like window
  (lights-off + 12 h) and the same estimated study span as the project's release-v2 staging run
  (`agentic/reports/p2_staging_scorecard_v2.json`), so both models are scored on identical epochs
  against the same expert labels.
- **Data:** `D:\sleep-edfx\sleep-edfx` (the full Sleep-EDFx download; superseded an earlier, incomplete
  copy — see below).

## Result: all 31 dev nights, 0 skipped

| | YASA | Project (v2) | Δ |
|---|---:|---:|---:|
| Accuracy | 0.744 | **0.790** | +0.046 |
| Cohen's κ | 0.635 | **0.701** | +0.066 |
| Macro-F1 | 0.642 | **0.717** | +0.076 |

| Per-class F1 | YASA | Project (v2) | Δ |
|---|---:|---:|---:|
| W | 0.749 | **0.821** | +0.072 |
| N1 | 0.302 | **0.545** | **+0.243** |
| N2 | 0.795 | **0.819** | +0.025 |
| N3 | 0.545 | **0.567** | +0.022 |
| REM | 0.816 | **0.834** | +0.018 |

The project's own model beats YASA on every metric and every stage, most on N1 — exactly the stage both
systems find hardest.

## Caveats

- This is one comparison, on this project's dev split, scored on this project's own estimated spans
  (same input for both models, so the comparison is fair, but YASA was not given a chance to pick its
  own window).
- YASA is a general-purpose, fast baseline, not the strongest published model; it is not being used as
  a ceiling, only as a same-night, fully offline reference point.
- U-Sleep was not run (excluded by the offline constraint, not evaluated).

## History: an earlier 23-night run is superseded

A first pass (27 Oct) used an older, partial Sleep-EDFx copy; 8 of 31 dev PSGs were absent, so those
nights were skipped (kept for the record at
`agentic/reports/r1_yasa_comparison_23nights_27oct.json`). The complete dataset was later found at
`D:\sleep-edfx\sleep-edfx` (153 SC + 44 ST PSGs, both subject sheets), and the hardcoded paths in
`agentic/tools/p2_parity.py`, `agentic/tools/r1_yasa.py` and `agentic/tests/test_edf_workflow.py` were
updated to it. This run is the first against the complete set.
