# Current release: v2

**Since 2 November 2026.** Release v2 is the system this project ships, demonstrates and reports. When
another document disagrees with this file, this file is right. Documents written for release v1 carry
a "Historical document" banner at the top; their content is unchanged.

## What v2 is

| Part | Release v2 | Release v1 (historical) |
|---|---|---|
| Staging model | `student_N6adv_full_s1`: EEG Fpz-Cz + horizontal EOG, 175,599 params | `student_N4kd`: EEG Fpz-Cz only, 139,606 params |
| Report model | Llama-3.2-3B-Student **v2**, retrained on v2 packets (`student/slm_models/v2/…-v2-q4_k_m.gguf`) | Llama-3.2-3B-Student (`student/slm_models/llama-3.2-3b-instruct-student-q4_k_m.gguf`) |
| Runtime | `llama-completion` b10927, frozen grammar, pinned chat template, temperature 0, seed 0 | Same (release v1.1); `deploy/server.py` used `llama-server` (PENDING K1) |
| Verifier, renderer, serializer | Unchanged from v1 (same pinned hashes) | — |
| Release file | `agentic/config/releases/v2.yaml`, execution manifest `4973f498b099fd9f…` | `v1.yaml`, `v1.1.yaml`, `deploy/frozen.json` |
| How to run it | `python -m agentic run-edf <file.edf> --release v2` (or `run-packet`) · the console: `python -m ui` | — |

## The numbers to cite for v2

All are measured. Every v2 staging number uses the **corrected** labels.

| What | Result | Source |
|---|---|---|
| Staging, held-out test (15 subjects, 33,365 epochs) | Accuracy 81.96%, **κ 0.756**, macro-F1 0.782 | `ADDING EOG VERSION - ADVANCED STUDENT/results/stage3_test_evaluation.json` |
| …against v1 on the same labels | v1 κ 0.718. v2 is better on 13 of 15 subjects; Δκ 95% CI +0.019 to +0.057 | same |
| …like-for-like ablation (5-fold CV, same labels and teacher) | +0.018 κ, CI +0.010 to +0.026 | `FOR JOURNAL/README.md` F3 |
| …with the eye channel missing | κ 0.744, still above v1 | same evaluation file |
| Standard Sleep-EDF-78 benchmark (B1, same architecture, from scratch) | κ 0.761 per recording; 0.769 fold-pooled at the final epoch; 0.775 at the best epoch, which is optimistic (`FOR JOURNAL` F1) | `ROADMAP B1 - BENCHMARK/VERIFICATION.md` |
| Real recordings end to end (Path B, 31 dev nights) | Mean κ 0.701 (v1 0.673); 30 verified + 1 sent to review (line noise) | `agentic/reports/p2_scorecards_v2_llama.json` |
| Reports, dev (31 nights) | 31/31 verified at 5/5, 0 violations, fidelity 527/527 | `ROADMAP R3 - V2 REPORT MODEL/results/dev_summary.json` |
| **Reports, held-out test (29 nights, final run P8)** | **28/29 verified, 0 flawed reports released**; 1 blocked by the verifier (an invented REM-latency value). 95% lower bound 82.2% | `ROADMAP P8 - V2 TEST EVALUATION/RESULTS.md` |

**Don't put these numbers next to v2's:** κ 0.7001, 0.6993 and 0.699 are v1 scored on the original
*stored* labels, which are shifted by one epoch on about 5% of epochs (PENDING K3, FOR JOURNAL F3).
v1's report result on test (29/29) is a correct result for v1, measured on v1's own packets. Cite it
as the earlier release, beside v2's 28/29, and never pooled with it.

## The test set has now been used for v2

P8 was the single pre-registered final evaluation. Any later model change, retrained or tuned, needs
a new evaluation set; this one can't be reused (`ROADMAP P8 - V2 TEST EVALUATION/TEST_ACCESS_LOG.md`).

## What stays v1, deliberately

These files are frozen, so they still name v1. **Changing them would change the hashes behind every
recorded parity, crash-recovery and reproducibility result**, so they are left as they are:

| File | What it says | How to read it |
|---|---|---|
| `deploy/server.py`, `deploy/frozen.json`, `pipeline.py` | The v1 deployment | Legacy v1 path. `pipeline.py` can't stage a new recording (PENDING K2); use the agent or the console |
| `agentic/cli.py` | `--release` defaults to `v1` | Always pass `--release v2`. The default stays because the agent's tests and replay cache are built on v1 |
| `agentic/config/policy.yaml` | `routing: v1 …`, `v2_enabled: false` | Declarative only; no code reads it. Unchanged to keep the policy hash of every recorded run |
| `distillation/results/phase2_*_packets/`, `distillation/results/packets/` | v1 (N4kd) evidence packets | v1 history. The v2 packets are in `ADDING EOG VERSION - PIPELINE INTEGRATION/work_n6adv_packets/` |
| `distillation/results/eval_students.json` | v1-era model table (stored labels) | History; the console labels these rows "stored" |

## Which documents are current

| Current | Historical (v1, with a banner) |
|---|---|
| This file · `PENDING.md` · `COMPARISON WITH EXISTING SYSTEMS/SYSTEM_ACCEPTANCE.md` · `FOR JOURNAL/README.md` · `ui/README.md` · `PLAIN LANGUAGE REPORT/` · `ROADMAP P8 - V2 TEST EVALUATION/` · the dated `ROADMAP *` and `ADDING EOG VERSION - *` folders (each true as of its date) | `README.md` · `README_V2.md` · `ABSTRACT.md` · `PROJECT_REPORT.md` · `PIPELINE_FLOW_README.md` · `Final_MODEL_METRICS_for STUDENT MODEL_REPORT.md` · `README_FOR_CODE_REVIEW.md` · `CODE_REVIEW_GUIDE.md` · `COMPARISON WITH EXISTING SYSTEMS/COMPARISON_WITH_EXISTING_SYSTEMS.md` |

**Figures:** use `figures/output_v2/` where a figure exists there (corrected for v2 on 2 Nov; see its README). The originals in `figures/output/` are v1-era. In particular, the original F19 contained four unsourced TinySleepNet values: don't use it.

**Drafts that still present v1 and need rewriting before use:** `S7_Report_Draft.md`,
`PRESENTATION_SCRIPT.md` and `FIGURE_GUIDE.md`. Each has a banner saying so. The numbers to use are in
the table above. A banner is a pointer, not a fix: these three still need their text changed.
