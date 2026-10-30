# ADDING EOG VERSION - PIPELINE INTEGRATION

**Question:** can the Stage 3 staging model (EEG + EOG, 175,599 parameters) replace N4kd
in the report pipeline, with the report model, grammar and verifier left frozen?

**Nothing outside this folder was changed.** The project's scripts are copied unchanged into
`harness/` and run there; their `results/` therefore land here, not in `distillation/results/`.

## What was done

| Step | Script / folder | Result |
|---|---|---|
| 1. Predictions as the chain expects them | `make_prob_caches.py` | Per-recording probability caches for the 31 validation, 29 test and 137 train nights, from the Stage 3 seed-1 checkpoint (sha256 `61e94bfc…`). Corrected labels; epochs the expert never scored are dropped from the *fitting* caches only, and kept in the *packet* caches so the night's timeline isn't shifted |
| 2. Check the harness | `work_n4kd_check/` | Running the copied chain on **N4kd** reproduces the shipped `reliability_table.json` (T = 1.2847), `sequence_decoding.json`, `night_confidence.json`, `derived_metric_reliability.json` and `n1_flag.json` exactly (the one difference is a test note the original added later) |
| 3. Refit every table on the new model (validation only) | `work_n6adv/` | See "What changed" |
| 4. Evidence packets | `work_n6adv_packets/packets_{dev,test,train}/` | 31 dev, 29 test, 137 train; schema 1.3, same fields as before |
| 5. Frozen report model on the new packets | `run_slm_on_packets.py`, `results/slm_dev/` | **Dev: 30/31 at 5/5, 0 violations, numeric fidelity 526/526** (frozen model on N4kd packets: 30/31) |
| 6. Test | `results/slm_test/` | see `INTEGRATION_REPORT.md` |
| 7. (Not needed) a retraining set | `build_trainset.py`, `results/trainset_n6adv/` | 123 + 14 oracle examples from the new train packets, built in case the report model needed retraining. **It didn't**, so it's unused |

## What changed in the tables (all fitted on validation)

| Item | N4kd | Stage 3 model |
|---|---|---|
| Calibration temperature | 1.2847 | **1.0021** (already calibrated) |
| Hypnogram decoder | REM min-run 3 | **plain argmax** (the validation search found no decoder that helps) |
| N1 flag threshold | 0.575 | 0.475 |
| Per-stage reliability | W, N2, N3, REM high; N1 low | **same tiers** (N1 low) |
| Derived-metric tiers | 12 fragile, 3 robust, 9 unreliable | 13 fragile, 3 robust, 8 unreliable (only `N1_Duration` moved: unreliable → fragile) |
| Packet attribution block | Gate 3a on N4kd | **null** (`status: not_run`). Claims cannot cite attribution, so the report is unaffected; the footer simply has nothing to show |

## Two deviations from the project's code (both in copies here)

1. **`work_n6adv_packets/distillation/build_packet.py` line 279** decides the hypnogram on the
   *rounded* (5-decimal) probabilities the packet ships. Without that, one epoch in one test night
   (`SC4201E0`, 0.48102 vs 0.48102) broke the builder's own self-consistency assertion.
2. Fitting caches drop never-scored epochs (the original dropped them from the labels, then
   mis-aligned the signal; see the label-alignment audit).

## How the report model must be run (important)

**Use the same path as the frozen evaluation:** `llama-completion`, one process per packet, the
GGUF's embedded chat template, `--jinja -cnv -st --grammar-file report/claims.gbnf -n 3000 -c 12288
--temp 0 --seed 0` (`candidates/run.py`, reproduced by `run_slm_on_packets.py --cli`).

Running the same model through **`llama-server`** (which is what `deploy/server.py` and `pipeline.py`
do) gives *different outputs on 8 of 31 dev nights* and a lower result, even on the old N4kd packets:

| Same frozen model, same dev packets | 5/5 clean | verifier FAIL |
|---|---:|---:|
| Frozen evaluation (cached outputs) | **30/31** | 0 |
| `llama-completion` (this folder, `--cli`) | **30/31 on the new packets** | 0 |
| `llama-server`, default template, N4kd packets | 20/31 | 5 |
| `llama-server`, pinned date template, N4kd packets | 23/31 | 3 |

The verifier caught every one of those failures (nothing unverified reaches a report), but the
deployed server path does not deliver the quality of the frozen evaluation. Worth fixing before
anything is deployed, and relevant to the Raspberry Pi plan (`AGENTIC AND SIMULATION`).
`results/slm_dev__server_*` hold those runs.

## Re-running

```
python make_prob_caches.py "<stage3 run>/student_N6adv_full_s1_fixlabels" --name student_N6adv_full_s1 [--include-test] [--include-train]
# fit (inside work_n6adv/distillation):  calibrate -> sequence_decode -> calibrate -> night_confidence -> metric_reliability -> fit_n1_flag
# packets (inside work_n6adv_packets/distillation): build_packet.py --model … --split val|test|train --out ../packets_…
python run_slm_on_packets.py work_n6adv_packets/packets_dev results/slm_dev --cli
```
