# B1: Verification of the Completed Run

**1 November 2026.** The user ran the benchmark on Kaggle; results are at
`kaggle/working/b1/` (10 folds + `b1_summary.json` + `all_results.json` + the user's own
`README.md` write-up). This note records what I checked against the raw files, independent of
that write-up, since the task was to verify, not to re-describe.

## What was checked, and the result

| Check | Result |
|---|---|
| All 10 folds present, `status: done` | **Pass** — `all_results.json` lists 10/10 |
| Every SC subject in exactly one fold's validation set | **Pass** — 78/78 subjects, 0 duplicates |
| Every SC recording scored exactly once across all folds | **Pass** — 153/153, 0 duplicates |
| Checkpoints load and match the declared architecture | **Pass** — `student_best.pt`/`student_last.pt` for each fold, 175,599 trainable parameters, ~4.1 MB each |
| Trained from scratch (`alpha=1.0`, `warm_start=False`) | **Pass** — confirmed in each checkpoint's saved `config` |
| Headline numbers (83.2% / κ 0.761 / macro-F1 0.737) | **Pass** — reproduced exactly from `b1_summary.json`'s top-level fields |
| Per-fold table in the README | **Pass** — reproduced exactly from `b1_summary.json`'s `per_fold[*].val` |
| Per-class F1 ranges by stage (W/N1/N2/N3/REM) | **Pass** — reproduced from the per-fold `val.per_class_f1` values |
| "Epoch-pooled N3 F1 ≈ 0.78" claim | **Pass** — recomputed as 0.783 |
| Fold 0 training curve (κ ≈0.72 at epoch 4, peak 0.746 at epoch 29, ≈0.740 at epoch 54) | **Pass** — exact match against `fold_0/training_metrics.jsonl` |
| Best-epoch range 7–61 across folds | **Pass** — exact match (7, 11, 15, 19, 24, 27, 29, 30, 52, 61) |
| Early-stopping self-consistency | **Pass** — fold 0 stops at epoch 54 = best(29) + patience(25), matching the trainer's own `EARLY_STOP_PATIENCE=25` |

**No fabrication, no leakage, no corrupted output.** Every specific number in the user's README traces
correctly to the raw per-fold files.

## The one real finding: two aggregation conventions give different headline numbers

`run_b1_benchmark.py` computes two different kinds of average, and the README's headline uses one
while its per-fold table uses the other, without flagging the difference as a general point (it does
flag it for N3 specifically):

| Aggregation | Accuracy | κ | Macro-F1 |
|---|---:|---:|---:|
| **Per-recording mean** (`b1_summary.json` top level: average each of the 153 recordings' own score) | 83.2% | 0.761 | **0.737** |
| **Per-fold epoch-pooled, then averaged across folds** (pool every epoch within a fold into one confusion matrix, then mean the 10 folds) | 83.5% | 0.775 | **0.790** |

These differ by **5.3 points of macro-F1** and 1.4 points of κ. Neither file retains enough information
(no per-recording epoch counts or per-class support) to reconstruct the fully literature-standard number
— one confusion matrix pooling every epoch from all 153 recordings together — without re-running
inference on the raw data.

**Why this matters for citing against the published table.** Published Sleep-EDF-78 baselines
(DeepSleepNet, TinySleepNet, etc.) conventionally report accuracy/κ/macro-F1 pooled across every test
epoch, not averaged per recording. The **epoch-pooled number is the more comparable one**, and it is
also the more favourable one here — macro-F1 0.790 would be the **second-best figure in the whole
published table** (`GPU_PLAN_AND_BENCHMARK.md` §2.1), just below MultiChannelSleepNet's 79.6 (13M
params) and level with SalientSleepNet's 79.5 (0.9M params), at **175,599 parameters**. The headline
0.737 macro-F1, by contrast, would rank near the bottom of that table — an artifact of a few
recordings with little or no N3, which pulls a per-recording mean down hard but barely moves an
epoch-pooled one.

## Comparison against the published Sleep-EDF-78 table (`GPU_PLAN_AND_BENCHMARK.md` §2.1)

Using the epoch-pooled number as the fairer comparison:

| Model | Inputs | Parameters | Acc % | Macro-F1 | κ | N1 F1 |
|---|---|---:|---:|---:|---:|---:|
| SeqSleepNet | EEG | 0.16M | 82.6 | 76.3 | 0.76 | 46.0 |
| TinySleepNet | EEG | 1.3M | 83.1 | 78.1 | 0.77 | 51.0 |
| LightSleepX | EEG + EOG | 0.049M | 82.6 | 78.7 | 0.77 | 63.0† |
| LMCSleepNet | Fpz-Cz + Pz-Oz + EOG | 1.49M | 84.1 | 77.7 | 0.77 | 48.5 |
| **Ours (B1, EEG + EOG)** | **Fpz-Cz + EOG** | **0.176M** | **83.5** | **79.0** (epoch-pooled) / 73.7 (per-recording) | **0.775** (epoch-pooled) / 0.761 (per-recording) | **53.1** (epoch-pooled) / 52.5 (per-recording) |
| SalientSleepNet | Fpz-Cz + Pz-Oz + EOG | 0.9M | 84.1 | 79.5 | — | 54.2 |
| SleepSatelightFTC | EEG | 0.47M | 84.8 | — | 0.787 | — |
| MultiChannelSleepNet | Fpz-Cz + Pz-Oz + EOG | 13M | **85.0** | **79.6** | **0.79** | 53.0 |

†LightSleepX's N1 F1 is flagged in our own doc as unusually high next to every other entry; treat with care.

**Reading:** on either convention, the result is competitive with 1.3–13M-parameter models at **under
0.18M parameters** — among the smallest in the table, alongside LightSleepX and SeqSleepNet. On the
epoch-pooled convention it is essentially tied for second place on macro-F1 and κ, behind only
MultiChannelSleepNet (74× the parameters) and roughly level with SalientSleepNet (5× the parameters).

## Caveats (also correctly stated in the user's own README)

- **Not single-channel.** This uses the EEG + EOG "full" arm, the same model deployed as v2's staging
  adapter. Most of the table's strongest small models (SeqSleepNet, TinySleepNet) are EEG-only, so a
  strict single-channel claim would need a separate `base`-arm run of this same protocol.
- **One seed per fold, not replicated.** Consistent with the rest of this project's CV protocol
  (`IMPROVEMENT ROADMAP/ROADMAP.md` §3's seed-noise rule was applied to the architecture-change
  experiments, not required here since B1 is a one-off external benchmark, not a model-selection run).
- **Mild overfitting after the best epoch** is visible and openly reported (fold 0: κ 0.746 at epoch 29
  → 0.740 at epoch 54), which is why early stopping and best-epoch checkpointing matter here.

## Recommendation

Report **both numbers** when this goes into the project report, labelled by convention, rather than a
single unlabelled macro-F1. For a one-line claim, use the epoch-pooled figures (closer to how the
literature reports its own numbers): **accuracy 83.5%, κ 0.775, macro-F1 0.790, N1 F1 53.1%**, at
175,599 parameters.
