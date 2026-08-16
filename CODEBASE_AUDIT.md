# Codebase Audit Report — Sleep-Staging

**Audited:** 2026-08-03 · branch `main` @ `b00a428` ("phase 2 push, report, paper update") · working tree clean
**Scope:** every tracked file, with emphasis on the 142 Python modules and 9 notebooks.

---

## 1. Executive Summary

This repository is the **combined working tree of two separate final-year project groups (Group 47 and Group 48)** who share sleep/PSG data infrastructure but pursue different research goals. It is a **research scratchpad, not a package** — there is no importable library, no tests, no CI, and no single entry point.

| Metric | Value |
|---|---|
| Python files | 142 (~36,976 lines) |
| Jupyter notebooks | 9 (all in `code/Phase2_48/`) |
| Markdown docs | 21 |
| Preprocessed tensors (`.pt`) | 1,140 (730 CFS + 197+197 Sleep-EDFx) |
| Model checkpoints (`.pth`/`.pt`) | 17 |
| Raw XML annotations | 2,786 |
| CSV files | 2,112 (2,056 are per-patient MESA feature files) |
| Generated figures (PNG/JPG) | ~475 |
| Papers/reports (PDF) | 128 |

**The two research tracks:**

- **Group 47 → Neural Architecture Search for disease detection.** Searches CNN/attention architectures (DARTS + ENAS/RL) over raw CFS polysomnography signals to predict ~20 comorbidity flags (heart disease, hypertension, asthma, depression…).
- **Group 48 → Explainable sleep staging + sleep-disorder detection.** A hierarchical temporal/spectral transformer stages sleep on Sleep-EDFx/MESA, then derived features feed gradient-boosted classifiers for insomnia / RLS / apnea, wrapped in a SHAP+LLM explainability layer ("SomnAI").

**Headline findings** (detail in §7): 9 of 25 modules in `code/` have **unresolvable imports**; four files are **byte-identical duplicates** across phase directories; hardcoded absolute paths to `C:\`, `D:\`, `F:\` and Kaggle drives are pervasive; and `setup.py` describes an entirely different project (NASLib).

---

## 2. Repository Topology

```
Sleep-Staging/
├── code/                  ← ALL source. 286 files
│   ├── *.py (25)          ← flattened "active" NAS + MESA working copy  ⚠ broken imports
│   ├── Phase1_47/         ← Group 47, Phase 1: DARTS/ENAS on CFS
│   │   ├── Phase1reworked/    ← the ENAS binary-disease rewrite (current for G47)
│   │   └── model/             ← single inference script + one sample tensor
│   ├── Phase1_48/         ← Group 48, Phase 1: MESA transformer + comorbidity RNN
│   │   └── Phase1reworked/    ← the temporal/spectral fusion rewrite (current for G48)
│   ├── Phase2_47/         ← 1 file (sleep-architecture feature extraction)
│   ├── Phase2_48/         ← 9 notebooks + 4 scripts (current Phase 2 work)
│   ├── EDA/               ← dataset/disease inventory scripts
│   └── Test/              ← 49 files: scratch, spikes, dead prototypes
├── csv-docs/              ← NSRR cohort master CSVs (CFS, MESA, SHHS, MrOS, WSC, APPLES)
├── dictionary/            ← NSRR data dictionaries + one stray result dir
├── data/                  ← raw CFS XML annotations + 2,056 MESA per-patient features
├── cfs_preprocessed/      ← 730 preprocessed CFS tensors
├── processed_sleepedf/    ← 197 Sleep-EDFx tensors + 197 spectral + index.csv
├── results/               ← 353 files: checkpoints, metrics JSON, figures
├── plots/                 ← 207 generated figures
├── reports/               ← LaTeX paper, PPTX reviews, consent forms, lit reviews
├── unsorted/              ← 121 reference PDFs (papers)
└── <root>/*.py (8)        ← thin launcher wrappers into code/
```

---

## 3. Root-Level Files

| File | Lines | Contribution |
|---|---|---|
| [requirements.txt](requirements.txt) | 27 | ⚠ **Mismatched.** Pins NASLib/NAS-benchmark deps (`nasbench301`, `pybnn`, `grakel`, `emcee`, `pyro-ppl`) and `scipy==1.4.1`, `pyyaml==5.4.1`. Does **not** list `mne`, `matplotlib`, `seaborn`, `xgboost`, `shap`, `optuna`, `catboost` — all of which the code actually imports. |
| [setup.py](setup.py) | 88 | ⚠ **Belongs to a different project.** Declares `name='naslib'`, author "AutoML Freiburg", reads `naslib/__version__.py` (does not exist) → `python setup.py` fails immediately. Vestigial. |
| [.gitignore](.gitignore) | 5 | Ignores `cfs_preprocessed\`, `sleep-edf-database-expanded-1.0.0\`, `*.zip`, `*.pyc`. Note the Windows-style trailing `\` is not valid gitignore syntax — `cfs_preprocessed/` is in fact tracked (730 files present). |
| [cfs_data.py](cfs_data.py) | 24 | Builds `cfs_visit5_selected.csv`: reads the CFS master CSV, synthesizes a `path` column pointing at `D:\cfs\...edf`, keeps 20 diagnosis columns + path. **This is the origin of the Group 47 label matrix.** |
| [run_training.py](run_training.py) / [run_full_training.py](run_full_training.py) | 33/44 | Launcher wrappers calling `train_mesa_transformer.train()` with fixed hyperparameters (8 epochs, batch 512, lr 3e-4, seq_len 20, `max_samples=5000`), data at `C:\mesa`. Comments document a ~4 h wall-clock budget. Both are near-identical; `run_full_training` only adds banner printing. |
| [run_nas_memory_efficient.py](run_nas_memory_efficient.py) | 31 | `subprocess` wrapper invoking `code/main_nas.py --strategy both` with 4 GB-GPU-safe flags. |
| [visualize_attention.py](visualize_attention.py) | 62 | Loads `checkpoints_mesa/best_model.pth` + MESA dataloader, renders attention heatmaps for 3 samples. |
| [visualize_from_results.py](visualize_from_results.py) | 117 | ⚠ **Metrics hardcoded in source.** Test/val metric dicts and 6×6 confusion matrices were pasted from terminal output, then re-plotted. Records a MESA transformer run at **47.3% test accuracy / 0.351 macro-F1**, with class `N4` having zero support. |
| [create_visualizations.py](create_visualizations.py) | 61 | ONNX export + visualization driver; reads metrics from checkpoint if present. |
| [plot_nas_reward.py](plot_nas_reward.py) | 34 | Plots NAS reward curve from `efficient_darts_results.json`. |
| `EFFICIENT_DARTS_README.md` | — | Usage/config guide for the efficient DARTS search: search space list, constraint flags, output layout. |
| `MEMORY_FIX_SUMMARY.md` | — | Post-mortem of a CUDA OOM on a 4 GB GPU. Documents the fix: batch 4→1, cells 4→3, nodes 3→2, channels 32→24, and **removal of `disjoint_cnn`, `lstm`, and full `attention` from the search space**. |
| `NAS_FEATURES.md` | — | Documents colored logging, MNE warning suppression, and the checkpoint/resume protocol. |
| `README_ATTENTION_VISUALIZATION.md` | — | Explains the three attention types (temporal / channel / inter-epoch) and how to read each heatmap. |
| `desktop.ini`, `datasets/desktop.ini` | — | Windows folder-metadata cruft; `datasets/` is otherwise empty. |

---

## 4. `code/` (root) — The Active NAS + MESA Working Copy

25 modules. This directory is a **flattened snapshot** assembled from `Phase1_47/` and `Phase1_48/`, and the flattening broke it — see §7.1.

### 4.1 NAS core (Group 47 lineage)

| File | Lines | Contribution |
|---|---|---|
| [nas_search_space.py](code/nas_search_space.py) | 650 | **The search-space definition.** Nine operation classes: `Conv1dOperation`, `DisjointCNNOperation` (temporal (k,1) then spatial (1,C) conv), `LSTMOperation`, `AttentionOperation` (MHSA with auto head-count reduction), `FullyConnectedOperation`, `DepthwiseSeparableConv`, `TemporalPooling`, `IdentityOperation`, `ZeroOperation`. `SEARCH_SPACE` maps 18 op names → factory lambdas; `DARTS_OPS` is the 10-op subset actually searched (disjoint-CNN and LSTM commented out for the 4 GB GPU). `Cell` performs the softmax-weighted mixed-op forward with `F.interpolate` length reconciliation; `Network` is the supernet (stem → cells with reduction at ⅓ and ⅔ depth → global pool → linear) holding `_arch_params` of shape `(cells, edges, ops)`, plus `discretize()` for argmax architecture extraction. |
| [darts.py](code/darts.py) | 440 | `DARTSTrainer`: bilevel optimization — SGD on weights (train split), Adam on α (val split). Implements both **first-order** (default) and **second-order unrolled** steps with a finite-difference Hessian-vector product. Handles `single_label` (CE) and `multi_label` (BCEWithLogits) task types, gradient clipping, cosine LR schedules, per-10-batch CUDA cache clearing, and full checkpoint save/resume (`best`/`latest`/`final`/`error`). |
| [efficient_darts_search.py](code/efficient_darts_search.py) | 619 | The **primary Group-47 search entry point.** `EfficientDARTSTrainer` subclasses `DARTSTrainer` and adds a differentiable-ish efficiency penalty computed from softmaxed α. `run_efficient_darts_search()` orchestrates: dataloaders → supernet → search loop → periodic JSON log dumps → final architecture + FLOPs/MACs/params report. Includes a defensive inline fallback for `memory_utils` if the import fails. |
| [efficiency_tracker.py](code/efficiency_tracker.py) | 253 | Analytic cost model. Closed-form FLOP estimators per op family (conv/dilated/separable/attention/LSTM/linear/pool), `estimate_architecture_efficiency()` walking cells while tracking channel doubling and length halving, and `check_efficiency_constraints()`. MACs are approximated as FLOPs/2. ⚠ The parameter count is a rough formula, not a real count. |
| [nas_evaluator.py](code/nas_evaluator.py) | 338 | `PerformanceEstimator` (param/FLOP estimation + a zero-cost proxy score) and `ArchitectureEvaluator` (`evaluate()` full training, `quick_evaluate()` 3-epoch proxy, `validate()`), plus `print_architecture()`. ⚠ Broken import of `rl_search`. |
| [main_nas.py](code/main_nas.py) | 530 | Unified CLI for `--strategy darts \| rl \| both`. Builds loaders (Sleep-EDF `LazyPSGDataset` or CFS CSV), runs the chosen search, evaluates the discovered architecture, writes `nas_results_<strategy>_<timestamp>/` with architecture JSON, results JSON, model weights. ⚠ Broken imports of `rl_search` and `dataloader`. |
| [memory_utils.py](code/memory_utils.py) | 45 | `clear_gpu_cache`, `get_gpu_memory_info`, `print_gpu_memory`, `set_memory_fraction`. Created specifically for the OOM fix. |
| [utils.py](code/utils.py) | 126 | ANSI `Colors` class + `print_header/section/info/success/warning/error/key_value/metric/progress`, and `suppress_warnings()` which silences MNE at import time. Imported by nearly every NAS module. |
| [cfs_dataset.py](code/cfs_dataset.py) | 252 | `CFSAilmentDataset` — lazy multi-label loader. Fast path loads a preprocessed `.pt`; slow path reads EDF via MNE, resamples, picks named channels (pads/truncates to `input_channels`), z-score or min-max normalizes, and `F.interpolate`s to fixed `input_length`. Negative label values are clamped to 0. `create_cfs_dataloaders()` does seeded shuffle-split (val/test accept either absolute counts ≥1 or ratios). |
| [preprocess_cfs_signals.py](code/preprocess_cfs_signals.py) | 432 | **Offline CFS preprocessing.** Matches `cfs-visit5-{id}.edf` → `{id}-nsrr.xml`, parses the first `Stages\|Stages` ScoredEvent to **clip the leading wake episode**, resamples, selects channels, normalizes, resizes, saves `.pt`, and rewrites the CSV `path` column to point at the tensor. This produced `cfs_preprocessed/` (730 files). |
| [run_efficient_darts.py](code/run_efficient_darts.py) | 96 | Config-as-class launcher. Encodes the 4 GB-GPU profile: 3 cells / 2 nodes / 24 channels, batch 1, 20 epochs, 7 channels `C3-M2,LOC,ECG1,EMG1,THOR EFFORT,ABDO EFFORT,SaO2` @128 Hz, constraints 50M FLOPs / 25M MACs / 500K params. |
| [run_nas.py](code/run_nas.py) / [run_nas_small.py](code/run_nas_small.py) | 79/120 | Larger and smaller NAS launcher configs. `run_nas_small` has a broken `dataloader` import. |
| [run_evaluation.py](code/run_evaluation.py) | 45 | Builds argv and calls `evaluate_darts_architecture.main()`. |
| [evaluate_darts_architecture.py](code/evaluate_darts_architecture.py) | 843 | **The heavyweight evaluation harness.** Loads the best architecture from search JSON, rebuilds a discrete model, trains it, and emits multi-label metrics + six plot families: training history, ROC curves, PR curves, per-class performance, prediction heatmap, per-class confusion matrices, plus a text metrics report. ⚠ Broken `rl_search` import. |
| [eval_saved_model.py](code/eval_saved_model.py) | 476 | Same family, but evaluates a saved checkpoint without retraining; adds `plot_history_from_results()` to recover curves from the search JSON. |
| [plot_nas_reward_curve.py](code/plot_nas_reward_curve.py) | 150 | Renders NAS validation-accuracy ("reward") curves, optionally overlaying FLOPs/MACs/params. |
| [data_extract_cfs.py](code/data_extract_cfs.py) | 54 | Combines selected physiological channels from raw CFS EDFs into a consolidated output directory (`D:\cfs\...\edfs_combined`). |

### 4.2 MESA / comorbidity modules (Group 48 lineage, copied here)

| File | Lines | Contribution |
|---|---|---|
| [visualize_attention_heatmaps.py](code/visualize_attention_heatmaps.py) | 721 | **The explainability centerpiece for the MESA transformer.** Eleven plotting functions covering temporal attention (within-epoch), channel attention (cross-channel), inter-epoch attention (context), multi-epoch grids, and a combined dashboard; `average_attention_heads()` collapses heads; `visualize_sample_attention()` drives all of it per sample. ⚠ Broken `mesa_transformer` / `mesa_dataloader` imports. |
| [export_and_visualize.py](code/export_and_visualize.py) | 332 | ONNX export (`export_to_onnx`) + the standard metric figure set: confusion matrix (raw + normalized), per-class metric bars, class distribution, summary metrics. |
| [evaluate_and_export.py](code/evaluate_and_export.py) | 121 | Thin driver: load checkpoint → evaluate → export → visualize. |
| [evaluate_checkpoint.py](code/evaluate_checkpoint.py) | 243 | Evaluates a comorbidity-classifier checkpoint without retraining; supports threshold optimization and multi-split evaluation. Documented in `EVALUATION_GUIDE.md`. |
| [plot_comorbidity_training_curves.py](code/plot_comorbidity_training_curves.py) | 234 | Large-font training-curve and per-class-metric plots from saved history JSON. |
| [example_usage.py](code/example_usage.py) | 103 | Tutorial script: build dataloader → model → forward → one training step. |
| [mesa.py](code/mesa.py) | 213 | Builds `mesa_final.csv`: parses MESA NSRR XMLs into per-record sleep-stage strings and joins them onto `mesa_selected.csv`. |

### 4.3 `code/*.md` — operational playbooks

These four documents are the most valuable narrative record in the repo; they log what went wrong and what fixed it.

- **`BIAS_MITIGATION.md`** — the class-imbalance toolkit: stratified splits, weighted BCE (`balanced`/`inverse`), focal loss, per-class threshold optimization. States base rates: insomnia ~5–8%, RLS ~4–5%, apnea ~7–9%.
- **`AGGRESSIVE_TRAINING.md`** — escalation guide for when the model collapses to all-negative predictions: `WeightedRandomSampler`, `aggressive`/`very_aggressive` class weights, focal γ=3.0, early stopping on macro-F1. Sets realistic expectations (macro-F1 0.1–0.3, ROC-AUC 0.6–0.8).
- **`OVERFITTING_FIXES.md`** — records a concrete failure (train F1 **0.4881** vs val F1 **0.0417** at epoch 32) and the response: embed 32→16, hidden 128→64, combined 256→128, dropout 0.3→0.5, weight decay 1e-5→1e-3, BatchNorm→LayerNorm, plus adding SVM/RF/XGBoost baselines.
- **`EVALUATION_GUIDE.md`** — evaluation/resume recipes and the output-artifact inventory per split.

---

## 5. Phase Directories

### 5.1 `code/Phase1_47/` — Group 47, Phase 1 (NAS on CFS)

Contains its own copy of the NAS stack. Relative to `code/`: `darts.py`, `efficiency_tracker.py`, `preprocess_cfs_signals.py`, `memory_utils.py` are **byte-identical**; `nas_search_space.py`, `efficient_darts_search.py`, `utils.py` differ trivially (in `nas_search_space.py` the supernet class is named **`Supernet`** rather than `Network`); `nas_evaluator.py` and `evaluate_darts_architecture.py` have **genuinely diverged** — both define a local `architecture_to_model()`, which is the function `code/` tries (and fails) to import from `rl_search`. **This directory is self-consistent; `code/` is not.**

- [model/predict.py](code/Phase1_47/model/predict.py) — standalone inference against `F:\model\checkpoints\cfsnet_best.pth`, with one sample tensor beside it.

#### `code/Phase1_47/Phase1reworked/` — the ENAS rewrite (current Group 47 direction)

Pivots from multi-label DARTS to **single-disease binary ENAS**, based on `carpedm20/ENAS-pytorch`.

| File | Lines | Contribution |
|---|---|---|
| [enas_binary_classification.py](code/Phase1_47/Phase1reworked/enas_binary_classification.py) | 508 | The ENAS model zoo: `FocalLoss`, `WeightedBCEWithLogitsLoss`, primitive ops (`SepConv1d`, `DilConv1d`, `StdConv1d`, `PoolBN`), `DAGNode`, `SharedModel` (weight-sharing supernet), and `Controller` — the LSTM policy network that samples DAGs. |
| [enas_trainer_binary.py](code/Phase1_47/Phase1reworked/enas_trainer_binary.py) | 536 | `ENASTrainer`: alternating shared-weight training and REINFORCE controller training, with **PR-AUC as the reward** (`pr_auc_and_best_f1`) and `compute_class_weight()` for imbalance. |
| [enas_binary_train_script.py](code/Phase1_47/Phase1reworked/enas_binary_train_script.py) | 323 | CLI: `--disease_column`, `--save_dir`, `--num_epochs`. Drives the whole search. |
| [final_train_script.py](code/Phase1_47/Phase1reworked/final_train_script.py) | 329 | Retrains the discovered DAG from scratch: `normalize_dag()`, `FocalLossWithLogits`, `ModelWrapper`, `find_best_threshold()`. |
| [cfs_dataset.py](code/Phase1_47/Phase1reworked/cfs_dataset.py) | 405 | The `code/cfs_dataset.py` above **plus** `create_binary_dataloaders()` and a sklearn stratified split — the single-disease variant. |
| [Test_visual.py](code/Phase1_47/Phase1reworked/Test_visual.py), [training_result_visual.py](code/Phase1_47/Phase1reworked/training_result_visual.py), [final_eval_visual.py](code/Phase1_47/Phase1reworked/final_eval_visual.py), [arch_visual.py](code/Phase1_47/Phase1reworked/arch_visual.py) | 488/918/202/136 | ⚠ **Log-scraping plotters.** Raw training logs are pasted into the source as triple-quoted strings, regex-parsed, then plotted. `training_result_visual.py` is 918 lines of which the majority is commented-out prior versions. `arch_visual.py` hardcodes a discovered DAG and draws it. |
| [utils.py](code/Phase1_47/Phase1reworked/utils.py) | 126 | Copy of the color/printing utilities. |

### 5.2 `code/Phase1_48/` — Group 48, Phase 1 (MESA transformer + comorbidity RNN)

| File | Lines | Contribution |
|---|---|---|
| [mesa_transformer.py](code/Phase1_48/mesa_transformer.py) | 618 | **The MESA sleep-staging model.** Five components: `PositionalEncoding`, `ChannelCNN` (per-channel 1-D embedding), `TemporalTransformerEncoder` (within-epoch self-attention, per channel), `ChannelAttentionFusion` (cross-channel), `InterEpochTransformer` (context across the 20-epoch window), then a classification head. Input `(B, seq_len, channels, time_steps)` = `(B,20,3,3840)`; output dict with `logits`, `probs`, **`uncertainty`** (prediction entropy), and optional `temporal_attention` / `channel_attention` / `epoch_attention` maps — the hooks the visualizer consumes. |
| [mesa_dataloader.py](code/Phase1_48/mesa_dataloader.py) | 349 | `MESADataset`: loads `(4, T)` tensors, **keeps only the first 3 channels**, windows into `seq_len` consecutive epochs, parses the `sleep_stages` digit-string column into labels, optionally filters unscored epochs. |
| [train_mesa_transformer.py](code/Phase1_48/train_mesa_transformer.py) | 569 | Training loop with its own `FocalLoss`, `compute_class_metrics()` (per-class P/R/F1/support + macro/weighted + confusion matrix), checkpointing, and metric printing. |
| [preprocess_mesa_signals.py](code/Phase1_48/preprocess_mesa_signals.py) | 356 | MESA EDF→tensor: XML hypnogram match, leading-wake clipping, channel selection (EEG1/EEG2/EEG3/Thor), **EEG resampled to 128 Hz while Thor keeps native rate**, per-channel normalization. |
| [comorbidity_classifier.py](code/Phase1_48/comorbidity_classifier.py) | 222 | Multi-label head: sleep-stage `nn.Embedding` → packed bidirectional LSTM/GRU → last valid timestep, concatenated with a projection of 12 PSG features → LayerNorm MLP → 3 logits (insomnia / RLS / apnea). Careful init (Xavier on input-hidden, orthogonal on hidden-hidden, forget-gate bias = 1). The reduced dimensions from `OVERFITTING_FIXES.md` are baked in as defaults. |
| [comorbidity_classifier_dataset.py](code/Phase1_48/comorbidity_classifier_dataset.py) | 299 | `ComorbidityDataset` + `parse_sleep_stages()`; feature columns are `ahi_a0h3, ai_all5, odi35, timest1p5, timest2p5, times34p5, timeremp5, slp_eff5, waso5, plmaslp5, slpprdp5, remlaiip5`. |
| [train_comorbidity_classifier.py](code/Phase1_48/train_comorbidity_classifier.py) | 1046 | **Largest training script.** Implements everything the three imbalance playbooks describe: `compute_class_weights()` (4 methods × multiplier), `optimize_thresholds()`, `create_stratified_splits()`, weighted sampling, focal/weighted-BCE/BCE selection, early stopping on macro-F1 or macro-recall, and five plot generators. ⚠ **`train_epoch` is defined twice** (L37 and L508) — the second silently shadows the first. |
| [train_ml_classifiers.py](code/Phase1_48/train_ml_classifiers.py) | 741 | Classical baselines: SVM (with `GridSearchCV` tuning), Random Forest, XGBoost, wrapped in a hand-rolled `MultiOutputClassifier`. |
| [extract_sequence_features.py](code/Phase1_48/extract_sequence_features.py) | 163 | Turns stage sequences into ~50 tabular features (stage distribution, transition counts/rates, sleep efficiency, REM/deep/light %, longest-bout stats) → 62 features total with the 12 PSG ones. |
| [focal_loss.py](code/Phase1_48/focal_loss.py) | 156 | `FocalLoss` and `WeightedBCELoss`, standalone. |
| [compare_all_models.py](code/Phase1_48/compare_all_models.py) | 207 | Loads NN + ML result JSONs, produces comparison plots and a metrics CSV. |
| [inference_comorbidity.py](code/Phase1_48/inference_comorbidity.py) | 131 | Checkpoint → predictions on new data. |
| [psg_feature_extraction.py](code/Phase1_48/psg_feature_extraction.py) | 1093 | **The clinical feature engine** (see §5.4). |

#### `code/Phase1_48/Phase1reworked/` — the temporal/spectral fusion rewrite (current Group 48 direction)

This is the code that became the Phase 2 notebooks and the paper.

| File | Lines | Contribution |
|---|---|---|
| [model.py](code/Phase1_48/Phase1reworked/model.py) | 178 | **The current sleep-staging architecture.** `SEBlock` (squeeze-excite) → `AdaptiveAtrousPyramid` (4 dilated conv branches at d=1,2,4,8 with a *learned softmax gate* over branches) → `EpochEncoder` → `ChannelAttentionFusion` (learns 2 weights to blend the 128-d temporal embedding against a projected 34-d spectral vector) → `PositionalEncoding` → `SleepTransformer` (4-layer, 4-head, pre-norm) → 5-class output. `SleepStagingModel.forward` accepts either a `(temporal, spectral)` tuple or temporal-only, which is exactly what makes the three ablations possible. |
| [model_spectral.py](code/Phase1_48/Phase1reworked/model_spectral.py) | ~120 | Spectral-only variant (no CNN branch). |
| [dataset.py](code/Phase1_48/Phase1reworked/dataset.py) | ~80 | `SleepEDFSequenceDataset` with `STAGE_TO_IDX = {W:0, N1:1, N2:2, N3:3, REM:4}`, `NUM_CLASSES=5`. |
| [preprocess.py](code/Phase1_48/Phase1reworked/preprocess.py) | 172 | Sleep-EDFx → 30 s epochs: hypnogram matching, epoch labeling, leading/trailing wake trimming. Produces `processed_sleepedf/`. |
| [spectral_preprocess.py](code/Phase1_48/Phase1reworked/spectral_preprocess.py) | 202 | The **34-dim spectral feature vector**: DWT features, STFT band powers, band ratios, `spectral_entropy`, `spectral_edge_frequency`. |
| [preprocess_mesa.py](code/Phase1_48/Phase1reworked/preprocess_mesa.py) | 366 | MESA preprocessing aligned to the Sleep-EDF output format (tensors + `index.csv` with `stage_sequence`). |
| [train.py](code/Phase1_48/Phase1reworked/train.py) | 234 | Training loop with `sleep_collate_fn` for variable-length night sequences. |
| [losses.py](code/Phase1_48/Phase1reworked/losses.py) | ~40 | Epoch-wise `FocalLoss`. |
| [utils.py](code/Phase1_48/Phase1reworked/utils.py) | ~60 | `set_seed`, `compute_specificity`, `compute_metrics` (accuracy, F1, precision, recall, ROC-AUC, AP, **Cohen's κ**, confusion matrix). |
| [feature_study.py](code/Phase1_48/Phase1reworked/feature_study.py) [v2](code/Phase1_48/Phase1reworked/feature_studyv2.py) [v3](code/Phase1_48/Phase1reworked/feature_studyv3.py) | 340/203/223 | **Three generations of the feature-selection pipeline.** v1: missing-value handling → mutual information → correlation pruning → XGBoost importance → L1-logistic → stability selection → backward ablation. v2: adds ElasticNet. v3: consolidates to imputation+scaling → ElasticNet stability selection → train/evaluate. |
| [classifier_train.py](code/Phase1_48/Phase1reworked/classifier_train.py) | 337 | Trains ML + MLP models for the three disorders; `best_threshold_gmean()` picks thresholds by geometric mean of sensitivity/specificity. |
| [physiological_feature_qc.py](code/Phase1_48/Phase1reworked/physiological_feature_qc.py) | ~120 | Quality-control sweep over MESA questionnaire/PSG columns. |
| [combined_features_classification.py](code/Phase1_48/Phase1reworked/combined_features_classification.py) | ~150 | Questionnaire + signal feature classification. |
| [plot_json.py](code/Phase1_48/Phase1reworked/plot_json.py) | 634 | Plots the fusion run's `training_metrics_fusion.jsonl`. ⚠ Hardcoded to `C:\PS\Sleep-Staging\results\...`. |
| [feature_extraction.py](code/Phase1_48/Phase1reworked/feature_extraction.py), [free_worker.py](code/Phase1_48/Phase1reworked/free_worker.py), [process_verify.py](code/Phase1_48/Phase1reworked/process_verify.py) | small | CSV merge helpers and a tensor sanity-check. `feature_extraction.py` is **entirely commented out**. |

### 5.3 `code/Phase2_47/` and `code/Phase2_48/`

**Phase2_47** contains exactly one file: [physiological_features.py](code/Phase2_47/physiological_features.py) (127 lines) — `extract_features(annotations, lights_off)` derives sleep-architecture metrics from a predicted stage sequence. This is the bridge from staging output to disorder prediction.

**Phase2_48** is where the current work lives — 9 Kaggle notebooks plus 4 scripts:

| Notebook | Cells | Contribution |
|---|---|---|
| `temporal-only-run.ipynb` | 15 | **Ablation A** — CNN temporal encoder → transformer, no spectral branch. |
| `spectral-only-run.ipynb` | 13 | **Ablation B** — MLP spectral encoder → transformer, no raw EEG. |
| `temporal-spectral-fusion.ipynb` | 24 | **The full model** — both branches with `fusion_collate_fn`. Results in `results/sleep staging/temporal spectral fusion/` cover four fusion strategies: concat, gated, cross-attention (each with a "fixed" rerun). |
| `sleep-disorder-detection.ipynb` | 33 | **The main disorder pipeline.** GPU XGBoost/LightGBM/CatBoost + soft-voting ensemble, Optuna (100 trials × target × model, 3-fold CV on PR-AUC), 5-fold OOF evaluation, feature importance, pickled models. Targets: doctor-diagnosed insomnia / RLS / apnea. Drops AHI-artifact rows. |
| `feature-importance.ipynb` | 34 | SVM pipeline with calibrated classifiers; opens with a root-cause diagnosis of the previous run's failure. |
| `experiment-4.ipynb` | 21 | Feature Study v3 full evaluation suite. |
| `experiment-5.ipynb` | 19 | PSG-derived (signal-engineered) feature evaluation — explicitly contrasted against questionnaire features. |
| `experiment-6.ipynb` | 25 | Merged PSG + questionnaire features; header documents what Experiment 5 got wrong. |
| `explainability.ipynb` | 24 | **"SomnAI" explainability suite** — SHAP explainers over the fitted XGBoost models plus `groq` LLM calls to generate clinician-facing and patient-facing narratives. |

Scripts: [preprocess.py](code/Phase2_48/preprocess.py) and [spectral_preprocess.py](code/Phase2_48/spectral_preprocess.py) are **byte-identical copies** of the `Phase1_48/Phase1reworked/` versions; [psg_feature_extraction.py](code/Phase2_48/psg_feature_extraction.py) differs from the Phase1_48 copy **only by a missing shebang line**; [check_cites.py](code/Phase2_48/check_cites.py) cross-checks `\cite{}` keys in `main.tex` against `references.bib`. A stray `project_summary.txt`` ` (trailing backtick in the filename) also sits here.

### 5.4 The clinical feature engine

`psg_feature_extraction.py` (1,093 lines, present in three near-identical copies) is the most substantial single-purpose module in the repo. It reads MESA EDF+XML and computes one CSV row per patient across six clinical domains:

- **Sleep architecture** — stage percentages, latencies, WASO, efficiency, fragmentation, segment analysis (`_segment_stages`, `epoch_mask`)
- **Respiratory** — flow-event detection (`_detect_flow_events` via bandpass + moving RMS/median), stage-stratified event filtering, AHI/RDI
- **Oxygenation** — SpO2 statistics and `_compute_odi` (3% desaturation index)
- **EEG** — spectral band features
- **Cardiac** — `_detect_rpeaks` + `_hrv_frequency_domain` HRV analysis
- **Movement/other** — PLM features, REM EMG atonia, bruxism

`process_patient()` / `run_batch()` drive the pipeline; output is the 2,056-file `data/mesa/features/` directory. `data/psg_feature_extraction_fixed.py` (1,169 lines) is a substantially revised fourth copy sitting outside `code/` entirely.

### 5.5 `code/EDA/`

- [Disease_list.py](code/EDA/Disease_list.py) — inventories disease variables across all five NSRR data dictionaries (CFS, APPLES, MESA, SHHS, WSC).
- [Disease_distribution.py](code/EDA/Disease_distribution.py) — builds `dataset_variable_distributions.xlsx` from `diseases_summary.xlsx`.

### 5.6 `code/Test/` — 49 files of scratch work

Unversioned experimentation. Categorized:

- **Load-bearing despite the location:** [rl_search.py](code/Test/rl_search.py) (486 lines — `PolicyNetwork`, `RLSearchTrainer` REINFORCE controller, `architecture_to_model`, `actions_to_architecture`) and [dataloader.py](code/Test/dataloader.py) (141 lines — `LazyPSGDataset`). **Four modules in `code/` import these and fail.**
- **Substantial prototypes:** [sleep_staging_transformer.py](code/Test/sleep_staging_transformer.py) (1,322 lines — the original hierarchical local/global transformer with `CFSSleepStagingDataset`, `CheckpointManager`, weighted sampling, confidence penalty); [psg_preprocessing.py](code/Test/psg_preprocessing.py) (983 lines — general EDF resampling/artifact-removal/filtering); [sleep_metrics.py](code/Test/sleep_metrics.py) (772 lines — AHI, ODI, SpO2, RDI, latency, WASO, efficiency, arousal index); [checkpoint_eval.py](code/Test/checkpoint_eval.py) (466 lines).
- **Dead duplicates:** `transformer-nabbed.py` and `sleepStagingTransformer_OLD.py` are the same file modulo 4 lines. `transformer.py`, `transformer-47.py`, `train.py` are superseded prototypes.
- **Cohort extractors:** `data_extract_{apples,mesa,mros,shhs,wsc}.py` — mostly stubs still pointing at `path/to/edf_directory`.
- **One-liners and probes:** `gpu_check.py` (6 lines), `metrics_extract.py` (13), `csv_merge.py` (9), `edf_read.py` (18), `eda.py` (23 — stage-duration totals hardcoded), `names.py`, `pt_check.py`, `tensor_check.py`, `wakecheck.py`, `view_signal.py`, `debug.py`, `test.py`, `hehe.py`.

---

## 6. Non-Code Assets

### 6.1 Input data

| Location | Contents |
|---|---|
| `csv-docs/` (21) | NSRR cohort master CSVs + harmonized variants: CFS visit 5, MESA sleep 0.8.0, SHHS1/SHHS2 + CVD events, MrOS visit 1/2, WSC + MSLT, APPLES. Plus the derived `cfs_visit5_selected.csv` (Group 47 labels), `mesa_selected.csv`, and `metrics to use.xlsx`. A `.bak` file is also tracked. |
| `dictionary/` (10) | Five NSRR data dictionaries. ⚠ Also contains a misfiled `BloodPressure/` results directory (ENAS checkpoint + JSON + two run logs) that belongs in `results/`. |
| `data/cfs_nsrr/polysomnography/annotations-events-nsrr/` | 730 CFS XML hypnograms — pairs 1:1 with `cfs_preprocessed/`. |
| `data/mesa/features/` | 2,056 per-patient `*_features.csv` — the output of the clinical feature engine. |
| `data/mesa/features (test with 5 edfs)/` | 6-file smoke-test output. |
| `cfs_preprocessed/` (730) | `{nsrrid}_preprocessed.pt` — Group 47's model input. |
| `processed_sleepedf/` | 197 `tensors/` + 197 `spectral/` + `index.csv`. ⚠ `index.csv` paths are **Linux paths from another machine** (`/home/geethalekshmy/GirishS/tensors/...`) — unusable locally without rewriting. |
| `datasets/` | Empty except `desktop.ini`. |

### 6.2 Results

| Location | Contents |
|---|---|
| `results/efficient_darts_results/` | The **primary Group 47 NAS result**: `efficient_darts_results.json` (best val acc **90.60%** over 20 epochs; best architecture = 3 cells, cell 0–1 dominated by `dil_conv_5x1`/pooling, **cell 2 almost entirely `attention_light`**; 158.5M FLOPs / 78.3M MACs / **170,904 params** — comfortably under the 500K budget), plus `darts_search.log` and 4 periodic log dumps. |
| `results/final_eval_results/` | Multi-label evaluation of that architecture: macro-F1 **0.097**, macro-AUC **0.777**, macro-AP 0.364, micro-F1 0.374 (P 0.717 / R 0.253), exact-match 0.316, Hamming 0.097. **Reads as high AUC but near-total precision/recall collapse under a fixed 0.5 threshold** — the classic imbalance signature the playbooks describe. Plus ROC/PR/confusion/heatmap/history figures. |
| `results/Disease detection/` | 10 per-disease ENAS runs: BloodPressure v1–v4, Depression v1–v2, Hayfever, asthma, diabetes, sinus. Each holds `enas_{code}_checkpoint.pth` and, for completed runs, `{code}_post_enas_best_model.pth`. |
| `results/sleep staging/` | Group 48 staging runs: `temporalv1/v2`, `spectralv1/v2`, and `temporal spectral fusion/` with six variants (concat, concat fixed, gated, gated fixed, cross-attn, cross-attn fixed). Each has checkpoints, confusion matrices, ROC curves, training curves, `.jsonl` metric logs. |
| `results/Automated Feature selection run/` vs `Manual Feature selection run/` | **A direct comparison experiment** — same three targets (Apnea, Insomnia, PLMD), automated vs manual feature selection, each with summary PNGs and a summary CSV. |
| `results/model_exports/` | `mesa_transformer.onnx` + 10 metric figures for test/validation. |
| `results/checkpoints_mesa/best_model.pth`, `results/evaluation_results/`, `results/main directory files/` | The MESA transformer checkpoint; a confusion/class-performance pair; and 20 loose working files (`mesa_final.csv`, `mesa_comprehensive.csv`, per-disorder feature CSVs at three selection stages, `best_model.pt`, NAS figures). |

### 6.3 Figures — `plots/` (207 files)

`attention_visualizations/` (51) is the largest — the MESA transformer explainability output. `pie_charts/` (117) holds per-cohort distribution charts. `Uncategorized/` (15) contains the figures that appear in the paper: `FLOPS_efficiency graph.png`, `FLOPSvsACC_scatter.png`, `NAS_Acc_Curve.png`, `NAS_Loss_curve.png`, `nas_reward_curve*.png`, `DFD_P2.png`, class/age distributions. `explainability/` (3) holds calibration and signal-importance plots. `methodology/` (3) holds one methodology diagram per group. `Sleep-EDFx/` (17) contains its own three plotting scripts (`plot_signal.py`, `sequence.py`, `distribution_visual.py`) alongside their output — and a stray `__pycache__`.

### 6.4 `reports/` and `unsorted/`

`reports/` holds the deliverables: `main.tex` (the IEEE-conference paper — Introduction/Scope/Motivation/Objectives → Search Strategy → Literature Review → **Methodology** (dataset, EEG channel selection, epoching, sequence construction, hierarchical architecture, intra-epoch attention, channel-wise fusion, optimization stability) → Performance Evaluation → Discussion → Conclusions → Future Directions), `bibil.bib`, both groups' Phase-1 PDFs and PPTX reviews, consent forms, meeting minutes, literature-survey spreadsheets (SSL and XAI), and an invention disclosure form.

`unsorted/` is a 121-file reference library: `NAS/` (11 papers incl. a 500K-hour sleep foundation model), `Sleep disorder detection/` (18 + subfolders for explainable/SOTA papers), `papers/` (13), plus loose PDFs.

---

## 7. Findings

### 7.1 🔴 `code/` cannot be imported — 9 of 25 modules have unresolvable imports

The flattening of `Phase1_47/` and `Phase1_48/` into `code/` left the dependencies behind. Every one of these fails at import time:

| Module | Missing import | Where the target actually lives |
|---|---|---|
| [main_nas.py](code/main_nas.py) | `rl_search`, `dataloader` | `code/Test/` |
| [nas_evaluator.py](code/nas_evaluator.py) | `rl_search` | `code/Test/` |
| [evaluate_darts_architecture.py](code/evaluate_darts_architecture.py) | `rl_search` | `code/Test/` |
| [run_nas_small.py](code/run_nas_small.py) | `dataloader` | `code/Test/` |
| [visualize_attention_heatmaps.py](code/visualize_attention_heatmaps.py) | `mesa_transformer`, `mesa_dataloader` | `code/Phase1_48/` |
| [export_and_visualize.py](code/export_and_visualize.py) | `mesa_transformer` | `code/Phase1_48/` |
| [evaluate_and_export.py](code/evaluate_and_export.py) | `mesa_transformer`, `mesa_dataloader`, `train_mesa_transformer` | `code/Phase1_48/` |
| [evaluate_checkpoint.py](code/evaluate_checkpoint.py) | `comorbidity_classifier`, `comorbidity_classifier_dataset`, `train_comorbidity_classifier` | `code/Phase1_48/` |
| [example_usage.py](code/example_usage.py) | `comorbidity_classifier`, `comorbidity_classifier_dataset` | `code/Phase1_48/` |

Because `nas_evaluator` is broken, `efficient_darts_search.py` — the documented main entry point in `EFFICIENT_DARTS_README.md` — fails transitively. **The stale `code/__pycache__/` still holds `rl_search`/`dataloader` `.pyc` files from when this directory did work**, confirming this is regression rather than an unfinished feature. Cheapest fix: copy `rl_search.py` and `dataloader.py` from `Test/` into `code/`, and the four MESA/comorbidity modules from `Phase1_48/`.

### 7.2 🟠 Four-way duplication of the same files

| Files | Status |
|---|---|
| `darts.py`, `efficiency_tracker.py`, `preprocess_cfs_signals.py`, `memory_utils.py` | **Byte-identical** between `code/` and `code/Phase1_47/` |
| `preprocess.py`, `spectral_preprocess.py` | **Byte-identical** between `Phase1_48/Phase1reworked/` and `Phase2_48/` |
| `psg_feature_extraction.py` | Phase1_48 vs Phase2_48 differ by **one line** (shebang); `data/psg_feature_extraction_fixed.py` is a diverged 4th copy (645 diff-lines) |
| `nas_search_space.py`, `efficient_darts_search.py`, `utils.py` | Trivially different (`Network` vs `Supernet` class name) |
| `nas_evaluator.py`, `evaluate_darts_architecture.py` | **Genuinely diverged** — the Phase1_47 copies define `architecture_to_model()` locally, which is precisely the symbol `code/` fails to import |
| `transformer-nabbed.py` / `sleepStagingTransformer_OLD.py` | 4 diff-lines apart, both dead |
| `utils.py` | Three copies (`code/`, `Phase1_47/`, `Phase1_47/Phase1reworked/`) |

A bug fixed in one copy does not reach the others. Since the phase directories are the archival record and `code/` is the working copy, the safest consolidation is to make `code/` complete and correct and treat the phase directories as read-only snapshots.

### 7.3 🟠 Hardcoded absolute paths throughout

Six distinct machine layouts appear in tracked source: `C:\Sleep-Staging\`, `C:\PS\Sleep-Staging\`, `D:\cfs\`, `D:\Sleep-Staging\`, `F:\Sleep-Staging\`, `F:\model\`, `C:\mesa`, `C:/Users/pujas/Downloads/`, `C:\Users\Hari\Documents\`, `/kaggle/input/...`, and `/home/geethalekshmy/GirishS/`. Nothing runs on a fresh checkout without editing. `processed_sleepedf/index.csv` is affected too, which means the *data* is machine-bound, not just the code.

### 7.4 🟠 `setup.py` and `requirements.txt` describe a different project

`setup.py` packages **NASLib** (AutoML Freiburg) and reads a nonexistent `naslib/__version__.py`. `requirements.txt` pins NAS-benchmark libraries while omitting `mne`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `shap`, `groq`, `pywt`, and `onnx` — all imported by actual code. `scipy==1.4.1` and `pyyaml==5.4.1` are incompatible with the `numpy>=1.22` also pinned. Effectively, there is no working install path.

### 7.5 🟡 Results and metrics embedded in source

- `visualize_from_results.py` — full test/val metric dicts and confusion matrices pasted from terminal output.
- `Phase1_47/Phase1reworked/Test_visual.py`, `final_eval_visual.py`, `training_result_visual.py` — training logs pasted as string literals and regex-parsed.
- `Phase1_47/Phase1reworked/arch_visual.py` — a discovered DAG hardcoded.
- `Test/eda.py` — stage-duration totals hardcoded.

These figures cannot be regenerated if a run is repeated, and the numbers can silently drift from the checkpoints they claim to describe.

### 7.6 🟡 Code-level defects worth fixing

- **`train_comorbidity_classifier.py` defines `train_epoch` twice** (L37 and L508); the later definition wins silently.
- **`DisjointCNNOperation` lazily constructs `self.spatial_conv` inside `forward()`** ([nas_search_space.py:146](code/nas_search_space.py#L146)). The layer is created after the optimizer is built, so its parameters are never registered with the optimizer and are lost on `state_dict()` save/load. The op is currently disabled in `DARTS_OPS`, which masks the bug.
- **`darts.py` uses deprecated in-place tensor call signatures** — `g.data.sub_(eta, ig.data)` and `p.data.add_(R, v)` ([darts.py:136](code/darts.py#L136), [darts.py:150](code/darts.py#L150)) — removed in modern PyTorch; the unrolled/second-order path will throw. Only the first-order default path works.
- **`efficiency_tracker.estimate_operation_flops` orders branches so `'attention_light'` is unreachable** — the `'attention' in op_name` test at [efficiency_tracker.py:120](code/efficiency_tracker.py#L120) matches first (`'attention_light'` contains the substring `'attention'`), so the `elif` at L123 is dead code, so lightweight attention is always costed at 4 heads instead of 2.
- **`Network.discretize()` can select `zero`/`identity` on every edge** with no top-k selection, so a "discovered" architecture may contain no learnable path.
- **Bare `except:` blocks** in `utils.py` swallow all MNE import errors.

### 7.7 🟡 Organizational issues

- `code/Test/` holds two modules (`rl_search.py`, `dataloader.py`) that the production path depends on — 49 files of scratch mixed with load-bearing code, none of it tested.
- `dictionary/BloodPressure/` is a results directory filed under data dictionaries.
- `data/psg_feature_extraction_fixed.py` is source code living in the data directory.
- `code/__pycache__/` and `plots/Sleep-EDFx/.../__pycache__/` are tracked despite `*.pyc` in `.gitignore` (they were committed before the rule existed).
- Tracked cruft: `csv-docs/cfs_visit5_selected.csv.bak`, three `desktop.ini`, `code/Phase2_48/project_summary.txt`` ` (trailing backtick in filename).
- **No tests anywhere** (`pytest` is in `requirements.txt`, `test_suite='pytest'` in `setup.py`, zero test files exist — `Test/test.py` and `Test/test_dataloader.py` are manual scripts).
- **No README.md at the root** — `setup.py` tries to `open("README.md")` and would fail. The four topic READMEs cover subsystems but nothing orients a newcomer to the repo as a whole.

---

## 8. Recommended Priorities

1. **Restore `code/` importability** — copy `rl_search.py` + `dataloader.py` from `Test/` and the four MESA/comorbidity modules from `Phase1_48/`. Without this the documented NAS entry point does not run. *(highest value, lowest effort)*
2. **Write a root `README.md`** documenting the 47/48 split, which directory is current for each group, and the actual entry points.
3. **Replace `setup.py` and `requirements.txt`** with ones describing this project. Generate requirements from actual imports.
4. **Centralize paths** into a config module or `.env` so a checkout runs unmodified; rewrite `processed_sleepedf/index.csv` to relative paths.
5. **De-duplicate**, keeping `code/` canonical and phase directories as frozen snapshots — or invert it, but pick one.
6. **Fix the defects in §7.6**, starting with the duplicate `train_epoch` and the `attention_light` FLOP mis-costing, since both silently produce wrong results rather than errors.
7. **Persist metrics to JSON at run time** instead of pasting them into plotting scripts.
8. **Move `Test/` scratch to a clearly-marked `scratch/`** once its two load-bearing modules are extracted.

---

*Report generated from a full read of the repository. All line numbers and metric values were verified against the files as of `b00a428`.*
