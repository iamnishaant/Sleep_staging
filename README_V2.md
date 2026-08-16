# README_V2 — Group 48 Lineage & Knowledge-Distillation Project

**Author:** continuation of Group 48 (explainable sleep staging + sleep-disorder detection)
**Created:** Phase 0 · repo @ `b00a428`
**Purpose:** orient this project on Group 48's work only, and document the exact state of the models, data, and entry points that a distillation pipeline will build on.

> This file documents *my* project. It does not modify or supersede any existing README. Group 47's NAS work is out of scope throughout.

---

## 0. M0 — the model everything downstream uses

> **`distillation/results/students/student_baseline_E0/student_best.pt`**
> 121,099 parameters · **un-distilled** · held-out test κ **0.6449** · 5.36× compression

The deployed model was **not** distilled. Knowledge distillation was attempted from two teachers, produced a statistically significant negative result (p = 0.00262) with a demonstrated mechanism, and the un-distilled compact student was deployed. See §5D and `PROJECT_REPORT.md`.

**⚠ Two distillations exist in this project.** *Staging-model* distillation (this repo, complete, negative) and *language-model* distillation (O5 / C4 / RQ4, not started). They are unrelated questions on different data. The result on one carries no implication for the other, and any write-up must say which is meant.

**Confidence fields available for the evidence packet, all validated:**

| Field | Source | Status |
|---|---|---|
| Per-stage confidence, calibrated | `reliability_table.json` — T = 1.3347, ECE 0.0454 → 0.0242 | ready |
| Per-stage reliability tiers | W, N2 **high** · N3, REM **medium** · N1 **low** | ready |
| `night_confidence` | `night_confidence.json` — ρ = −0.7502 on test, verified independent of stage composition | ready |
| Attribution descriptors | Gate 3a — see `distillation/PREREGISTRATION_gate3a.md` | **pending, go/no-go** |

---

## 1. Directory Ownership

### Group 48 — in scope

| Path | Role |
|---|---|
| `code/Phase1_48/Phase1reworked/` | Phase-1 rewrite: dataset, preprocessing, losses, metrics, training loop. **See §2.3 — `model.py` here is stale.** |
| `code/Phase1_48/mesa_transformer.py` | Alternative teacher candidate (MESA, 6-class) |
| `code/Phase1_48/mesa_dataloader.py`, `train_mesa_transformer.py`, `preprocess_mesa_signals.py` | MESA pipeline supporting the above |
| `code/Phase1_48/psg_feature_extraction.py` | Clinical feature engine (feeds disorder detection, not staging) |
| `code/Phase1_48/comorbidity_classifier*.py`, `train_ml_classifiers.py`, `focal_loss.py` | Phase-1 disorder detection |
| `code/Phase2_48/` | **Current work.** 9 notebooks. Contains the real model definitions (§2.3) |
| `processed_sleepedf/` | Sleep-EDFx tensors + spectral features + index |
| `results/sleep staging/` | All staging runs and checkpoints |
| `data/mesa/features/` | 2,056 per-patient clinical feature CSVs |

### Group 47 — out of scope, do not read for guidance

| Path | Why excluded |
|---|---|
| `code/Phase1_47/` | NAS (DARTS/ENAS) for comorbidity detection |
| `code/*.py` (repo-root level of `code/`) | Flattened NAS working copy; 9 modules have unresolvable imports |
| `code/Test/` | Scratch and dead prototypes |
| `cfs_preprocessed/`, `data/cfs_nsrr/` | CFS cohort data for Group 47 |
| `results/efficient_darts_results/`, `results/Disease detection/` | NAS results |
| `setup.py` | Packages NASLib (a different project entirely) |
| `requirements.txt` | Pins NAS-benchmark libraries; omits `mne`, `torch`-adjacent and boosting libs actually used |

---

## 2. Teacher Candidates

### 2.1 Candidate A — Fused Sleep-Staging Model (Sleep-EDFx, 5-class)

**This is the model that produced every usable staging checkpoint in the repo.**

Class definition lives **inline in `code/Phase2_48/temporal-spectral-fusion.ipynb`**, not in a `.py` file. The best checkpoint corresponds to an *earlier* revision of that notebook class (simpler spectral encoder, 3 transformer layers, no auxiliary heads).

**Inputs**
- `x_temporal`: `[B, T, 3000]` — T epochs of 30 s single-channel EEG @ 100 Hz
- `x_spectral`: `[B, T, 34]` — 34-dim spectral vector per epoch
- `padding_mask`: `[B, T]` bool, `True` at padded positions

**Layer-by-layer (best checkpoint: `concat` fusion, 714,765 params)**

| Stage | Module | In → Out |
|---|---|---|
| **Temporal branch** | reshape | `[B,T,3000]` → `[B·T, 1, 3000]` |
| | `AdaptiveAtrousPyramid.branches` — 4 × `Conv1d(1→64, k=7, dil=1/2/4/8, pad=3d)` | `[B·T,1,3000]` → 4 × `[B·T,64,3000]` |
| | `.gate` — `AdaptiveAvgPool1d(1)` → `Conv1d(256→4, k=1)` → `Softmax(dim=1)` | `[B·T,256,3000]` → `[B·T,4,1]` |
| | learned weighted sum of the 4 branches | → `[B·T,64,3000]` |
| | `SEBlock(64)` — mean → `Linear(64→4)` → ReLU → `Linear(4→64)` → sigmoid → scale | → `[B·T,64,3000]` |
| | `.proj` — `Conv1d(64→64, k=1)` | → `[B·T,64,3000]` |
| | `AdaptiveAvgPool1d(1)` + squeeze | → `[B·T,64]` |
| | `.fc` — `Linear(64→128)`, reshape | → `[B,T,128]` |
| **Spectral branch** | `spectral_encoder` — `Linear(34→128)` | `[B,T,34]` → `[B,T,128]` |
| **Fusion (`concat`)** | `cat` → `Linear(256→128)` → `LayerNorm(128)` → `GELU` | `[B,T,256]` → `[B,T,128]` |
| **Context** | `PositionalEncoding(d=128, max_len=512)`, additive | `[B,T,128]` |
| | `TransformerEncoder` × **3**: `d_model=128, nhead=4, ff=512, dropout=0.2, norm_first=True, batch_first=True` | `[B,T,128]` |
| | `cls` — `Linear(128→5)` | → `[B,T,5]` |

**Output:** logits `[B, T, 5]` over `{W, N1, N2, N3, REM}`.

### Revision provenance — why the *older* revision is the teacher

The notebook's **current** class differs from the one above in three ways: an MLP spectral encoder (`Linear(34→128)→GELU→LN→Linear(128→128)→GELU→LN`) instead of a single `Linear`, **6** transformer layers instead of 3, and auxiliary heads `temporal_cls` / `spectral_cls` for deep supervision.

**The teacher deliberately reproduces the older revision.** Every later revision scored worse:

| Revision | Trainable params | κ |
|---|---|---|
| **older — concat, 3 layers, Linear spectral, no aux** | **649,229** | **0.6663** |
| current — concat, 6 layers, MLP spectral, + aux heads | 1,262,359 | 0.5551 |

Nearly 2× the parameters for −0.11 κ. The decision was made on measurement, not on which revision is newest. `distillation/teacher_model.py` carries this rationale in its module docstring and refuses to instantiate anything but `concat`, so the choice cannot be silently reverted by someone assuming "latest = best".

The extracted class loads `concat/best_model_fusion.pt` with `strict=True` — zero missing and zero unexpected keys — which confirms the reconstruction is exact.

### 2.2 Candidate B — MESA Transformer (MESA, 6-class)

`code/Phase1_48/mesa_transformer.py`, 618 lines.

**Input:** `[B, 20, 3, 3840]` — 20 epochs × 3 EEG channels × 30 s @ 128 Hz

| Stage | Module | In → Out |
|---|---|---|
| Per-channel CNN ×3 | `Conv1d(1→64, k=50, s=6)` → BN → GELU → `MaxPool(8)`; `Conv1d(64→128, k=8, p=4)` → BN → GELU → `MaxPool(4)` | `[·,1,3840]` → `[·, ~19, 128]` |
| Projection ×3 | `Linear(128→256)` | → `[·, tokens, 256]` |
| Intra-epoch ×3 | `TemporalTransformerEncoder(d=256, nhead=8, layers=2, ff=512)` | → `[·, tokens, 256]` |
| Channel fusion | `ChannelAttentionFusion(d=256, nhead=4)` | → `[B·20, 3, 256]` |
| Epoch projection | `Linear(3·256 → 256)` | → `[B, 20, 256]` |
| Inter-epoch | `InterEpochTransformer(d=256, nhead=8, layers=2, ff=512, max_len=20)` | → `[B, 20, 256]` |
| Head | `Linear(256→512)` → GELU → Dropout → `Linear(512→6)` | → `[B, 20, 6]` |

**Output:** dict with `logits`, `probs`, `uncertainty` (per-epoch predictive entropy), and optional `temporal_attention` / `channel_attention` / `epoch_attention`.

### 2.3 ⚠ `Phase1reworked/model.py` is stale — no checkpoint matches it

`code/Phase1_48/Phase1reworked/model.py` defines `SleepStagingModel` with attributes `encoder` / `fusion` / `context`, where `fusion` is a `ChannelAttentionFusion` that projects spectral features with a bare `Linear(34→128)` and blends via 2 softmax weights.

**No checkpoint in `results/sleep staging/` has that key structure.** Every fusion checkpoint uses `temporal_encoder` / `spectral_encoder` / `fusion` / `context` — the notebook class. The prompt's assumption that `model.py` is "the teacher candidate" and that its dual-signature `forward` makes the student trivial does not hold against the actual weights.

`model_spectral.py` **does** match `spectralv1/best_model.pt` (`input_projection` + `context`, 3 layers, 665,477 params) — it is the one source file in `Phase1reworked/` that corresponds to a real checkpoint.

**Consequence for Phase 4:** the "no architectural surgery needed" assumption is false. The student needs a purpose-written class. This is straightforward, but it must be written, not inherited.

---

## 3. `processed_sleepedf/` — contents and state

| Item | State |
|---|---|
| `tensors/` | 197 `.pt`, shape `[T_epochs, 3000]` — e.g. `SC4001E0-PSG.pt` is `[841, 3000]` |
| `spectral/` | 197 `.pt`, shape `[T_epochs, 34]` — matching epoch counts |
| `index.csv` | 197 rows; columns `tensor_path`, `stage_sequence`, `spectral` |

**Broken:** both path columns point at other machines.
- `tensor_path` → `/home/geethalekshmy/GirishS/tensors/tensors/SC4001E0-PSG.pt` (Linux)
- `spectral` → `C:\PS\Sleep-Staging\processed_sleepedf\spectral\SC4001E0-PSG_spectral.pt` (Windows)

The files themselves are present locally. **This is a path-rewrite, not missing data** — Phase 1, task 1.

`stage_sequence` is a space-separated string of stage labels per epoch. Label vocabulary must be verified in Phase 1 (`N4` present ⇒ R&K scoring ⇒ merge into N3).

Recording IDs are Sleep-EDFx filenames (`SC4001E0-PSG`); subject identity is the first 5 characters (`SC400` = subject 00, night 1 vs 2). **Per-recording splitting is not sufficient — `SC4001E0` and `SC4002E0` are two nights of the same subject.** Phase 1 must split on *subject*, not recording, or the two nights leak across splits.

---

## 4. Recorded Results — what actually trained

All figures below are **validation** metrics read from the `training_metrics*.jsonl` logs. Best epoch by Cohen's κ. Parameter counts are **trainable only** — the raw `state_dict` totals are inflated by the non-trainable positional-encoding buffer (65,536 elements at `d=128, max_len=512`).

| Run | Fusion | Layers | Trainable params | Acc | **κ** | Macro-F1 |
|---|---|---|---|---|---|---|
| **`temporal spectral fusion/concat`** ← **TEACHER** | concat | 3 | **649,229** | 0.7575 | **0.6663** | 0.6814 |
| `temporal spectral fusion/Gated` | gated | 3 | 648,973 | 0.7300 | 0.6270 | 0.6469 |
| `gated fixed` | gated + aux | 6 | 1,262,103 | 0.6607 | 0.5666 | 0.6381 |
| `concat fixed` | concat + aux | 6 | 1,262,359 | 0.6544 | 0.5551 | 0.6301 |
| `cross attn fixed` | cross-attn + aux | 6 | 1,527,063 | 0.6443 | 0.5455 | 0.6224 |
| `spectralv1` (best log) | — | 3 | 599,941 | 0.6800 | 0.5502 | 0.5603 |
| `spectralv2` | — | 6 | 1,211,781 | 0.6206 | 0.5171 | 0.6059 |
| `Cross Attn` | cross-attn | 3 | 748,301 | 0.5447 | 0.3170 | 0.3391 |
| **`temporalv1`** | — | 4 | 809,869 | 0.4140 | **0.0812** | 0.1888 |
| **`temporalv2`** | — | 6 | 1,206,413 | 0.1875 | **0.0483** | 0.1713 |

### Checkpoint selection criterion

From `temporal-spectral-fusion.ipynb`, cell 11:

```python
if metrics['f1_weighted'] > best_f1:
    best_f1 = metrics['f1_weighted']
    torch.save(model.state_dict(), '/kaggle/working/best_model_fusion.pt')
```

Selection was on **weighted F1 on the validation set**, not κ. For the concat run, epoch 99 happens to maximise *both* `f1_weighted` (0.7533) and κ (0.6663), so the checkpoint on disk **is** the κ=0.6663 epoch. Verified, not assumed.

Because selection ran over 150 epochs on the same split that reports the number, κ=0.6663 is **selection-biased**, on top of the leakage described below.

### ⚠ On the κ = 0.738 figure

**κ = 0.738 appears in no committed log anywhere in this repository.** Every `training_metrics*.jsonl` was parsed; the observed ceiling across all runs is **κ = 0.6663** (concat fusion, epoch 99, validation).

If κ = 0.738 appears in the project proposal, it is unsupported by anything in the repo and should be corrected. The defensible statement is: *"best validation κ = 0.6663, from a recording-level split with subject leakage and epoch selection on the same split; no held-out test number has ever been produced."*

The **47.3% accuracy** figure in `visualize_from_results.py` is a separate matter and is **not** evidence about this model: that script's class list is 6-class `['W','N1','N2','N3','N4','REM']` and it exports `mesa_transformer.onnx`. It describes Candidate B on MESA — a different model on a different dataset.

### Three things this table says

1. **The "fixed" reruns are worse, not better.** concat 0.6663 → 0.5551; gated 0.6270 → 0.5666. Only cross-attn improved (0.3170 → 0.5455). Whatever the "fix" was, it cost the two best configurations ~0.09 κ. Do not assume "fixed" = current best.

2. **Both temporal-only baselines failed to train.** `temporalv1` peaks at κ=0.081; `temporalv2` peaks at κ=0.048 **at epoch 5 of 150** and gets worse after — 18.75% accuracy is below the 20% chance rate for 5 balanced classes. These are divergent runs, not weak models.

3. **No test split exists.** `Phase1reworked/train.py` does an 80/20 train/val split at file level and never holds out a test set. Every number above is validation.

### On the two conflicting figures in the prompt

- **κ ≈ 0.738** — does not appear anywhere in the repo. The highest recorded value is **0.6663**. If 0.738 is in the proposal, it is either from a run whose logs were not committed, or aspirational.
- **47.3% accuracy** in `visualize_from_results.py` — this is **not** the Sleep-EDF staging model. That script's class list is `['W','N1','N2','N3','N4','REM']` (6 classes) and it exports `mesa_transformer.onnx`. It describes **Candidate B on MESA**, a different model on a different dataset. It is not evidence about Candidate A.

So the Phase 2 gate question rests on a conflation of two models. The real uncertainty is narrower: *does `concat/best_model_fusion.pt` reproduce κ=0.6663 on a proper held-out split?*

---

## 5. Entry Points That Actually Work

| Task | Entry point | Notes |
|---|---|---|
| Sleep-EDFx preprocessing | `code/Phase1_48/Phase1reworked/preprocess.py` | → `processed_sleepedf/tensors/` |
| Spectral features | `code/Phase1_48/Phase1reworked/spectral_preprocess.py` | → 34-dim vectors |
| Dataset | `Phase1reworked/dataset.py` → `SleepEDFSequenceDataset` | `feature='temporal'\|'spectral'\|'fusion'`; sliding window (default 256, overlap 64) |
| Collate | `Phase1reworked/train.py` → `sleep_collate_fn` | **Already pads labels with `-100` and returns `padding_mask`** — matches the KD loss contract exactly |
| Metrics | `Phase1reworked/utils.py` → `compute_metrics` | accuracy, F1, precision, recall, ROC-AUC, AP, Cohen's κ, confusion matrix |
| Focal loss | `Phase1reworked/losses.py` → `FocalLoss` | operates on flattened `[B·T, C]` |
| Model definitions | `code/Phase2_48/temporal-spectral-fusion.ipynb`, cells 5–6 | **must be extracted to a `.py`** |

### Checkpoint loading caveat

Every checkpoint except `spectralv1/best_model.pt` was saved under `nn.DataParallel` and carries a `module.` prefix. Strip it before loading:

```python
sd = {k.removeprefix("module."): v for k, v in sd.items()}
```

Checkpoints are bare `state_dict`s — no epoch, optimizer, or config. Architecture must be inferred (as done in §4) or reconstructed from the notebook.

---

## 5A. Phase 1 Results

### Index repair — done

`processed_sleepedf/index.csv` rewritten to repo-relative POSIX paths; original preserved as `index.csv.orig`. All **394** referenced files (197 temporal + 197 spectral) verified present. `fix_index_paths.py` is idempotent and will not clobber the backup on re-run.

### Data verification — clean

| Check | Result |
|---|---|
| Recordings | 197 (153 SC + 44 ST) |
| Load failures | **0** |
| dtype | `torch.float32`, all |
| Temporal shape | `[T, 3000]`, all |
| Spectral shape | `[T, 34]`, all |
| Epoch-count alignment (temporal = spectral = labels) | **197/197 aligned, 0 mismatches** |
| Label vocabulary | exactly `{W, N1, N2, N3, REM}` — **no `N4`**, no remapping needed |
| Total epochs | 237,950 |

**Class distribution** (majority:minority = 4.6 : 1)

| Stage | Epochs | Share |
|---|---|---|
| N2 | 88,983 | 37.40% |
| W | 70,154 | 29.48% |
| REM | 34,184 | 14.37% |
| N1 | 25,175 | 10.58% |
| N3 | 19,454 | 8.18% |

N1 at 10.58% and N3 at 8.18% are the classes to watch in the Phase 6 no-rebalancing run.

### Subject-level splits — built and asserted

Subject key = `recording_id[:5]`. **Confirmed valid for both cohorts:** SC is `SC4<ss><night>E0` and ST is `ST7<ss><night>J0`, so the first 5 characters capture prefix + 2-digit subject in both cases. Verified empirically — 153 SC → 78 subjects, 44 ST → 22 subjects (exactly 2 nights each).

**97 of 100 subjects have more than one night** — precisely what a recording-level split would have leaked.

Stratified on cohort, seed 42:

| Split | Subjects | Recordings | Epochs | SC / ST | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|---|
| train | 69 | 137 | 169,909 | 107 / 30 | 31.1% | 10.3% | 36.5% | 8.0% | 14.1% |
| val | 16 | 31 | 34,610 | 23 / 8 | 24.3% | 11.8% | 41.0% | 9.0% | 13.8% |
| test | 15 | 29 | 33,431 | 23 / 6 | 26.6% | 10.5% | 38.2% | 8.2% | 16.5% |

Assertions pass on **subject IDs**: no subject in two splits, no subject lost, no recording lost.

### ⚠ The teacher's own split leaked — this constrains Phase 2

Reconstructed exactly from the notebook: `train_test_split(range(197), test_size=0.2, random_state=42)` → 157 train / 40 val, **recording-level**.

- **30 of its 35 validation subjects also appear in its training set** (86% leakage).
- Only **5 subjects / 10 recordings / 13,861 epochs** were never seen by the teacher: `SC409`, `SC428`, `SC434`, `SC459`, `SC474`. All SC — **no ST subject is teacher-clean**.
- Notebook cell 14 builds `test_dataset = FusionSleepDataset(csv_path, ...)` with **no `file_indices`** — i.e. all 197 recordings, 157 of which it trained on. The committed `confusion_matrix.png` and `roc_curves.png` under `temporal spectral fusion/` were computed on train + val combined and are **not** generalization estimates.

### 5A.1 Finding: the committed fusion evaluation figures were computed on training data

**Stated precisely, for onward discussion.**

**Where.** `code/Phase2_48/temporal-spectral-fusion.ipynb`, code cell **14** (the evaluation cell that writes `confusion_matrix.png`, `roc_curves.png` and `pr_curves.png`).

**What the code does.** The dataset used for evaluation is constructed as:

```python
WINDOW_SIZE  = 256
csv_path     = '/kaggle/working/output.csv'
test_dataset = FusionSleepDataset(csv_path, window_size=WINDOW_SIZE, overlap=0)
```

`FusionSleepDataset.__init__` has the signature `(self, csv_path, file_indices=None, window_size=256, overlap=0)` and applies row selection only when `file_indices` is supplied:

```python
self.df = pd.read_csv(csv_path)
if file_indices is not None:
    self.df = self.df.iloc[file_indices].reset_index(drop=True)
```

Cell 14 **passes no `file_indices`**. The evaluation set is therefore the entire index — all **197** recordings.

**Why that is a problem.** Training in cell 12 used:

```python
set_seed(42)
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
```

which allocates **157** recordings to training. Those 157 are a subset of the 197 that cell 14 evaluates on. Consequently the committed confusion matrices and ROC/PR curves are computed on a set that is **~80% training data**, and describe fit rather than generalization.

**Second, independent issue.** The cell-12 split is **recording-level**, not subject-level. Sleep-EDFx records most subjects across two nights (`SC4001E0` / `SC4002E0` are subject `SC400`, nights 1 and 2). Reconstructing that split shows **30 of its 35 validation subjects also appear in its training set**. So even the validation metric — the source of κ = 0.6663 — is measured largely on subjects seen during training.

**Scope of the claim.** This concerns the *evaluation protocol* only. It says nothing about the model architecture, the preprocessing, or the training procedure, all of which are sound. The affected artefacts are the reported generalization figures and the plots derived from them:

- `results/sleep staging/temporal spectral fusion/*/confusion_matrix.png`
- `results/sleep staging/temporal spectral fusion/*/roc_curves.png`
- `results/sleep staging/temporal spectral fusion/*/pr_curves.png`
- any κ / accuracy / F1 quoted from those runs as held-out performance

**What would correct it.** Re-evaluate using a subject-level held-out split — i.e. `distillation/splits.json`, whose test split shares no subject with train or val. Phase 2 of this project produces exactly that number.

**Reproducing this check.** `distillation/eval_teacher.py` reconstructs the cell-12 split deterministically (`train_test_split(range(197), test_size=0.2, random_state=42)`) and reports the teacher separately on its training recordings, its leaked validation recordings, and the 5 subjects it never saw.

---

Consequence: **14 of the 15 new test subjects were in the teacher's training set.** Teacher metrics on this split will be optimistic. `splits.json` therefore carries `teacher_provenance.subject_seen_by_teacher` per subject, so Phase 2 can report:

1. teacher on the new test split (optimistic — quantifies the leak), and
2. teacher on the 5 teacher-clean subjects (honest, but n=5 with wide CI, SC-only).

The gap between those two numbers is the decision input for whether the teacher must be retrained before distillation is defensible.

---

## 5B. Phase 2 Results — the GATE

110 recordings evaluated (all four groups fully covered), 256-epoch non-overlapping windows matching the teacher's training config. Raw per-recording predictions cached in `distillation/results/teacher_preds/`; full metrics in `distillation/results/eval_teacher.json`.

### Performance by exposure level

| Group | Epochs | Acc | **κ** | Macro-F1 |
|---|---|---|---|---|
| `teacher_train` — recordings it trained on | 79,010 | 0.8127 | **0.7412** | 0.7459 |
| `teacher_val_leaked` — val, subject also in train | 35,022 | 0.7725 | **0.6855** | 0.6953 |
| `teacher_val_clean` — **never-seen subjects** | 13,861 | 0.7191 | **0.6151** | 0.6398 |
| `new_test_split` — Phase 1 subject-level test | 33,431 | 0.7657 | 0.6782 | 0.7080 |
| *reported (validation, leaky)* | — | *0.7575* | *0.6663* | *0.6814* |

**A monotonic exposure gradient: 0.7412 → 0.6855 → 0.6151.** Performance falls with each step away from training exposure. Δ(train → clean) = **0.126 κ**.

This is more informative than the "train ≈ leaked-val" equality that would have indicated pure memorisation. What we see is a dose–response: same recording (0.7412) > same subject, different night (0.6855) > unseen subject (0.6151). The teacher memorises recordings *and* carries substantial subject-specific advantage across nights.

### Reconstruction validated

The reported κ = 0.6663 was measured on all 40 validation recordings = 30 leaked + 10 clean. Epoch-weighting the two measured subgroups:

```
(0.6855 x 35,022 + 0.6151 x 13,861) / 48,883 = 0.6655   vs reported 0.6663
```

A 0.0008 agreement confirms the split reconstruction (`train_test_split(range(197), test_size=0.2, random_state=42)`) is exactly right, and that κ = 0.6663 is a blend of mostly-leaked and partly-clean measurement.

### Per-subject, teacher-clean cohort

| Subject | Cohort | Epochs | κ | Acc | Macro-F1 |
|---|---|---|---|---|---|
| SC434 | SC | 3,083 | 0.7241 | 0.8138 | 0.5594 |
| SC428 | SC | 2,197 | 0.7217 | 0.7929 | 0.7593 |
| SC409 | SC | 2,237 | 0.6735 | 0.7823 | 0.6755 |
| SC474 | SC | 3,273 | **0.5066** | 0.6630 | 0.4107 |
| SC459 | SC | 3,071 | **0.4351** | 0.5848 | 0.4756 |

mean **0.6122**, sd **0.1330**, range 0.4351–0.7241. 95% CI (t, df=4) = **[0.447, 0.777]**.
**3 of 5 subjects sit above 0.6663; 2 sit far below.** The result straddles, and the spread is wide.

### ⚠ REM is substantially memorised

Per-class F1 by exposure — the drop is not uniform:

| Stage | train | leaked | clean | train → clean |
|---|---|---|---|---|
| W | 0.8984 | 0.8942 | 0.8922 | +0.0063 |
| N1 | 0.3740 | 0.3183 | 0.3311 | +0.0429 |
| N2 | 0.8422 | 0.8024 | 0.7419 | +0.1002 |
| N3 | 0.8180 | 0.7612 | 0.6753 | +0.1427 |
| **REM** | **0.7968** | 0.7004 | **0.5586** | **+0.2382** |

W is stable (trivially separable). N1 is poor everywhere (0.33–0.37) — a genuine difficulty, not leakage. But **REM loses 0.24 F1 between training and unseen subjects** — it looks strong at 0.80 on training data and mediocre at 0.56 on subjects it has never seen. REM performance in the published figures is largely memorisation.

For a distillation project this is the most important line in the table: REM soft targets from this teacher would be confidently wrong on unseen subjects, and the student would inherit that.

### Verdict against the pre-committed rule

> *"If clean-subset κ lands within ~0.05 of 0.6663 with a tight per-subject spread, proceed to Phase 3 with the existing teacher. Otherwise retrain."*

| Condition | Value | Met? |
|---|---|---|
| Clean κ within ~0.05 of 0.6663 | 0.6151 pooled → gap **0.0512** | ✗ (marginally) |
| Tight per-subject spread | sd **0.1330**, range **0.29**, 3 above / 2 below | ✗ (clearly) |

**→ RETRAIN.** The aggregate misses the threshold only marginally, but the spread condition fails decisively, and the REM finding is independent grounds on its own.

### No evidence for ST under clean conditions

All 5 teacher-clean subjects are SC. **Zero ST subjects were ever held out.** ST is the temazepam cohort — 22 subjects, 44 recordings, different population and hardware. Every ST number ever reported for this model was measured on subjects it trained on, so κ = 0.6151 is an **SC-only** estimate and ST generalisation is entirely unmeasured. The retrain fixes this structurally: `splits.json` is cohort-stratified and holds out 6 ST recordings.

---

## 5C. Phase 3 Results — the E1 ladder, and the E1b teacher

Phase 2 forced a retrain. The retrained teacher (**E0**) is a faithful replication of the inherited architecture, hyperparameters and schedule, differing *only* in the protocol: subject-level splits, macro-F1 checkpoint selection, test never loaded. It landed at test κ **0.5122** — honest, but weak. The E1 ladder is the attempt to make the teacher strong enough to be worth distilling from.

### The ladder

| Rung | Change from E0 | Rationale |
|---|---|---|
| **E0** | — (faithful replication, clean protocol) | Establishes the honest baseline |
| **E1a** | + focal loss, α = inverse-frequency (power 1.0) | Attack the N1/N3 minority-class problem |
| **E1b** | α power **0.5**, focal `pt` bug **fixed** | Correct E1a's compounding over-correction |

### The `pt` bug — why E1b exists

E1a computed the focal modulator as `pt = exp(-ce)` where `ce` was the **α-weighted** cross-entropy. That makes `pt = p^α`, so the modulator became `(1-p^α)²` instead of `(1-p)²`.

The consequence is not neutral. For α > 1 (N1 = 1.372, N3 = 1.774) it *inflates* the weight beyond α itself; for α < 1 (W = 0.456, N2 = 0.389) it *deflates* it further. The bug therefore **compounds** the class rebalancing rather than being orthogonal to it. E1b removes the compounding and softens α to sqrt-inverse-frequency:

| | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| E1a α (power 1.0) | 0.4563 | 1.3715 | 0.3888 | 1.7742 | 1.0091 |
| E1b α (power 0.5) | 0.7027 | 1.2182 | 0.6486 | 1.3855 | 1.0449 |

Both changes push the same direction. Flag liveness was verified before running: 12 real training steps under each config with identical seed and data diverge from step 0 (max |Δloss| 0.765), so the config demonstrably reaches the loss rather than being a decorative constant.

### Headline results — all three exposure levels

All figures are the **best-macro-F1 checkpoint**, evaluated with the same script (`eval_final.py`). `train_sample` is a 30-recording sample of the model's own training set; `val` is the 16-subject selection split; `test_heldout` is the 15-subject split, loaded exactly once.

| Model | Group | n epochs | Accuracy | κ | macro-F1 | entropy (nats) |
|---|---|---:|---:|---:|---:|---:|
| **E0** | train | 33,963 | 0.8065 | 0.7527 | 0.7911 | 0.540 |
| | val | 34,610 | 0.5910 | 0.4854 | 0.5862 | 0.652 |
| | **test** | 33,431 | **0.6160** | **0.5122** | **0.6063** | 0.655 |
| **E1a** | train | 33,963 | 0.7787 | 0.7182 | 0.7618 | 0.622 |
| | val | 34,610 | 0.6205 | 0.5290 | 0.6260 | 0.681 |
| | **test** | 33,431 | **0.6506** | **0.5592** | **0.6458** | 0.710 |
| **E1b** | train | 33,963 | 0.8924 | 0.8541 | 0.8538 | 0.573 |
| | val | 34,610 | 0.7347 | 0.6469 | 0.6977 | 0.651 |
| | **test** | 33,431 | **0.7086** | **0.6055** | **0.6548** | 0.650 |

**Improvement E0 → E1b on held-out test: accuracy +0.0926, κ +0.0933, macro-F1 +0.0485.** The exposure gradient is monotone in every case (train > val > test), which is the expected shape and confirms no leakage.

### Per-class F1 by exposure level

| Model | Group | W | N1 | N2 | N3 | REM |
|---|---|---:|---:|---:|---:|---:|
| E0 | train | 0.924 | 0.554 | 0.763 | 0.777 | 0.938 |
| | val | 0.806 | 0.382 | 0.575 | 0.571 | 0.598 |
| | test | 0.807 | 0.339 | 0.622 | 0.651 | 0.612 |
| E1a | train | 0.892 | 0.495 | 0.746 | 0.782 | 0.894 |
| | val | 0.819 | 0.453 | 0.555 | 0.583 | 0.721 |
| | test | 0.818 | 0.386 | 0.640 | 0.674 | 0.712 |
| **E1b** | train | 0.945 | 0.615 | 0.905 | 0.882 | 0.922 |
| | val | 0.877 | 0.449 | 0.750 | 0.667 | 0.745 |
| | **test** | **0.821** | **0.353** | **0.756** | **0.686** | **0.658** |

**N2 is the story.** On validation it moves 0.554 → 0.750 (+0.196) from E1a to E1b. N2 is ~38% of all epochs, so suppressing it costs far more accuracy and κ than the N1 gain could ever repay. E1a's α of 0.389, compounded by the `pt` bug, was doing exactly that. Recovering N2 is where E1b's +0.114 validation accuracy comes from.

### E1b confusion matrix — held-out test (rows = true, columns = predicted)

| | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| **W** | **7,438** | 891 | 223 | 50 | 295 |
| **N1** | 492 | **1,280** | 1,217 | 91 | 423 |
| **N2** | 594 | 1,116 | **9,610** | 660 | 798 |
| **N3** | 219 | 10 | 573 | **1,895** | 54 |
| **REM** | 475 | 457 | 1,028 | 75 | **3,467** |

Precision / recall / support, held-out test:

| Stage | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| W | 0.8069 | 0.8360 | 0.8212 | 8,897 |
| N1 | 0.3410 | 0.3654 | 0.3528 | 3,503 |
| N2 | 0.7596 | 0.7521 | 0.7558 | 12,778 |
| N3 | 0.6839 | 0.6888 | 0.6863 | 2,751 |
| REM | 0.6883 | 0.6301 | 0.6579 | 5,502 |

The dominant off-diagonal cell is **N2 → N1 (1,116)** and **N1 → N2 (1,217)** — a symmetric confusion, consistent with the separability finding below.

### The N1 finding — equal detection, honestly earned

E1a and E1b reach statistically identical N1 F1 on validation, but by completely different means:

| | N1 F1 (val) | N1 prediction ratio (val) |
|---|---:|---:|
| E1a | 0.453 | **2.41×** |
| E1b | 0.449 | **1.06×** |

E1a was not detecting N1 better — it was **guessing N1 2.4× more often than N1 occurs**, inflating recall at precision's expense. E1a never dropped below 2.3× at any point across all 75 epochs. E1b achieves the same detection at a calibrated rate, i.e. **2.3× fewer false N1 calls**.

This is confirmed independently by prior correction (below): E1a gained +0.0726 κ from post-hoc prior correction precisely *because* it was miscalibrated; E1b gains **+0.0000**.

### Prior correction — fitted on validation, applied to test

`prior_correction.py` divides the softmax by α^g and fits `g` on validation only.

| Model | fitted g | test κ as-trained | test κ corrected | Δκ |
|---|---:|---:|---:|---:|
| E1a | 0.775 | 0.5592 | **0.6318** | **+0.0726** |
| E1b | 0.025 | 0.6055 | 0.6055 | **+0.0000** |

A fitted `g` of 0.025 is essentially the identity transform. **E1b needs no post-hoc correction because its calibration is already correct in-model.** This is the cleanest available confirmation that the focal fix worked as intended rather than merely shifting numbers around.

### Separability — N1 remains representation-bound

One-vs-rest ROC-AUC and the decisive pairwise N1-vs-N2 score, held-out test:

| Model | W | N1 | N2 | N3 | REM | **N1-vs-N2 (pairwise)** |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.9449 | 0.7658 | 0.8797 | 0.9515 | 0.8968 | — |
| E1a | 0.9622 | 0.8026 | 0.9037 | 0.9553 | 0.9233 | 0.8142 |
| E1b | 0.9617 | 0.8103 | 0.8899 | 0.9575 | 0.9020 | 0.8089 |

N1-vs-N2 pairwise AUC is **unchanged** (0.8142 → 0.8089). The loss fix did not, and could not, improve N1's underlying separability — it only fixed where the operating point sat. This is consistent with the earlier E2 cancellation: N1 is limited by the representation, not by sampling or thresholds.

### Verdict — E1b is the best teacher, and still not better than the student

| Model | Params | Test κ | Test macro-F1 |
|---|---:|---:|---:|
| Inherited (honest, unseen subject only) | 649,229 | 0.5066 | 0.4107 |
| E0 teacher | 649,229 | 0.5122 | 0.6063 |
| E1a teacher | 649,229 | 0.5592 | 0.6458 |
| E1a + prior correction | 649,229 | 0.6318 | 0.6706 |
| **E1b teacher** | 649,229 | **0.6055** | **0.6548** |
| Student distilled (from E0) | 121,099 | 0.5925 | 0.6611 |
| **Student baseline** | **121,099** | **0.6449** | **0.6915** |

Paired per-subject comparison against the 121,099-parameter student baseline (15 subjects, Wilcoxon signed-rank):

| Comparison | Student better on | Mean Δκ | p |
|---|---|---:|---:|
| Student vs **E0** | 14 / 15 | +0.1337 | **0.00043** |
| Student vs **E1b** | 10 / 15 | +0.0309 | 0.135 (n.s.) |

**The correct claim is that the 121K student *matches* the 649K E1b teacher (not significantly different) at 5.36× compression, and significantly beats the faithful replication of the original design.** Do not write "outperforms E1b" — the subject-level test does not support it.

### Why E1b still loses to a model 5× smaller

| Model | train κ | test κ | Generalisation gap |
|---|---:|---:|---:|
| E0 | 0.7527 | 0.5122 | +0.2405 |
| E1a | 0.7182 | 0.5592 | +0.1590 |
| **E1b** | **0.8541** | **0.6055** | **+0.2486** |

E1b improved largely by fitting its training subjects much harder (train κ 0.7182 → 0.8541), and only part of that transferred. The remaining bottleneck is **overfitting, not loss shaping** — so the next lever, if one is ever pulled, is regularisation, not another α variant.

---

## 5D. RQ3 — does teacher quality change the distillation outcome?

Two students were distilled under **identical** conditions — same 121,099-parameter architecture, same data, splits, schedule, class weighting, `α = 0.5`, `T = 3.0`, seed 42 — differing only in which teacher supplied the soft targets. The comparison baseline (`α = 1.0`, hard labels only) is the *same stored model* in both cases: with `α = 1.0` the trainer never reads the teacher, so the baseline is teacher-independent and bit-for-bit reproducible.

### The result

| Teacher | Teacher test κ | Distilled student test κ | Gain vs baseline (0.6449) |
|---|---:|---:|---:|
| E0 | 0.5122 | 0.5925 | **−0.0523** |
| **E1b** | **0.6055** | **0.6134** | **−0.0314** |

**Distillation hurts at both points, but the penalty tracks teacher quality.** A +0.0933 κ improvement in the teacher shrank the penalty by +0.0209 κ — a **40% reduction**. Slope ≈ 0.224 κ of student penalty per κ of teacher quality; naive linear break-even would need a teacher near **κ ≈ 0.75**. Two points is directional evidence, not a fitted law, and it should be reported as such.

### The penalty is real, not noise

Paired per-subject Wilcoxon signed-rank over the 15 test subjects, E1b-distilled vs baseline:

| | Value |
|---|---|
| Distilled better on | **2 / 15** subjects |
| Mean Δκ | **−0.0352** |
| Range | −0.0871 … +0.0333 |
| **p** | **0.00262** |

So the E1b result is a **significant** negative, not a wash. Distillation reliably degrades this student on unseen subjects, on 13 of 15 held-out subjects individually.

### Report the test figure, not the validation figure

| Rung | Validation gain | **Test gain** | Ratio |
|---|---:|---:|---:|
| E0 | −0.0134 | **−0.0523** | 3.9× |
| E1b | −0.0072 | **−0.0314** | 4.4× |

Validation understates the penalty roughly four-fold in both rungs, because validation is also the checkpoint-selection split. The test column is the honest one.

### The student beats its teacher — and the baseline beats everything

| | Teacher test κ | Distilled student test κ | Δ |
|---|---:|---:|---:|
| E0 rung | 0.5122 | 0.5925 | **+0.0803** |
| E1b rung | 0.6055 | 0.6134 | **+0.0079** |

In both rungs the 121K distilled student **exceeds its own 649K teacher**. That is not the finding it appears to be: the un-distilled baseline at κ **0.6449** beats every teacher *and* every distilled student. The soft targets are not adding information — they are constraining the student toward a weaker function than it would find on its own.

### Mechanism

Two pieces of evidence, both already in the logs:

1. **The baseline moves away from the teacher as it improves.** Its KL-to-teacher rises monotonically 0.196 → 5.802 across training while its validation macro-F1 climbs 0.3312 → 0.6825. The better this student gets, the *less* it resembles the teacher — so a KD term that pulls it toward the teacher is pulling it away from its own optimum.
2. **A better-calibrated teacher is easier to match, and hurts less.** E1b's soft loss sat below E0's at every epoch (3.782 → 0.428 vs 3.988 → …), and produced exactly the smaller penalty predicted.

**Conclusion.** Knowledge distillation is counter-productive in this setting, and the size of the harm is a function of the teacher–student quality gap rather than an artefact of one bad teacher. A teacher must exceed the student's own un-distilled performance before its soft targets carry usable information; here even the best teacher (κ 0.6055) falls below the baseline student (κ 0.6449).

### Per-class, held-out test

| | W | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|
| Distilled from E0 | 0.804 | 0.390 | 0.733 | 0.703 | 0.675 |
| Distilled from E1b | 0.826 | 0.368 | 0.761 | **0.728** | 0.671 |
| **Baseline** | **0.845** | **0.389** | **0.773** | 0.737 | **0.713** |

The baseline leads on every class. Notably the E1b-distilled student reaches N3 precision 0.8248 — the highest of any model — but at recall 0.6521, so it converts to a lower F1 than the baseline's balanced 0.7438 / 0.7303.

**Checkpoint integrity:** validation reproduced locally at macro-F1 0.6806 against 0.6809 logged in training (Δ −0.0003), confirming checkpoint, model class and splits all agree before any test number was read.

---

## 6. How to Run (populated as phases land)

```
distillation/
    fix_index_paths.py        Phase 1 — rewrite index.csv to repo-relative paths
    verify_data.py            Phase 1 — integrity + class distribution + label vocab
    make_splits.py            Phase 1 — subject-level 70/15/15 → splits.json
    teacher_model.py          Phase 1 — teacher class extracted from the notebook
    eval_teacher.py           Phase 2 — GATE: teacher by exposure level
    exposure_analysis.py      Phase 2 — permanent leakage-diagnostic artefact
    eval_final.py             Phase 2/3 — held-out test, per-recording κ + entropy
    diagnose_separability.py  Phase 3 — one-vs-rest AND pairwise AUC
    prior_correction.py       Phase 3 — val-fitted class-prior correction
    cache_teacher_logits.py   Phase 3 — fp16 logit cache, resumable
    student_model.py          Phase 4 — student definition + compression accounting
    kd_loss.py (+ test)       Phase 5 — T² verified by test_kd_loss.py (13/13)
    train_student.py          Phase 5/6
    evaluate_student.py       Phase 6 — reproduces val, then evaluates test
    calibrate.py              Phase 7 — temperature scaling, ECE
    make_figures.py           Phase 7 — report figures from JSON artefacts
    reliability_table.json    Phase 7 — final deliverable

    kaggle_train_teacher.py   E0 — frozen, do not edit
    kaggle_train_improved.py  E1a / E1b / E4 ladder (EXPERIMENT constant at top)
    kaggle_train_student.py   distilled + baseline students (TEACHER_TAG at top)
```

**Kaggle scripts are paste-whole, edit-nothing.** Both carry their experiment selector as a constant at the top of the file with a warning comment, because leaving a required manual edit in the loop caused E1a to be re-run verbatim (seed 42 reproduced it bit-for-bit — the log looked correct and answered nothing, at a cost of ~5.5 GPU-hours). `kaggle_train_student.py` additionally cross-checks the attached checkpoint's own metadata against `TEACHER_TAG` and hard-fails on a mismatch, since every rung saves a file named `teacher_best.pt`.

**Baseline students are trained once and reused.** With `alpha=1.0` the trainer passes `cache if distilling else None`, and the dataset's window index derives from sequence length and stride only — never the cache. With `SEED = 42` the baseline is therefore bit-for-bit reproducible and teacher-independent, so `RUN_BASELINE = False` on later rungs. It must be set back to `True` if the student architecture, `CLASS_WEIGHT_POWER`, `USE_SPECTRAL`, `EPOCHS` or `SEED` ever change.

**Environment:** Kaggle free tier — ~30 GPU-h/week, ≤12 h sessions, disk wiped between sessions. Every long-running script checkpoints to `/kaggle/working/` and resumes. Runtimes observed: E1b teacher 333.8 min (75 epochs, single T4), student rung ~3 h with the baseline skipped.

### Report figures

`python distillation/make_figures.py` regenerates all eight from the JSON artefacts; add `--dark` for the dark-mode set in `figures_dark/`. **No numbers are hardcoded in that script**, so re-running after any new rung updates every figure. Regenerate it after any evaluation, or the figures will silently disagree with the tables.

| Figure | Shows | Source artefact |
|---|---|---|
| `fig1_exposure_gradient` | The leakage proof — inherited κ by exposure level, against the published 0.6663 | `exposure_gradient.json` |
| `fig2_compression_vs_kappa` | Every model, parameters vs test κ | `eval_final_*.json`, `eval_students.json` |
| `fig3_per_class_f1` | Per-class F1, four models | same |
| `fig4_pairwise_auc` | Pairwise separability — isolates N1-vs-N2 as the bottleneck | `separability_diagnostic_E1b_E1a.json` |
| `fig5_kl_vs_performance` | **The RQ2 mechanism** — two stacked panels: macro-F1 rising while the baseline's KL-to-teacher rises with it | `students/*/training_metrics.jsonl` |
| `fig6_confusion_matrix` | Best model, row-normalised | `eval_students.json` |
| `fig7_reliability_diagram` | Calibration before/after temperature scaling | `reliability_table.json` |
| `fig8_rq3_teacher_quality` | **The RQ3 finding** — distillation penalty against teacher quality | `eval_final_*.json`, `eval_students.json` |

Two deliberate constraints in that script, both easy to undo by accident:

- **Fig 3 shows four series, not six.** The categorical order validates four adjacent slots; a fifth puts yellow beside orange and fails the colour-separation floor. E1a and the E0-distilled student are omitted from the chart and reported in the tables instead.
- **Fig 5 is two stacked panels sharing an x-axis, never a dual-axis chart.** Two y-scales on one plot manufacture a correlation the data does not contain.

---

## 7. Open Issues Carried Into Later Phases

| # | Issue | Status | Affects |
|---|---|---|---|
| 1 | Teacher class existed only inside a notebook | ✅ extracted to `distillation/teacher_model.py`, loads `strict=True` | — |
| 2 | `Phase1reworked/model.py` matches no checkpoint — **stale** | ✅ documented (§2.3); teacher does not import it | Phase 4 |
| 3 | 2 nights per subject; must split on subject | ✅ done, subject key validated for SC **and** ST | — |
| 4 | `index.csv` had foreign absolute paths | ✅ repaired, 394/394 files verified | — |
| 5 | Teacher's own split was recording-level; 86% subject leakage | ✅ **closed** — retrained under subject-level splits (E0/E1a/E1b); inherited model retained only as a leakage reference | — |
| 6 | No held-out test set has ever existed; κ=0.6663 is validation + selection-biased | ✅ **closed** — 15-subject test split built, loaded once per model; honest figures in §5C | — |
| 7 | κ=0.738 unsupported by any committed log; ceiling is 0.6663 | ✅ documented (§4) — **correct the proposal** | — |
| 8 | Committed fusion confusion matrices / ROC curves were computed on train+val combined | ⚠ do not cite as generalization | reporting |
| 9 | `temporalv1`/`temporalv2` diverged (κ 0.081 / 0.048) — unusable as a baseline | ✅ **closed** — baseline regenerated via the `alpha=1.0` arm of the same trainer; test κ 0.6449 | — |
| 12 | E1a focal `pt = exp(-ce)` computed `p^α`, compounding the class rebalancing | ✅ **closed** — fixed in E1b (`FIX_FOCAL_PT`); +0.0933 test κ over E0 (§5C) | — |
| 13 | E1b overfits: train κ 0.8541 vs test 0.6055, gap +0.2486 | ⚠ **open** — remaining bottleneck is regularisation, not loss shaping | future work |
| 14 | ST cohort generalises far worse than SC (test κ 0.4755 vs 0.6182) | ⚠ **open** — 3 ST subjects held out; population + hardware differ | reporting |
| 15 | Test subject SC461 is an outlier (κ 0.2378 vs next-worst 0.4385) across every model | ⚠ note — recording-quality artefact, not a model failure; drags the mean ~0.025 | reporting |
| 10 | All checkpoints are bare `state_dict` with `module.` prefix | ✅ handled in `load_teacher()` | — |
| 11 | Param counts in raw `state_dict` include the 65,536-element PE buffer | ✅ §4 reports trainable-only | reporting |
