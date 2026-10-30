# Matrices of Teacher and Student Model

This document provides a comprehensive breakdown of the performance metrics and per-class values for the key models in this project. All evaluations are performed on the **held-out Test Set** (29 recordings, 33,431 epochs from 15 unseen subjects).

---

## 1. Overall Performance Metrics

Here is the high-level comparison of the baseline, the final teacher ensemble, and the final deployed student model.

| Metric | Baseline (`student_baseline_E0`) | Teacher Ensemble (`ENSmc`) | Final Student (`student_N4kd`) |
|---|---|---|---|
| **Accuracy** | 73.60% | **78.68%** | 77.69% |
| **Cohen's Kappa** | 0.6449 | **0.7140** | 0.6993 |
| **Macro F1 Score** | 69.15% | **75.42%** | 73.76% |
| **Parameters** | 121,099 (Single-Scale) | ~412,000 (3 Models) | 139,606 (Multi-Scale) |
| **Receptive Field** | 0.25 seconds | 8.75 seconds | 8.75 seconds |

> **Summary:** The Teacher Ensemble (ENSmc) achieves the highest scores across all metrics by combining the strengths of three different models. The Final Student (N4kd), distilled from this teacher, retains most of the performance gains while being nearly 3x smaller, making it highly efficient for deployment.

---

## 2. Per-Class F1 Scores (Sleep Stages)

This table shows how well each model detects individual sleep stages. F1 score balances precision (not false-alarming) and recall (not missing true events).

| Sleep Stage | Baseline F1 | Teacher F1 (`ENSmc`) | Final Student F1 (`N4kd`) | Note |
|---|---|---|---|---|
| **W (Wake)** | 80.4% | 89.4% | **89.7%** | The student actually slightly outperforms the teacher here. |
| **N1 (Light Sleep)** | 39.0% | **46.2%** | 43.0% | Hardest stage to predict without EOG (Eye) channels. |
| **N2 (Core Sleep)** | 73.3% | **80.9%** | 80.5% | Huge jump from baseline due to the 8.75s receptive field catching Sleep Spindles. |
| **N3 (Deep Sleep)** | 70.3% | **79.4%** | 77.9% | Significant gain due to Coarse Branch catching slow Delta waves. |
| **REM (Dreaming)** | 67.5% | **81.3%** | 77.7% | Strong improvement, further aided by the `L=3` sequence decoder. |

---

## 3. Deep Dive: Final Student (`student_N4kd`) Detailed Matrices

For a complete picture of the deployed model, here are the detailed Precision, Recall, and Confidence metrics per class.

| Sleep Stage | F1 Score | Precision | Recall | Mean Confidence | Reliability Status |
|---|---|---|---|---|---|
| **W (Wake)** | **89.7%** | 95.8% | 84.4% | 91.6% | **HIGH** |
| **N1 (Light Sleep)** | **43.0%** | 38.5% | 48.7% | 55.8% | **LOW** (Representation-bound) |
| **N2 (Core Sleep)** | **80.5%** | 79.8% | 81.3% | 71.1% | **HIGH** |
| **N3 (Deep Sleep)** | **77.9%** | 76.4% | 79.4% | 74.3% | **HIGH** |
| **REM (Dreaming)** | **77.7%** | 79.3% | 76.1% | 75.2% | **HIGH** |

### Why is N1 performance low?
N1 is defined physiologically by *slow rolling eye movements*. Because our final model uses **only EEG (brainwaves)** and no EOG (eye movement) channels, it lacks the necessary information to distinguish N1 perfectly. Our formal analysis proved that no mathematical trick can fix this; it requires new data (EOG). This is why the Evidence Packet flags N1 predictions as "Low Reliability" to prevent the downstream LLM from making false claims.

---

## 4. Key Takeaways for Code Review

1. **The Architecture Upgrade Worked:** Moving from a single-scale (0.25s) to a multi-scale (8.75s) architecture allowed the models to actually "see" long sleep events like Spindles (N2) and Slow Waves (N3). This alone drove a massive +7% to +14% F1 increase across stages.
2. **The Ensemble Teacher is Strong:** By averaging probabilities from three diverse models (including one with EOG), the `ENSmc` teacher corrected individual model errors, achieving a 0.714 Kappa.
3. **Distillation was Successful:** The student model (`N4kd`) successfully learned from the teacher's soft probabilities, capturing the inter-class nuances (like N2/N3 transitions) and achieving a 0.699 Kappa—far superior to the 0.645 baseline, while remaining small enough to run locally.

---

## 5. Model Architectures & Channels

### Channels Overview
| Model | Channels Used | Why? |
|---|---|---|
| **Baseline (`E0`)** | 1 Channel: `EEG Fpz-Cz` | Starting point, relies only on frontal brainwaves. |
| **Final Student (`N4kd`)** | 1 Channel: `EEG Fpz-Cz` | Designed for lightweight local deployment using only EEG. EOG was left for future work. |
| **Teacher Ensemble (`ENSmc`)** | Mixed (1-Channel and 3-Channel models) | Averages 3 models. Two use 1 channel (`EEG Fpz-Cz`). One uses 3 channels (`EEG Fpz-Cz`, `EOG horizontal`, `EEG Pz-Oz`) to capture Eye movements (N1/REM) and Occipital brainwaves (N3). |

*> **Note:** ECG (heart rate) was not used in any of these models.*

### A. Baseline Model Architecture (`student_baseline_E0`)
- **Encoder:** Single-Scale 1D-CNN
- **Receptive Field:** 0.25 seconds (25 samples) — *Too small to see actual physiological sleep events!*
- **Transformer:** 1 token (Global Average Pool collapses temporal structure)
- **Parameters:** 121,099
```text
Input (1 EEG channel) 
  -> Conv1d(kernel=25) 
  -> Global Average Pool (1 token) 
  -> Transformer Encoder (2 layers, 64 dim) 
  -> Linear Classifier 
  -> Softmax (5 classes)
```

### B. Teacher Ensemble Architecture (`ENSmc`)
The teacher averages the softmax probability predictions of **3 independently trained models**:
1. `student_baseline_E0` (Single-Scale, EEG only)
2. `student_N2multiscale_fix` (Multi-Scale, EEG only)
3. `student_N3mc_s42` (Multi-Scale, **EEG + EOG + EEG Pz-Oz**)

**The Multi-Scale Backbone (used in models 2 & 3):**
- **Fine Branch:** `Conv1d(kernel=50, stride=6)` → 0.5s window (catches fast Spindles / K-Complexes)
- **Coarse Branch:** `Conv1d(kernel=200, stride=25)` → 2.0s window (catches slow Delta / Sawtooth waves)
- **Combined Receptive Field:** 8.75 seconds (875 samples)
- **Transformer:** 4 tokens per epoch (Attention Pooling preserves within-epoch structure)
- **Parameters:** ~139K (1-channel) / ~151K (3-channel)

### C. Final Student Architecture (`student_N4kd`)
Uses the exact same **Multi-Scale backbone** as the strong teacher members, but restricted to 1 channel (EEG Fpz-Cz) for local CPU deployment.

- **Branches:** Fine (0.5s) + Coarse (2.0s) parallel convolutions
- **Combined Receptive Field:** 8.75 seconds
- **Transformer:** 4 tokens per epoch, 64 embedding dim, 2 heads, 2 layers, 0.2 dropout
- **Sequence Window:** 256 epochs (model sees ~2.1 hours of continuous sleep per forward pass)
- **Parameters:** 139,606
- **Loss Function:** 50% Knowledge Distillation Loss (Temperature=3.0) + 50% Cross-Entropy Loss
```text
Input (1 EEG channel) 
  -> [Fine Branch Conv1d] + [Coarse Branch Conv1d] 
  -> Concatenate (RF = 8.75s)
  -> Attention Pool (4 tokens)
  -> Transformer Encoder (2 layers, 64 dim) 
  -> Linear Classifier 
  -> Softmax (5 classes)
```
