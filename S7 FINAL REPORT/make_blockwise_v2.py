"""Block-wise system architecture, release v2: the successor of distillation/figures/fig_blockwise_architecture.png
(hand-drawn for v1; that file is left unchanged). Same seven-panel layout; every value is from the v2 code or a
results file:
  inputs/features   agentic/signals.py (34 = DWT db4 L5 x 4 stats + 10 STFT), trainer eog_epoch_features (8)
  model             ADDING EOG VERSION - ADVANCED STUDENT/kaggle_train_student_advanced.py (AdvancedStudent, arm "full")
  results           CURRENT_RELEASE.md, ROADMAP P8, R2b, friend_bundle/RESULTS_RUN1.md

    python "S7 FINAL REPORT/make_blockwise_v2.py"
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "images" / "A0_blockwise_architecture_v2.png"
COPY = HERE.parent / "distillation" / "figures" / "fig_blockwise_architecture_v2.png"   # next to the v1 original

COL = {"data": "#CFE8E1", "grey": "#E6E7EA", "eeg": "#CADCF2", "spec": "#F4CCD3", "eog": "#D9E8CB",
       "tf": "#F5EDC1", "gate": "#D2EAD3", "fuse": "#DCD3EA", "out": "#F8DCC0", "red": "#F8E4E4",
       "slm": "#CDE9E6", "ok": "#D5ECD7"}


def panel(ax, x, y, w, h, title):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=2.4, ec="#111"))
    ax.text(x + 0.5, y - 0.4, title, fontsize=14, weight="bold", va="bottom")     # header just above the frame


def b(ax, x, y, w, h, title, body="", c="grey", fs=9.2, tfs=10.2, ec="#333", lw=1.2, ls="-", tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.4", fc=COL.get(c, c),
                                ec=ec, lw=lw, ls=ls))
    if body:                                  # y axis is inverted: the box top is y
        ax.text(x + w / 2, y + 0.7, title, ha="center", va="top", fontsize=tfs, weight="bold", color=tc)
        ax.text(x + w / 2, y + 1.3 + tfs / 7.9, body, ha="center", va="top", fontsize=fs, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tfs, weight="bold", color=tc)


def a(ax, p, q, c="#222", ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15, lw=lw, color=c, ls=ls))


def note(ax, x, y, s, fs=9, c="#444", ha="center", style="italic", weight="normal"):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va="center", style=style, weight=weight)


def strip(ax, x, y, w, label, text, hl=False):
    ax.add_patch(Rectangle((x, y), w, 2.4, fc="#EAF1FA" if hl else "#F2F2F2", ec="#1F4E79" if hl else "#999", lw=1.3))
    ax.text(x + 1, y + 1.2, label, fontsize=9.5, weight="bold", color="#1F4E79" if hl else "#555", va="center")
    ax.text(x + 11, y + 1.2, text, fontsize=9.2, va="center")


def main():
    fig, ax = plt.subplots(figsize=(22, 22))
    ax.set_xlim(0, 200); ax.set_ylim(0, 200); ax.axis("off"); ax.invert_yaxis()
    ax.text(100, 0.8, "Block-Wise System Architecture (Release v2): Distilled EEG + EOG Staging > Evidence Packet > "
            "Verified On-Device Report", ha="center", va="center", fontsize=17, weight="bold")

    # ---------------------------------------------------------------- 1. input and representation
    panel(ax, 2, 6, 42, 62, "1. Input and Representation")
    note(ax, 23, 9, "PSG signals (Sleep-EDFx, PhysioNet)", fs=11, c="#111", style="normal", weight="bold")
    b(ax, 5, 11, 36, 7, "EEG Fpz-Cz  +  horizontal EOG", "197 recordings - 100 subjects - 237,950 epochs", "data")
    b(ax, 5, 20.5, 36, 7, "Preprocessing", "100 Hz - 30 s epochs - corrected labels\n(drop-then-trim bug: 5-7% labels shifted)")
    b(ax, 5, 30, 36, 9, "Feature extraction (per epoch)", "EEG: 34 = DWT db4 L5 (6 bands x 4 stats)\n+ STFT band powers, ratios, entropy, SEF95\n"
      "EOG: 8 = 4 band log-powers, variance,\nsaccade rate, mean |dx|, EOG-EEG corr.", fs=8.6)
    for i in range(3):
        a(ax, (23, 18 + i * 9.5 - (0 if i < 2 else 0)), (23, 20.5 + i * 9.5)) if i < 2 else None
    tens = [("[B,T,3000]\nEEG", "eeg"), ("[B,T,3000]\nEOG", "eog"), ("[B,T,34]\nspectral", "spec"),
            ("[B,T,8]\nEOG feats", "eog")]
    for i, (t, c) in enumerate(tens):
        b(ax, 4.5 + i * 9.4, 42, 8.6, 7, t, "", c, tfs=8.2)
        a(ax, (23, 39), (8.8 + i * 9.4, 42), lw=1.0)
    note(ax, 23, 52, "subject-level split 69 / 16 / 15\n(test locked, every read logged)", fs=9, c="#1F4E79", style="normal", weight="bold")
    note(ax, 23, 55.5, "B = batch   T = 256 epochs per window (~128 min)", fs=8.8)
    note(ax, 23, 58.5, "3000 = 30 s at 100 Hz   emb = 64", fs=8.8)

    # ---------------------------------------------------------------- 2. feature extractor
    panel(ax, 46, 6, 152, 62, "2. Feature Extractor - Gated Multi-Branch Encoder (v2)")
    rows = [(10, "EEG channel", "eeg", "Multi-scale epoch encoder (v1 backbone)",
             "fine: Conv k50 s6 (24) > pool 8 > Conv k8 (48)\ncoarse: Conv k200 s25 (24) > pool 4 > Conv k6 (48)\n"
             "4 tokens > Linear 96->64 > attention pooling  -  receptive field 8.75 s"),
            (22, "EOG channel", "eog", "EOG encoder (6,768)",
             "saccade: Conv k50 s6 > Conv k8   -   slow eye movements: Conv k400 s50 > Conv k5\n+ 8 EOG features > 32-d, LayerNorm"),
            (34, "EEG (TF view)", "tf", "Time-frequency encoder (7,848)",
             "STFT 2 s Hann / 1 s hop, 0.5-30 Hz (60 bins), log power, BatchNorm\n2 x Conv over time > mean + max pool > 32-d"),
            (46, "Spectral", "spec", "Spectral projection (v1)", "Linear 34 -> 64 (no activation)")]
    for y, lab, c, t, body in rows:
        b(ax, 48, y, 15, 8, lab, "", c, tfs=9.5)
        b(ax, 66, y, 64, 8.5, t, body, c, fs=8.0, tfs=9.6)
        a(ax, (63, y + 4), (66, y + 4))
    b(ax, 135, 10, 20, 8.5, "h (64-d)", "per epoch", "eeg", tfs=9.6)
    b(ax, 135, 22, 20, 8.5, "+ gated EOG (8,320)", "h += sigmoid(G[h,e]) * P e", "gate", fs=8.4, tfs=9.4)
    b(ax, 135, 34, 20, 8.5, "+ gated TF (8,320)", "same closed gate", "gate", fs=8.4, tfs=9.4)
    a(ax, (130, 14), (135, 14)); a(ax, (130, 26), (135, 26)); a(ax, (130, 38), (135, 38))
    a(ax, (145, 18.5), (145, 22)); a(ax, (145, 30.5), (145, 34))
    b(ax, 160, 22, 35, 20.5, "Channel fusion (v1)", "concat[h, spectral] > Linear 128->64\nLayerNorm + GELU\n\n"
      "Local context (4,672)\ndepthwise Conv k7 over +/-3 epochs\nzero-initialised residual", "fuse", fs=8.6, tfs=9.8)
    a(ax, (155, 38), (160, 34)); a(ax, (130, 50), (177, 42.5))
    b(ax, 160, 46, 35, 8.5, "[B, T, 64]", "to the sequence model", "fuse", tfs=9.4)
    a(ax, (177.5, 42.5), (177.5, 46))
    note(ax, 145, 53.5, "gates closed at init: P = 0, gate bias -4", fs=8.6)
    strip(ax, 48, 57, 108, "TEACHER", "ENSmc = mean softmax of 3 students: 121,099 + 139,606 + 151,606 = 412,311 params (EEG only)")
    strip(ax, 48, 61, 108, "STUDENT", "v2 N6adv: v1 backbone 139,606 + EOG 15,088 + TF 16,168 + local 4,672 + aux 65 = 175,599", hl=True)
    note(ax, 177, 61, "untrained v2 = exactly v1's\nfunction; trained with\nmodality dropout 0.25", fs=8.8, c="#1F4E79", style="normal", weight="bold")

    a(ax, (100, 68), (100, 72)); a(ax, (194, 68), (194, 72))

    # ---------------------------------------------------------------- 3. sequencing
    panel(ax, 2, 72, 128, 46, "3. Sleep Staging Sequencing")
    b(ax, 5, 82, 16, 9, "[B, T, 64]", "fused epochs", "fuse", tfs=9.5)
    b(ax, 25, 82, 15, 9, "Positional\nencoding", "", "tf", tfs=9.5)
    ax.add_patch(Rectangle((44, 77), 50, 20, fill=False, ec="#667", ls="--", lw=1.2))
    note(ax, 69, 75.5, "Transformer encoder x 2 layers (pre-norm, norm_first=True)", fs=9.8, c="#111", style="normal", weight="bold")
    for i, (t, c) in enumerate([("Norm >\nMulti-Head\nAttention\n(2 heads)", "ok"), ("Add", "tf"), ("Norm >\nFeed\nForward\n(256)", "spec"), ("Add", "tf")]):
        w = 13 if i % 2 == 0 else 6
        x = 45.5 + [0, 14, 21, 35][i]
        b(ax, x, 79, w, 16, t, "", c, tfs=9)
    b(ax, 98, 80, 13, 12, "Linear\n64 -> 5", "", "eeg", tfs=9.5)
    b(ax, 114, 80, 13, 12, "Softmax\nper epoch", "", "fuse", tfs=9.5)
    for p, q in (((21, 86.5), (25, 86.5)), ((40, 86.5), (44, 86.5)), ((94, 86.5), (98, 86.5)), ((111, 86), (114, 86))):
        a(ax, p, q)
    note(ax, 69, 101, "Output logits [B, T, 5]   =   W | N1 | N2 | N3 | REM", fs=11, c="#111", style="normal", weight="bold")
    note(ax, 69, 104.5, "attention spans the whole 256-epoch window  -  padding mask hides positions where a recording ends", fs=8.8)
    strip(ax, 5, 107.5, 122, "AUX HEAD", "Linear 64 -> 1 predicts stage transitions (65 params, training only, loss weight 0.1)")
    strip(ax, 5, 111.5, 122, "v2 TEST", "81.96% accuracy - kappa 0.756 - macro-F1 0.782 (15 held-out subjects); 0.744 without EOG; v1 0.718", hl=True)

    # ---------------------------------------------------------------- 4. KD
    panel(ax, 132, 72, 66, 46, "4. Knowledge Distillation (works)")
    b(ax, 135, 76, 28, 7, "Ensemble teacher (frozen)", "logits cached once", "grey", fs=8.6, tfs=9.4)
    b(ax, 167, 76, 28, 7, "v2 student", "warm start from v1, 10 epochs frozen", "eeg", fs=8.6, tfs=9.4)
    b(ax, 152, 86, 26, 5, "p_T = softmax(z / T)", "", "tf", tfs=9.4)
    a(ax, (149, 83), (160, 86)); a(ax, (181, 83), (170, 86)); a(ax, (165, 91), (165, 93))
    b(ax, 135, 93, 60, 8.5, "L = alpha CE + (1 - alpha) T^2 KL  + 0.1 L_transition",
      "alpha = 0.5   T = 3 (T^2 mandatory)   AdamW 3e-4, 75 epochs, EMA 0.999,\nlabel smoothing 0.2, gain 0.75-1.33, quality mask", "ok", fs=8.4, tfs=9.6)
    b(ax, 135, 103.5, 60, 12, "RESULT: the student beats its teacher",
      "kappa 0.756 vs teacher 0.732 with 43% of its parameters (held-out test)\nLike-for-like EOG gain +0.018 kappa, 5/5 CV folds\n"
      "Why: single-teacher KD hurt (0.6134 < control 0.6449, v1 era);\nan ensemble teacher beats the undistilled student", "red",
      fs=8.4, tfs=9.6, ec="#9B2C2C", tc="#9B2C2C")

    a(ax, (126, 118), (126, 122)); a(ax, (165, 118), (165, 122))

    # ---------------------------------------------------------------- 5. evidence packet
    panel(ax, 2, 122, 196, 22, "5. Evidence Packet (schema 1.3) - the only context any downstream module may read")
    items = [("Predicted\nhypnogram", "eeg"), ("Calibrated\nprobabilities", "eeg"), ("Per-stage\nreliability", "ok"),
             ("Night-level\nconfidence", "ok"), ("24 derived\nsleep metrics", "fuse"), ("Measured error\nper metric", "fuse"),
             ("Hedge / assert\nflag per metric", "out"), ("Model\nprovenance", "grey")]
    for i, (t, c) in enumerate(items):
        b(ax, 5 + i * 24, 125, 22, 7.5, t, "", c, tfs=9.4)
    note(ax, 100, 135.5, "one versioned JSON per night  -  19 evidence items with stable IDs  -  3 robust / 13 fragile / 8 unreliable metrics",
         fs=10, c="#111", style="normal", weight="bold")
    note(ax, 100, 138.5, "calibration T = 1.0021 (already calibrated)  -  ECE 0.0133  -  night confidence vs expert agreement r = -0.637 (test)  -  "
         "tiers: kappa 0.788 high > 0.731 > 0.689 low", fs=9.2, c="#1F4E79")

    a(ax, (100, 144), (100, 148)); a(ax, (192, 144), (192, 148))

    # ---------------------------------------------------------------- 6. SLM + verification
    panel(ax, 2, 148, 108, 48, "6. Offline Small Language Model + Deterministic Verification")
    b(ax, 5, 151, 30, 11, "Prompt P1", "packet serialised\nchat template pinned\nto the training date", "tf", fs=8.6, tfs=9.6)
    b(ax, 39, 151, 34, 11, "Llama-3.2-3B-Student v2", "LoRA r16 on 123 nights > GGUF Q4_K_M\n2.02 GB, llama-completion b10927\nGBNF grammar - temp 0 - seed 0",
      "slm", fs=8.4, tfs=9.6)
    b(ax, 77, 151, 30, 11, "Claims (JSON)", "every claim cites an\nevidence ID and a\npacket field + value", "eeg", fs=8.6, tfs=9.6)
    a(ax, (35, 156.5), (39, 156.5)); a(ax, (73, 156.5), (77, 156.5)); a(ax, (92, 162), (80, 166))
    b(ax, 5, 166, 47, 11, "Layer 1 - structure (closed world)", "valid JSON, known claim types, fields,\nevidence IDs, enums, no duplicates", "ok", fs=8.6, tfs=9.4)
    b(ax, 57, 166, 50, 11, "Layer 2 - policy (reads the packet)", "cited value == packet value  -  unreliable\nvalues hedged  -  5 mandatory items for the tier",
      "ok", fs=8.6, tfs=9.4)
    a(ax, (52, 171.5), (57, 171.5))
    note(ax, 56, 179.5, "30 violation codes, frozen and hash-pinned - no language model checks a language model", fs=9, c="#111", style="normal", weight="bold")
    b(ax, 5, 182, 32, 7.5, "0 violations > VERIFIED_REPORT", "", "ok", tfs=9.2)
    b(ax, 40, 182, 34, 7.5, "any violation > FALLBACK_SUMMARY", "", "red", tfs=9.2)
    b(ax, 77, 182, 30, 7.5, "untraced number > REVIEW_ONLY", "", "out", tfs=9.2)
    note(ax, 56, 193, "Test (29 locked nights): 28 verified, 1 blocked, 0 flawed reports released  -  "
         "untrained base model: 31/31 flawed without the gate", fs=9, c="#1F4E79", style="normal", weight="bold")

    a(ax, (110, 172), (112, 172))

    # ---------------------------------------------------------------- 7. agent + delivery + edge
    panel(ax, 112, 148, 86, 48, "7. Agentic Workflow > Delivery on the Edge")
    steps = ["ingest + QC", "preprocess", "stage pass 1", "find the night", "stage pass 2", "build packet", "report + verify", "deliver once"]
    for i, s in enumerate(steps):
        r, k = divmod(i, 4)
        x, y = 115 + k * 20.5, 151 + r * 9
        b(ax, x, y, 18, 6.5, s, "", "grey", tfs=9)
        if k:
            a(ax, (x - 2.5, y + 3.25), (x, y + 3.25), lw=1.2)
    a(ax, (186, 157.5), (124, 160), lw=1.2)
    b(ax, 115, 169, 39, 11, "Deterministic agent", "run id = hash(input, policy, release)\nresume from step markers - retries\nRAM / heat wait - trace - review queue",
      "fuse", fs=8.4, tfs=9.4)
    b(ax, 157, 169, 38, 11, "One page per night", "Clinician report  -  Simple view\n(plain language, graded only\nwith robust/fragile metrics)", "out", fs=8.4, tfs=9.4)
    b(ax, 115, 182, 80, 7.5, "Edge (Pi 5 designed): offline Docker 4 CPU / 16 GB: 31/31 reports verified, identical, 151 s, 5.2 GB",
      "", "grey", tfs=8.6)
    note(ax, 155, 193, "14/14 fault scenarios, invariants I1-I8, 31/31 byte-identical reruns  -  Path B mean kappa 0.701", fs=9,
         c="#1F4E79", style="normal", weight="bold")

    note(ax, 100, 198.5, "Modules 1-7 built and measured on x86  -  edge: x86 container and ARM64 staging parity (emulated) measured; "
         "Raspberry Pi 5 timing and multi-night trends are the next phase", fs=10.5, c="#555")
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(OUT, COPY)
    print("wrote", OUT, "and", COPY)


if __name__ == "__main__":
    main()
