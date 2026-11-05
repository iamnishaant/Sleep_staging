"""Simple one-page architecture of release v2 (slides / report). Same facts as make_blockwise_v2.py, fewer details.

    python "S7 FINAL REPORT/make_simple_architecture.py"
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "images" / "A00_simple_architecture_v2.png"


def panel(ax, x, y, w, h, title, fc, ec, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=1.8, ls=ls))
    ax.text(x + w / 2, y + 2.3, title, ha="center", va="center", fontsize=11.5, weight="bold", color="#111")


def box(ax, x, y, w, h, text, fc="white", ec="#555", fs=9, weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=ec, lw=1.1))
    tx = x + w / 2 if ha == "center" else x + 1.5
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, weight=weight, linespacing=1.3)


def arrow(ax, p, q, c="#222", lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=lw, color=c, shrinkA=0, shrinkB=0))


def caption(ax, x, s):
    ax.text(x, 91.5, s, ha="center", va="top", fontsize=9.5, color="#333", style="italic", linespacing=1.3)


def main():
    fig, ax = plt.subplots(figsize=(26, 11))
    ax.set_xlim(0, 264); ax.set_ylim(108, -4); ax.axis("off")
    ax.text(132, -1.5, "Release v2 Architecture: Distilled EEG + EOG Sleep Staging  →  Evidence Packet  →  "
            "Verified On-Device Report", ha="center", va="center", fontsize=16, weight="bold")

    # training-only region (teacher + KD never ship to the device)
    ax.add_patch(FancyBboxPatch((56, 3.5), 150, 39, boxstyle="round,pad=0,rounding_size=1.5", fc="#FAFAFA",
                                ec="#C00000", lw=1.5, ls="--"))
    ax.text(204, 5.6, "TRAINING ONLY (not on the device)", ha="right", va="center", fontsize=9.5, weight="bold",
            color="#C00000")

    # 1. input
    panel(ax, 2, 8, 22, 82, "1. Input", "#DDEBF7", "#2F5597")
    box(ax, 4, 15, 18, 9, "One night of PSG\n(Sleep-EDFx)", weight="bold")
    box(ax, 4, 31, 18, 8, "EEG\nFpz-Cz", fc="#CADCF2")
    box(ax, 4, 43, 18, 8, "Horizontal\nEOG", fc="#D9E8CB")
    ax.text(13, 57, "sampled at 100 Hz", ha="center", fontsize=9, color="#333")
    box(ax, 4, 64, 18, 12, "Unit of input:\n30-s epoch\n\nlabel: W, N1, N2,\nN3 or REM", fs=8.6)
    caption(ax, 13, "Two-channel\nsleep recording")
    arrow(ax, (24, 49), (27, 49))

    # 2. preprocessing
    panel(ax, 27, 8, 28, 82, "2. Preprocessing", "#FFF2CC", "#BF9000")
    steps = [(15, 7, "Resample to 100 Hz"), (25, 7, "Cut into 30-s epochs\n(3000 samples each)"),
             (35, 14, "Features per epoch\n34 EEG spectral\n(24 DWT + 10 STFT)\n+ 8 EOG features"),
             (52, 7, "Group 256 epochs\ninto one window")]
    for i, (y, h, t) in enumerate(steps):
        box(ax, 29, y, 24, h, t, fs=8.6)
        if i:
            arrow(ax, (41, steps[i - 1][0] + steps[i - 1][1]), (41, y), lw=1.2)
    arrow(ax, (41, 59), (41, 63), lw=1.2)
    for y, t, c in ((63, "EEG  256 × 3000", "#CADCF2"), (70, "EOG  256 × 3000", "#D9E8CB"),
                    (77, "Features  256 × 42", "#F4CCD3")):
        box(ax, 29, y, 24, 5.5, t, fc=c, fs=8.6)
    caption(ax, 41, "Signals become epochs\n+ hand-made features")

    arrow(ax, (55, 20), (58, 20))
    ax.text(56.5, 17.5, "EEG", ha="center", fontsize=8, color="#444")
    arrow(ax, (55, 68), (58, 68))

    # 3a. teacher
    panel(ax, 58, 8, 78, 32, "3a. Teacher: ENSmc ensemble (frozen)", "#EDEDED", "#666", ls="--")
    for i, (n, p) in enumerate((("A", "121,099"), ("B", "139,606"), ("C", "151,606"))):
        box(ax, 60, 13 + i * 7, 22, 5.5, f"Student {n}   {p}", fs=8.5)
        arrow(ax, (82, 15.75 + i * 7), (86, 20 + i * 2), lw=1.1)
    box(ax, 86, 17, 20, 10, "Average of the\n3 softmax outputs", fs=8.8)
    arrow(ax, (106, 22), (110, 22))
    box(ax, 110, 17, 23, 10, "Teacher logits z_t\n(5 stages per epoch)", fc="#F8DCC0", fs=8.8, weight="bold")
    ax.text(97, 36.5, "EEG only  ·  412,311 parameters  ·  run once, outputs cached", ha="center", fontsize=8.8,
            style="italic", color="#333")

    # 3b. student
    panel(ax, 58, 46, 78, 44, "3b. Student: N6adv (trained, then deployed)", "#E2EFDA", "#548235")
    enc = [(52, 7, "EEG encoder\nmulti-scale CNN", "#CADCF2"), (61, 7, "EOG encoder\nCNN + 8 EOG features", "#D9E8CB"),
           (70, 7, "Time-frequency encoder\nSTFT + CNN", "#F5EDC1"), (79, 6, "Spectral: Linear 34 → 64", "#F4CCD3")]
    for i, (y, h, t, c) in enumerate(enc):
        box(ax, 60, y, 25, h, t, fc=c, fs=8.5)
        arrow(ax, (85, y + h / 2), (88, 62 + i * 4), lw=1.1)
    box(ax, 88, 58, 14, 20, "Gated\nfusion\n\ngates start\nclosed\n(bias −4)", fc="#DCD3EA", fs=8.6)
    arrow(ax, (102, 68), (105, 68))
    box(ax, 105, 58, 14, 20, "Transformer\n× 2 layers\n\ntemporal\ncontext over\n256 epochs", fc="#FCE4D6", fs=8.6)
    arrow(ax, (119, 68), (122, 68))
    box(ax, 122, 58, 12, 20, "Linear\n64 → 5\n\nlogits z_s\nper epoch", fc="#F8DCC0", fs=8.6, weight="bold")
    ax.text(97, 87.6, "EEG + EOG  ·  175,599 parameters (43% of the teacher)", ha="center", fontsize=8.8,
            style="italic", color="#333")
    caption(ax, 97, "Learns from the true labels and the teacher;\nonly the student goes to the device")

    # 4. knowledge distillation
    panel(ax, 140, 8, 64, 32, "4. Knowledge Distillation", "#FBE5D6", "#C55A11")
    box(ax, 142, 13, 25, 7, "Soft targets\np_T = softmax(z_t / T)", fs=8.6)
    box(ax, 142, 25, 25, 7, "Student soft output\np_S = softmax(z_s / T)", fs=8.6)
    box(ax, 171, 12.5, 31, 21, "KD loss\n\nL = α·CE(z_s, y)\n+ (1 − α)·T²·KL(p_T ‖ p_S)\n+ 0.1·L_trans\n\nα = 0.5,   T = 3",
        fs=9, weight="bold")
    arrow(ax, (133, 22), (142, 16.5))
    arrow(ax, (130, 58), (142, 30.5))
    ax.text(139.5, 45, "z_s", fontsize=8.5, color="#444", style="italic")
    arrow(ax, (167, 16.5), (171, 20)); arrow(ax, (167, 28.5), (171, 26))
    ax.text(172, 37.3, "CE: true labels  ·  KL: copy the teacher's soft view  ·  L_trans: stage changes", ha="center",
            fontsize=8.2, color="#333", style="italic")

    # 5. calibration + evidence packet
    panel(ax, 140, 46, 42, 44, "5. Calibration + Evidence Packet", "#DEEBF7", "#2E75B6")
    box(ax, 142, 52, 38, 10, "Temperature scaling\nT_c = 1.0021 → calibrated probabilities", fs=8.6)
    arrow(ax, (134, 61), (142, 58))
    arrow(ax, (161, 62), (161, 65), lw=1.2)
    box(ax, 142, 65, 38, 23, "", fc="white")
    ax.text(161, 67.6, "Evidence Packet (one JSON per night)", ha="center", va="center", fontsize=8.8, weight="bold")
    ax.text(143.5, 77.8, "• hypnogram + probabilities\n• night confidence tier (entropy)\n• N1 confidence flag\n"
            "• 24 sleep metrics, each tagged\n   robust / fragile / unreliable\n• 19 evidence items with IDs",
            ha="left", va="center", fontsize=8.4, linespacing=1.3)
    caption(ax, 161, "Probabilities you can trust,\npacked as checkable evidence")

    # 6. SLM
    panel(ax, 186, 46, 34, 44, "6. Small Language Model", "#E2F0EF", "#2E8B87")
    box(ax, 188, 53, 30, 12, "Llama-3.2-3B student\nLoRA fine-tuned\n4-bit GGUF, offline", fs=8.6, weight="bold")
    box(ax, 188, 68, 30, 8, "Grammar-constrained\ntemperature 0", fs=8.6)
    box(ax, 188, 79, 30, 8, "Claims (JSON): each\ncites an evidence ID", fc="#CADCF2", fs=8.6)
    arrow(ax, (203, 65), (203, 68), lw=1.2); arrow(ax, (203, 76), (203, 79), lw=1.2)
    arrow(ax, (180, 72), (188, 60))
    caption(ax, 203, "Writes the report\nas citable claims")

    # 7. verifier
    panel(ax, 224, 46, 38, 44, "7. Deterministic Verifier", "#F2F2F2", "#7F7F7F")
    box(ax, 226, 53, 34, 7, "Layer 1: structure check", fc="#D5ECD7", fs=8.6)
    box(ax, 226, 62.5, 34, 7, "Layer 2: every number = packet", fc="#D5ECD7", fs=8.6)
    arrow(ax, (243, 60), (243, 62.5), lw=1.2); arrow(ax, (243, 69.5), (243, 72), lw=1.2)
    for y, t, c in ((72, "0 violations → VERIFIED_REPORT", "#D5ECD7"), (77.5, "violation → FALLBACK_SUMMARY", "#F8E4E4"),
                    (83, "untraced number → REVIEW_ONLY", "#F8DCC0")):
        box(ax, 226, y, 34, 4.6, t, fc=c, fs=8.2)
    arrow(ax, (218, 83), (226, 56.5))
    caption(ax, 243, "No unchecked number\nreaches the user")

    # 8. output
    panel(ax, 210, 8, 52, 32, "8. Output on the Device", "#EDE7F6", "#7030A0")
    box(ax, 213, 14, 46, 19, "One-page sleep report\n\nhypnogram  ·  confidence card\nverified narrative\n(clinician view + simple view)",
        fs=9, weight="bold")
    arrow(ax, (243, 46), (243, 40))

    # agent strip
    box(ax, 27, 100, 235, 5.5, "Deterministic agent (offline, on the device) runs steps 2 → 8 in order:  hash-pinned to the "
        "release  ·  resumes after a crash  ·  bad input → REJECTED", fc="#EDE7F6", ec="#7030A0", fs=9.5, weight="bold")

    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
