"""Architecture diagrams for the S7 final report (release v2). Every box states what the v2 code does:
staging model = ADDING EOG VERSION - ADVANCED STUDENT/kaggle_train_student_advanced.py (AdvancedStudent, arm "full"),
agent = agentic/, report gate = report/, release = agentic/config/releases/v2.yaml.

    python "S7 FINAL REPORT/make_architecture_figures.py"      -> S7 FINAL REPORT/images/A*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "images"
C = {"in": "#E8EEF8", "model": "#DCE8F7", "pkt": "#FBF1DE", "slm": "#F6E6D3", "gate": "#E3F2E6",
     "agent": "#EFEDE8", "bad": "#F8E1E1", "warn": "#FFF4CC", "grey": "#F2F2F2", "edge": "#E6E1F2"}


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 10); ax.set_ylim(0, h * 10); ax.axis("off")
    ax.text(w * 5, h * 10 - 3, title, ha="center", va="top", fontsize=15, weight="bold")
    return fig, ax


def box(ax, x, y, w, h, title, body="", color="grey", dashed=False, fs=9.5, tfs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=C.get(color, color), ec="#555", lw=1.3, ls="--" if dashed else "-"))
    if body:
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=tfs, weight="bold")
        ax.text(x + w / 2, y + h - 2.2 - tfs * 0.42, body, ha="center", va="top", fontsize=fs, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tfs, weight="bold")


def arrow(ax, a, b, text="", color="#444", ls="-", rad=0.0, fs=8.5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.5, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 1.2, text, ha="center", va="bottom", fontsize=fs,
                color=color, style="italic")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- A1 system architecture
def a1():
    fig, ax = canvas(16, 10.5, "System architecture, release v2 (offline; every runtime file hash-pinned)")
    # layer frames
    for y, h, label in ((72, 24, "OFFLINE: training and fitting (Kaggle T4 GPU, run once)"),
                        (49, 19, "RELEASE v2: pinned artefacts (agentic/config/releases/v2.yaml, manifest 4973f498...)"),
                        (8, 37, "RUNTIME: deterministic agent on the device (CPU only, no network)")):
        ax.add_patch(FancyBboxPatch((2, y), 156, h, boxstyle="round,pad=0.3,rounding_size=1", fc="none",
                                    ec="#999", lw=1, ls=(0, (4, 3))))
        ax.text(4, y + h - 1.2, label, fontsize=10, weight="bold", color="#555", va="top")
    # offline
    box(ax, 5, 74, 28, 17, "Sleep-EDFx data", "197 nights / 100 subjects\nsubject-level split 69/16/15\nlabels corrected (5-7%)", "in")
    box(ax, 38, 74, 36, 17, "Staging training (KD)", "ensemble teacher ENSmc (412,311)\n-> v1 N4kd (139,606)\n-> v2 N6adv, EEG+EOG (175,599)", "model")
    box(ax, 79, 74, 36, 17, "Packet tables (validation)", "temperature, sequence decoder,\nper-stage reliability, night-confidence\ntiers, metric reliability, N1 flag", "pkt")
    box(ax, 120, 74, 35, 17, "Report model (LoRA SFT)", "reference targets (123 nights)\nLlama-3.2-3B + LoRA r16\n-> GGUF Q4_K_M (2.02 GB)", "slm")
    arrow(ax, (33, 82), (38, 82)); arrow(ax, (74, 82), (79, 82)); arrow(ax, (115, 82), (120, 82))
    # release
    rel = [("staging checkpoint", "student_best.pt"), ("packet builder + tables", "work_n6adv_packets/"),
           ("report model + grammar", "GGUF + claims.gbnf"), ("chat template", "pinned to training date"),
           ("verifier + renderer", "report/ (30 rules)"), ("runtime", "llama-completion b10927")]
    rx = [16.5 + i * 25.5 for i in range(6)]                 # release box centres
    for i, (t, b) in enumerate(rel):
        box(ax, rx[i] - 11.5, 51, 23, 11, t, b, "gate", fs=8.2, tfs=8.6)
    for a, b in ((56, rx[0]), (97, rx[1]), (137.5, rx[2]), (137.5, rx[3])):   # offline product -> pinned file
        arrow(ax, (a, 74), (b, 62.5), color="#888")
    # runtime pipeline
    steps = [("EDF in", "EEG Fpz-Cz\n+ EOG", "in"), ("Ingest + QC", "channels, length,\nflat/noise (Q01/Q03)", "agent"),
             ("Preprocess", "100 Hz, 30 s epochs,\n34 spectral feats", "agent"), ("Staging", "pass 1 -> night span\n-> pass 2", "model"),
             ("Evidence packet", "schema 1.3 JSON,\n19 evidence items", "pkt"), ("Report SLM", "grammar, temp 0,\nclaims cite fields", "slm"),
             ("Verifier gate", "L1 structure +\nL2 policy", "gate"), ("Render + deliver", "one page per\nnight (outbox)", "gate")]
    for i, (t, b, c) in enumerate(steps):
        x = 5 + i * 19.2
        box(ax, x, 25, 16.5, 13, t, b, c, fs=8, tfs=9.5)
        if i:
            arrow(ax, (x - 2.7, 31.5), (x, 31.5))
    sx = [5 + i * 19.2 + 8.25 for i in range(8)]             # runtime step centres
    for r, s in ((0, 3), (1, 4), (2, 5), (3, 5), (4, 6), (4, 7), (5, 5)):       # pinned file -> step that loads it
        arrow(ax, (rx[r], 51), (sx[s], 38), color="#888", ls="--")
    box(ax, 5, 10, 70, 10, "Agent core (workflow only, never clinical content)",
        "run id = hash(input, policy, release) - resume from step markers - retries - RAM/heat wait -\nbounded walk - trace.jsonl - one delivery per night - review queue", "agent", dashed=True, fs=8.5, tfs=10)
    box(ax, 80, 10, 37, 10, "Outcomes", "VERIFIED_REPORT - FALLBACK_SUMMARY\nREVIEW_ONLY - REJECTED", "warn", fs=8.5, tfs=10)
    box(ax, 121, 10, 34, 10, "Interfaces", "local web console (python -m ui),\nSimple view, Evidence page", "grey", fs=8.5, tfs=10)
    arrow(ax, (140, 25), (98, 20), color="#888")
    save(fig, "A1_system_architecture.png")


# ---------------------------------------------------------------- A2 staging model v2
def a2():
    fig, ax = canvas(17, 10, "Release v2 staging student (N6adv 'full'): 175,599 parameters, gates initialised closed")
    ax.text(85, 91, "Backbone = v1 N4kd (139,606) unchanged. With the new paths closed, the untrained v2 computes exactly v1's function.\n"
            "Modality dropout 0.25 trains it to work without EOG (test kappa 0.744 without EOG vs 0.756 with it).",
            ha="center", va="top", fontsize=9, style="italic")
    box(ax, 2, 64, 22, 14, "EEG Fpz-Cz", "256 epochs x 3,000\n(30 s at 100 Hz)", "in")
    box(ax, 2, 41, 22, 14, "Horizontal EOG", "256 x 3,000\n+ 8 EOG features", "in")
    box(ax, 2, 4, 22, 14, "34 spectral features", "DWT db4 (24) +\nSTFT bands (10)", "in")
    box(ax, 29, 62, 44, 19, "Multi-scale epoch encoder (v1)", "fine: Conv k50 s6 (24) -> pool 8 -> Conv k8 (48)\ncoarse: Conv k200 s25 (24) -> pool 4 -> Conv k6 (48)\n"
        "4 tokens -> Linear 96->64 -> attention pool\nreceptive field 8.75 s", "model", fs=8.3)
    box(ax, 29, 40, 44, 16, "EOG encoder (6,768)", "saccade: Conv k50 s6 -> Conv k8\nslow eye movements: Conv k400 s50 -> Conv k5\n+ 8 features -> 32-d", "model", fs=8.3)
    box(ax, 29, 21, 44, 14, "Time-frequency encoder (7,848)", "STFT 2 s window, 1 s hop, 0.5-30 Hz\n2 x Conv over time -> 32-d", "model", fs=8.3)
    box(ax, 29, 3, 44, 12, "Spectral projection (v1)", "Linear 34 -> 64", "model", fs=8.5)
    box(ax, 79, 62, 24, 19, "+ gated EOG (8,320)", "h += sigmoid(G[h,e]) * P e\nP = 0, gate bias -4\nat initialisation", "gate", fs=8.2, tfs=10)
    box(ax, 108, 62, 24, 19, "+ gated TF (8,320)", "same closed-gate\nresidual on h", "gate", fs=8.2, tfs=10)
    box(ax, 137, 62, 30, 19, "Fusion (v1)", "concat[h, spectral]\nLinear 128 -> 64\nLayerNorm, GELU", "model", fs=8.5)
    box(ax, 137, 40, 30, 16, "Local context (4,672)", "depthwise Conv k7 (+/-3 epochs)\nzero-initialised residual", "gate", fs=8.3)
    box(ax, 137, 19, 30, 16, "Transformer encoder (v1)", "sinusoidal positions, 2 layers\nd 64, 2 heads, FF 256, pre-norm", "model", fs=8.3)
    box(ax, 84, 3, 70, 12, "Heads", "Linear 64 -> 5 per epoch (W, N1, N2, N3, REM) + transition head (65, training only)", "pkt", fs=8.5)
    arrow(ax, (24, 71), (29, 71)); arrow(ax, (24, 48), (29, 48)); arrow(ax, (24, 11), (29, 9))
    arrow(ax, (24, 66), (29, 30), color="#888")
    arrow(ax, (73, 71.5), (79, 71.5), text="h"); arrow(ax, (103, 71.5), (108, 71.5)); arrow(ax, (132, 71.5), (137, 71.5))
    arrow(ax, (73, 50), (91, 62), color="#888", text="e (32-d)"); arrow(ax, (73, 30), (120, 62), color="#888")
    arrow(ax, (73, 9), (152, 62), color="#888")
    arrow(ax, (152, 62), (152, 56)); arrow(ax, (152, 40), (152, 35)); arrow(ax, (152, 19), (140, 15))
    save(fig, "A2_staging_model_v2.png")


# ---------------------------------------------------------------- A3 report generation + verifier
def a3():
    fig, ax = canvas(16, 7.5, "Verified report generation: the release gate sits before any text is shown")
    box(ax, 1, 42, 21, 20, "Evidence packet", "schema 1.3\n24 metrics + reliability\nnight-confidence tier\n19 evidence IDs", "pkt", fs=8.5)
    box(ax, 26, 42, 21, 20, "Prompt P1", "pinned chat template\n(training date)\npacket serialised", "agent", fs=8.5)
    box(ax, 51, 42, 29, 20, "Report SLM (v2)", "Llama-3.2-3B-Student, Q4_K_M\nllama-completion b10927\nGBNF grammar (claims only)\ntemp 0, seed 0, ctx 12,288", "slm", fs=8.5)
    box(ax, 85, 50, 34, 15, "Layer 1: structure", "valid JSON, known claim types,\nfields, evidence IDs, enums,\nno duplicates (closed world)", "gate", fs=8.3)
    box(ax, 85, 28, 34, 18, "Layer 2: policy (packet)", "cited value == packet value\nunreliable values hedged\nno unsafe item without flag\n5 mandatory items for the tier", "gate", fs=8.3)
    box(ax, 128, 46, 30, 16, "VERIFIED_REPORT", "render page; V-HTML: every\nnumber traces to the packet\n(else REVIEW_ONLY V01)", "gate", fs=8.5)
    box(ax, 128, 20, 30, 18, "FALLBACK_SUMMARY", "F02 crash / timeout (retried)\nF03 any violation:\ntext never shown;\npage = packet numbers only", "bad", fs=8.5)
    arrow(ax, (22, 52), (26, 52)); arrow(ax, (47, 52), (51, 52)); arrow(ax, (80, 55), (85, 57))
    arrow(ax, (102, 50), (102, 46)); arrow(ax, (119, 40), (128, 54)); arrow(ax, (119, 34), (128, 29))
    ax.text(124, 49, "0 violations", fontsize=8.5, color="#2b7a3d", style="italic", ha="center")
    ax.text(124, 25.5, "any violation", fontsize=8.5, color="#a33", style="italic", ha="center")
    ax.text(80, 12, "Test (29 locked nights): 28 verified, 1 blocked (invented REM latency -1.0 where the packet held None), "
            "0 flawed reports released.\nAblation: untrained Llama-3.2-3B releases 31/31 flawed reports without the gate, 0/31 with it.",
            ha="center", fontsize=9.5, style="italic")
    save(fig, "A3_report_and_verifier.png")


# ---------------------------------------------------------------- A4 agent Path B
def a4():
    fig, ax = canvas(16, 8.5, "Agent workflow, Path B (raw EDF, no expert hypnogram): each step returns the next decision")
    steps = [("01 ingest", "R01 unreadable\nR03 < 3 h\nW05 channel missing"), ("02 preprocess", "resample 100 Hz\n30 s epochs\n34 features"),
             ("03 stage pass 1", "staging adapter\non the whole file"), ("04 study period", "sleep span from\npass 1 (R04 none)"),
             ("05 quality", "Q01 flat, Q03 line\nnoise -> review\nW02 / W10 warn"), ("06 stage pass 2", "stage the span only\nW08 span moved"),
             ("07 build packet", "release's builder\nR10 schema"), ("08 report", "wait RAM/heat (F01)\ngenerate, retry (F02)\nverify (F03)"),
             ("09 V-HTML", "render, check\nprovenance (V01)"), ("10 deliver", "exactly once\noutbox + trace")]
    for i, (t, b) in enumerate(steps):
        r, c = divmod(i, 5)
        x, y = 4 + c * 31, 50 - r * 30
        col = "model" if "stage" in t else "gate" if t[:2] in ("09", "10") else "slm" if t[:2] == "08" else "pkt" if t[:2] == "07" else "agent"
        box(ax, x, y, 26, 20, t, b, col, fs=8.3, tfs=10)
        if c:
            arrow(ax, (x - 5, y + 10), (x, y + 10))
    arrow(ax, (134, 50), (17, 40), rad=0.0, color="#444")
    box(ax, 4, 2, 72, 13, "Recovery and identity", "run id = hash(EDF + crop, policy, release manifest); finished steps replay from DONE markers;\n"
        "an interrupted step is wiped and re-run; bounded walk; every transition in trace.jsonl", "agent", dashed=True, fs=8.3, tfs=10)
    box(ax, 82, 2, 74, 13, "Outcomes", "VERIFIED_REPORT | FALLBACK_SUMMARY (F01-F04) | REVIEW_ONLY (R04, Q01, Q03, W05, I01, V01)\n"
        "| REJECTED (R01, R03, R10); warnings W02/W08/W10 deliver and copy to the review queue", "warn", fs=8.3, tfs=10)
    save(fig, "A4_agent_workflow.png")


# ---------------------------------------------------------------- A5 training / KD pipeline
def a5():
    fig, ax = canvas(16, 6.5, "How the staging model was built: from a leaky baseline to the v2 student")
    items = [("Inherited model", "649,229 params\nkappa 0.6663 (leaky)\n0.5066 unseen subject", "bad"),
             ("Subject-level split", "69 / 16 / 15 subjects\ntest locked, reads logged", "in"),
             ("Teacher ladder", "E0 -> E1a -> E1b\nsingle-teacher KD hurt\n(control 0.6449 > 0.6134)", "warn"),
             ("Receptive field fix", "0.25 s -> 8.75 s\nmulti-scale encoder", "model"),
             ("Ensemble teacher", "3 diverse students\n412,311 params\nkappa 0.714", "model"),
             ("v1 student N4kd", "139,606, EEG only\nKD alpha 0.5, T 3", "model"),
             ("Label correction", "drop-then-trim bug\n5-7% labels shifted", "warn"),
             ("v2 student N6adv", "175,599, EEG + EOG\nwarm start, gates\ntest kappa 0.756", "gate")]
    for i, (t, b, c) in enumerate(items):
        r, k = divmod(i, 4)
        x, y = 3 + k * 39.5, 38 - r * 19
        box(ax, x, y, 34, 15, t, b, c, fs=8.5, tfs=10)
        if k:
            arrow(ax, (x - 5.5, y + 7.5), (x, y + 7.5))
    arrow(ax, (140, 38), (20, 34), color="#444")
    ax.text(80, 9, "KD loss  L = alpha CE(z_s, y) + (1 - alpha) T^2 KL(softmax(z_t/T) || softmax(z_s/T)),  alpha = 0.5, T = 3.\n"
            "Mechanism: a teacher helps only if it beats the student's own undistilled performance; an ensemble of comparable, diverse students does.",
            ha="center", fontsize=9.5, style="italic")
    save(fig, "A5_training_pipeline.png")


# ---------------------------------------------------------------- A6 edge deployment + simulation
def a6():
    fig, ax = canvas(16, 7.5, "Edge target and simulation ladder (Raspberry Pi 5, 16 GB)")
    box(ax, 3, 30, 50, 34, "Raspberry Pi 5 (designed)", "4x Cortex-A76 2.4 GHz, 16 GB LPDDR4X, NVMe\nRaspberry Pi OS Lite 64-bit, systemd service\n"
        "nftables: all outbound traffic dropped\nnightly: EDF in -> agent -> one page by morning\n\nbudget: ~6-9 min and ~6-8 GB per night\n"
        "(bandwidth-bound: <= 7.43 tokens/s ceiling)", "edge", fs=8.8)
    rows = [("L0 analytical", "time + memory budget", "done", "gate"),
            ("L1 Docker x86\n4 CPU / 16 GB / no network", "network off, no secrets; staging 31/31 identical;\n31/31 reports verified, text identical;\n151 s and 5.2 GB per report (x86)", "done", "gate"),
            ("L2 ARM64 under QEMU", "E1: 8 SC nights, hypnograms identical,\nmax |dp| 7.7e-7 (bar 1e-4)", "met (subset)", "gate"),
            ("L3 fault replay", "14/14 scenarios, I1-I8", "done on x86", "gate"),
            ("L5 real Pi 5", "speed, heat, power, 7-night soak", "needs hardware", "bad")]
    ax.text(60, 66.5, "level", fontsize=9, weight="bold"); ax.text(106, 66.5, "what it checks / result", fontsize=9, weight="bold")
    ax.text(157, 66.5, "status", fontsize=9, weight="bold", ha="right")
    for i, (t, b, s, c) in enumerate(rows):
        y = 55 - i * 11
        box(ax, 60, y, 43, 9.5, t, "", c, tfs=8.8)
        ax.text(106, y + 4.75, b, fontsize=8.3, va="center")
        ax.text(157, y + 4.75, s, fontsize=8.5, ha="right", va="center", weight="bold")
    arrow(ax, (53, 47), (60, 47))
    save(fig, "A6_edge_and_simulation.png")


if __name__ == "__main__":
    for f in (a1, a2, a3, a4, a5, a6):
        f()
