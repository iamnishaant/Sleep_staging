"""Generate every data-driven figure in FIGURES.md for the S7 report / journal.

    python figures/make_figures.py           all figures -> figures/output/*.png (300 dpi)
    python figures/make_figures.py F3 F15    only the listed figures

Data policy: anything with a committed JSON is READ LIVE from `distillation/results/`,
`kaggle/working/b1/` and `ROADMAP R2 - VERIFIER ABLATION/results/`. Historical numbers that
exist only as tables in PROJECT_REPORT.md are hardcoded here with the source named in a comment,
so a figure cannot silently drift from a changed artefact without the change being visible here.

F2 (leak schematic), F4 (architecture) and F21 (evidence-packet flow) are diagrams, not plots:
draw them by hand in your figure tool, they are not generated here.
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RES = REPO / "distillation" / "results"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

STAGES = ["W", "N1", "N2", "N3", "REM"]
C = {"W": "#4472c4", "N1": "#c00000", "N2": "#70ad47", "N3": "#7b5ba7", "REM": "#ed7d31"}
ACCENT, GREY, GREEN, RED = "#0366d6", "#9aa0a6", "#2e7d32", "#c62828"
PARSED = []


def load(path, missing=""):
    p = Path(path)
    if not p.exists():
        print(f"    !! missing {p} {'- ' + missing if missing else ''}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def finish(fig, name, suffixed=False):
    p = OUT / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    PARSED.append(name)
    print(f"    wrote {p.relative_to(REPO)}")


def source(fig, txt):
    fig.text(0.01, 0.01, txt, fontsize=7, color=GREY, style="italic")


# ---------------------------------------------------------------- F1 subject-level split
def f1():
    splits = [("Train", 54, 15, 83), ("Validation", 12, 4, 19), ("Test", 12, 3, 15)]
    fig, ax = plt.subplots(figsize=(6, 3))  # subjects per split, SC/ST
    for i, (name, sc, st, _) in enumerate(splits):
        ax.barh(i, sc, color=ACCENT, label="SC (sleep cassette)" if i == 0 else None)
        ax.barh(i, st, left=sc, color="#f4a63a", label="ST (sleep telemetry)" if i == 0 else None)
        ax.text(sc / 2, i, str(sc), va="center", ha="center", color="w", fontweight="bold")
        ax.text(sc + st / 2, i, str(st), va="center", ha="center", color="w", fontweight="bold")
    ax.set_yticks(range(3), [s[0] for s in splits])
    ax.set_xlabel("Subjects"); ax.set_xlim(0, 86)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("Subject-level, cohort-stratified split (69 / 16 / 15 subjects)")
    source(fig, "distillation/make_splits.py — test split loaded exactly once per model.")
    finish(fig, "F1_subject_split")


# ---------------------------------------------------------------- F3 exposure gradient
def f3():
    # PROJECT_REPORT §3 (eval_teacher.json exposure breakdown): the pivotal measurement
    kap = [0.8058, 0.6797, 0.5066]
    f1s = {"N3": [0.889, 0.837, 0.000], "REM": [0.876, 0.718, 0.184]}
    labels = ["Same recording\n(trained on)", "Same subject,\ndifferent night",
              "Genuinely unseen\nsubject"]
    published = 0.6663
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    bars = a1.bar(labels, kap, color=[GREY, "#f4a63a", RED], width=0.55)
    a1.axhline(published, color=ACCENT, ls="--", lw=1.5)
    a1.text(2.4, published, f" published κ\n {published:.4f}", color=ACCENT, fontsize=9, va="bottom")
    a1.bar_label(bars, fmt="%.3f"); a1.set_ylabel("Cohen's κ"); a1.set_ylim(0, 1)
    a1.set_title("κ by exposure level")
    xs = np.arange(3); w = 0.36
    for j, (stg, vals) in enumerate(f1s.items()):
        a2.bar(xs + (j - 0.5) * w, vals, w, color=C[stg], label=stg)
        a2.bar_label(a2.containers[j], fmt="%.2f", fontsize=8)
    a2.set_xticks(xs, labels); a2.set_ylabel("F1"); a2.set_ylim(0, 1)
    a2.legend(frameon=False); a2.set_title("Stage F1 by exposure level")
    source(fig, "PROJECT_REPORT.md §3 / eval_teacher.json. Published figure was a blend of the first two bars.")
    finish(fig, "F3_exposure_gradient")


# ---------------------------------------------------------------- F5 receptive field
def f5():
    ab = load(RES / "multiscale_encoder_ablation.json", "fig falls back to §6b constants")
    if ab:
        enc = ab["the_defect"]; rf = ab["design"]["measured_receptive_field"]
    fields = [(enc["student_encoder_atrous_k7_dil_1_4"]["seconds"] if ab else 0.25, "Old student\n(atrous)", RED),
              (enc["teacher_encoder_atrous_k7_dil_1_2_4_8"]["seconds"] if ab else 0.49, "Teacher\n(atrous)", "#f4a63a"),
              (rf["seconds"] if ab else 8.75, "Rebuilt\n(multi-scale)", GREEN)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axvspan(0.5, 2.0, color="#f4a63a", alpha=0.12)
    ax.axvspan(1.0, 3.0, color="#ed7d31", alpha=0.12)
    ax.barh(range(3), [v for v, _, _ in fields], height=0.5, color=[c for _, _, c in fields], zorder=3)
    for y, (v, lab, _) in enumerate(fields):
        ax.text(v + 0.08, y, f"{v:g} s", va="center", zorder=4)
    ax.text(1.25, 2.72, "spindle / K-complex / slow wave (0.5–2 s)", fontsize=8, color="#8a6d1e", ha="center")
    ax.text(2.45, -0.55, "sawtooth wave, REM (1–3 s)", fontsize=8, color="#9c5518", ha="center")
    ax.set_yticks(range(3), [lab for _, lab, _ in fields])
    ax.set_xlabel("Receptive field / event duration (seconds)"); ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.75, 3.0)
    ax.set_title("The encoder must contain the events it classifies")
    source(fig, "multiscale_encoder_ablation.json (field measured by backpropagation, §6b).")
    finish(fig, "F5_receptive_field")


# ---------------------------------------------------------------- F6 teacher ladder
def f6():
    # §4 result table + "Where the gain came from"; PROJECT_REPORT.md
    kappa = [0.5122, 0.5592, 0.6055]
    pc = {"W": [0.807, 0.819, 0.821], "N1": [0.339, 0.453, 0.449],
          "N2": [0.622, 0.554, 0.756], "N3": [0.651, 0.583, 0.667], "REM": [0.612, 0.721, 0.745]}
    names = ["E0\n(baseline)", "E1a\n(focal, defective)", "E1b\n(focal, fixed)"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    bars = a1.bar(names, kappa, color=[GREY, "#f4a63a", GREEN], width=0.55)
    a1.bar_label(bars, fmt="%.4f"); a1.set_ylim(0, 0.75); a1.set_ylabel("Cohen's κ")
    a1.set_title("Teacher κ (identical architecture)")
    xs = np.arange(5); w = 0.9 / 3
    tcol = [GREY, "#f4a63a", GREEN]
    for j, tname in enumerate(names):
        a2.bar(xs + (j - 1) * w, [pc[s][j] for s in STAGES], w * 0.95,
               color=tcol[j], label=tname.split("\n")[0])
    a2.set_xticks(xs, STAGES); a2.set_ylabel("F1"); a2.set_ylim(0, 1); a2.legend(ncol=5, frameon=False)
    a2.set_title("Validation F1 per stage — the fix is N2, not N1", fontsize=10)
    source(fig, "PROJECT_REPORT.md §4 (E1a computed the focal modulator on the weighted CE: α-power bug).")
    finish(fig, "F6_teacher_ladder")


# ---------------------------------------------------------------- F7 encoder ablation
def f7():
    ab = load(RES / "multiscale_encoder_ablation.json", "cannot draw")
    if ab is None:
        return
    prog = ab["branch_liveness"]["progression"]
    names = [p["model"].replace("student_", "") for p in prog]
    pct = [p["pct_predictions_changed"] for p in prog]
    mac = [ab["performance"].get("best", {}).get("macro_f1")] * 1
    old_mf1 = next(c["macro_f1"] for c in ab["performance"]["comparisons_same_validation_split"].values()
                   if c.get("note", "").startswith("same EEG_SCALE, old"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.barh(names[::-1], pct[::-1], color=[RED, "#f4a63a", GREEN], height=0.55)
    for y, p in enumerate(pct[::-1]):
        a1.text(p + 1, y, f"{p:.0f}%", va="center")
    a1.set_xlim(0, 85); a1.set_xlabel("% of predictions changed when raw EEG is zeroed")
    a1.set_title("Is the waveform branch alive?")
    a2.bar(["old encoder\n(same scale)", "multi-scale\n(rebuilt)"], [old_mf1, ab["performance"]["best"]["macro_f1"]],
           color=[GREY, GREEN], width=0.5)
    a2.bar_label(a2.containers[0], fmt="%.4f"); a2.set_ylim(0, 0.8)
    a2.set_ylabel("Best validation macro-F1"); a2.set_title("Same data / schedule; only the encoder changed")
    source(fig, "multiscale_encoder_ablation.json (validation only; test split never loaded).")
    finish(fig, "F7_encoder_ablation")


# ---------------------------------------------------------------- F8 channel ablation
def f8():
    # §6c: EOG + Pz-Oz, the near-textbook trap
    mk = [("test", 0.0028, GREEN), ("validation", -0.0122, RED)]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs, dks = [m[0] for m in mk], [m[1] for m in mk]
    ax.bar(xs, dks, 0.5, color=[m[2] for m in mk])
    ax.axhline(0, color="#444", lw=1)
    ax.bar_label(ax.containers[0], fmt="%+.4f")
    ax.set_ylabel("Δ κ vs single-channel student"); ax.set_ylim(-0.02, 0.012)
    ax.set_title("EOG + Pz-Oz: a gain on test that loses on validation")
    ax.text(0.5, 0.05, "paired Wilcoxon p = 0.865\n(better on 15/29 recordings)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#444")
    source(fig, "PROJECT_REPORT.md §6c.")
    finish(fig, "F8_channel_ablation")


# ---------------------------------------------------------------- F9 cohort transfer
def f9():
    ct = load(RES / "cohort_transfer.json", "cannot draw")
    if ct is None:
        return
    cv_f1 = [ct["decomposition"]["f1_change_vs_cv"][s] and
             (ct["per_class_f1_ST"][s] - ct["decomposition"]["f1_change_vs_cv"][s]) for s in STAGES]
    st_f1 = [ct["per_class_f1_ST"][s] for s in STAGES]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    xs = np.arange(5); w = 0.36
    ax.bar(xs - w / 2, cv_f1, w, color=ACCENT, label="CV (same-cohort)")
    ax.bar(xs + w / 2, st_f1, w, color="#f4a63a", label="ST cohort (all 22 subjects held out)")
    ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=8); ax.bar_label(ax.containers[1], fmt="%.2f", fontsize=8)
    ax.set_xticks(xs, STAGES); ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.legend(frameon=False)
    ax.set_title("SC→ST transfer: the gap is one stage, not the sleep stages")
    source(fig, "cohort_transfer.json (κ 0.738 → 0.552; Δ wake F1 −0.276 is the lone large drop).")
    finish(fig, "F9_cohort_transfer")


# ---------------------------------------------------------------- F10 prior + amplitude negatives
def f10():
    pa = load(RES / "prior_adaptation.json", "panel (a) missing")
    pn = load(RES / "cohort_transfer_pernorm.json", "panel (b) missing")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    if pa:
        v = pa["variants"]
        names = ["uncorrected", "unsupervised\nEM", "oracle priors\n(true p(y))"]
        ks = [v["uncorrected"]["kappa"], v["em_adapted"]["kappa"], v["oracle_prior"]["kappa"]]
        bars = a1.bar(names, ks, color=[GREY, RED, GREEN], width=0.55)
        a1.axhline(v["uncorrected"]["kappa"], color=GREY, ls=":", lw=1)
        a1.bar_label(bars, fmt="%.4f"); a1.set_ylim(0.4, 0.6); a1.set_ylabel("κ on held-out ST cohort")
        a1.set_title("(a) Prior correction: perfect knowledge buys 1.2%")
    if pn:
        bars = a2.bar(["global scale\n(baseline)", "per-recording\nnormalisation"],
                      [0.5516, pn["kappa_ST"]], color=[GREY, RED], width=0.5)
        a2.bar_label(bars, fmt="%.4f"); a2.set_ylim(0.4, 0.6); a2.set_ylabel("κ on held-out ST cohort")
        a2.set_title("(b) Amplitude normalisation: transfer gets worse")
        a2.text(0.5, 0.05, "paired Δ = −0.0175, CI [−0.0318, −0.0033]", transform=a2.transAxes,
                ha="center", fontsize=9, color="#444")
    source(fig, "prior_adaptation.json + cohort_transfer_pernorm.json (both hypotheses excluded, §6h/§6i).")
    finish(fig, "F10_negative_results")


# ---------------------------------------------------------------- F11 KD mechanism
def f11():
    # §6 RQ2: the baseline student computes but never optimises the KD term
    ep = [0, 20, 40, 74]
    kl = [0.196, 3.394, 4.732, 5.802]
    mf1 = [0.3312, 0.6450, 0.6692, 0.6825]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
    a1.plot(ep, kl, "o-", color=RED); a1.set_ylabel("KL(baseline ‖ E0 teacher)")
    a1.set_title("Divergence from the teacher ×30 ↑")
    a2.plot(ep, mf1, "o-", color=GREEN); a2.set_ylabel("validation macro-F1")
    a2.set_title("...while the student gets better ↑", fontsize=10)
    for a in (a1, a2):
        a.set_xlabel("epoch")
    source(fig, "PROJECT_REPORT.md §6 RQ2 (-educated teacher = E0, κ 0.5122 < student 0.6449).")
    finish(fig, "F11_kd_mechanism")


# ---------------------------------------------------------------- F12 penalty vs teacher quality
def f12():
    # §6 RQ2/RQ3 + §6d: penalty vs the teacher's own test κ
    t = [0.5122, 0.6055, 0.7140]
    pen = [-0.0523, -0.0314, 0.0008]
    names = ["E0", "E1b", "ensemble"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.axhline(0, color="#444", lw=1, ls="--")
    for x, y, n in zip(t, pen, names):
        c = GREEN if y > 0 else RED
        ax.scatter(x, y, s=70, color=c, zorder=3)
        ax.annotate(f"{n}\n({y:+.4f})", (x, y), xytext=(0, 10 if y < 0 else -22),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel("Teacher test κ"); ax.set_ylabel("Δκ distilled − matched baseline")
    ax.set_ylim(-0.07, 0.03); ax.set_xlim(0.49, 0.74)
    ax.set_title("The penalty is a function of the teacher–student gap")
    source(fig, "PROJECT_REPORT.md §6 (α = 0.5, T = 3.0; ensemble = 3 students averaged).")
    finish(fig, "F12_penalty_vs_teacher")


# ---------------------------------------------------------------- F13 calibration + REM latency
def f13():
    kd = load(RES / "kd_result.json", "panel (a) missing")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    if kd:
        e = kd["the_result_that_matters"]["calibration_uncalibrated_ECE_15_bins"]
        vals = [e["hard_labels"]["test"], e["distilled"]["test"]]
        bars = a1.bar(["hard labels", "distilled"], vals, 0.5, color=[GREY, GREEN])
        a1.bar_label(bars, fmt="%.4f"); a1.set_ylabel("un-calibrated ECE (test)")
        a1.set_ylim(0, 0.05); a1.set_title("(a) ECE −33%")
        a1.text(0.5, 0.9, "accuracy ties (paired p = 0.24)", transform=a1.transAxes, ha="center", fontsize=9)
    rem = [33.7, 18.3]  # PROJECT_REPORT §6d derived-metric table
    bars = a2.bar(["hard labels", "distilled"], rem, 0.5, color=[GREY, GREEN])
    a2.bar_label(bars, fmt="%.1f"); a2.set_ylabel("REM-latency error (min)")
    a2.set_ylim(0, 40); a2.set_title("(b) REM-latency error −46%")
    source(fig, "kd_result.json + PROJECT_REPORT §6d — what distillation actually transferred.")
    finish(fig, "F13_calibration_transfer")


# ---------------------------------------------------------------- F14 params vs kappa
def f14():
    # §7 complete-results table (held-out test, 15 subjects)
    pts = [
        ("inherited (honest subj.)", 649229, 0.5066, GREY),
        ("E0 teacher", 649229, 0.5122, GREY),
        ("E1a teacher", 649229, 0.5592, GREY),
        ("E1b teacher", 649229, 0.6055, GREY),
        ("student baseline (121K)", 121099, 0.6449, "#f4a63a"),
        ("student multi-scale", 139606, 0.6992, "#f4a63a"),
        ("student + EOG + Pz-Oz", 151606, 0.7020, "#f4a63a"),
        ("ensemble teacher", 410000, 0.7140, GREY),
        ("★ delivered (distilled)", 139606, 0.7001, ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    offs = {  # hand-tuned label offsets to avoid overlaps in the 139–151K cluster
        "delivered (distilled)": (6, 14), "student multi-scale": (-52, -24),
        "student + EOG + Pz-Oz": (34, 8), "student baseline (121K)": (12, 10),
        "ensemble teacher": (-40, 8), "E1b teacher": (0, 10), "E1a teacher": (0, 10),
        "E0 teacher": (18, 10), "inherited (honest subj.)": (-24, -16)}
    for lab, p, k, c in pts:
        star, l = lab[:1] == "★", lab.replace("★ ", "")
        ax.scatter(p, k, s=140 if star else 60, color=c, marker="*" if star else "o", zorder=3)
        ax.annotate(l.replace("(", "\n("), (p, k), xytext=offs.get(l, (0, 9)),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xscale("log"); ax.set_ylim(0.48, 0.75); ax.set_xlim(8e4, 9e5)
    ax.set_xlabel("trainable parameters (log)"); ax.set_ylabel("held-out test κ")
    ax.set_title("4.65× compression, no accuracy loss")
    source(fig, "PROJECT_REPORT.md §7. 95% CI of delivered κ [0.6432, 0.7514] — top three not separable at n=15 subjects.")
    finish(fig, "F14_params_vs_kappa")


# ---------------------------------------------------------------- F15 confusion matrix (delivered)
def f15():
    folder = RES / "probs_student_N4kd"
    if not folder.exists():
        print("    !! probs_student_N4kd missing")
        return
    cm = np.zeros((5, 5), dtype=int)
    for f in sorted(folder.glob("*.npz")):
        d = np.load(f)
        cm += np.bincount(5 * d["labels"].astype(int) + d["probs"].argmax(1),
                          minlength=25).astype(int).reshape(5, 5)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="w" if cmn[i, j] > 0.55 else "#333")
    ax.set_xticks(range(5), STAGES); ax.set_yticks(range(5), STAGES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
    ax.set_title("Delivered student, held-out test (argmax)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    source(fig, f"computed from {len(list(folder.glob('*.npz')))} recordings (probs_student_N4kd/).")
    finish(fig, "F15_confusion_matrix")


# ---------------------------------------------------------------- F16 per-class F1
def f16():
    # delivered = README headline; best teacher = §7 per-class table (E1b)
    delivered = [0.897, 0.429, 0.805, 0.779, 0.778]
    teacher = [0.821, 0.353, 0.756, 0.686, 0.658]
    xs = np.arange(5); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(xs - w / 2, teacher, w, color=GREY, label="best 649K teacher (E1b)")
    ax.bar(xs + w / 2, delivered, w, color=[C[s] for s in STAGES], label="delivered student (139K)")
    ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=8); ax.bar_label(ax.containers[1], fmt="%.2f", fontsize=8)
    ax.set_xticks(xs, STAGES); ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.legend(frameon=False)
    ax.set_title("Every class improves; N1 remains the floor")
    for i, c in enumerate(ax.containers[0]):
        if STAGES[i] == "N1":
            ax.annotate("humans agree on N1\nonly 25–45% of the time", (i, delivered[i]),
                        xytext=(0, -34), textcoords="offset points", fontsize=8, color="#444", ha="center")
    source(fig, "README.md headline per-class F1 + PROJECT_REPORT §7.")
    finish(fig, "F16_per_class_f1")


# ---------------------------------------------------------------- F17 B1 per-fold k
def f17():
    s = load(REPO / "kaggle" / "working" / "b1" / "b1_summary.json", "cannot draw")
    if s is None:
        return
    kf = [f["val"]["kappa"] for f in s["per_fold"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(range(10), kf, 0.6, color=ACCENT)
    ax.axhline(np.mean(kf), color=GREEN, ls="--", lw=1.5, label=f"mean = {np.mean(kf):.3f}")
    ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=8)
    ax.set_xticks(range(10), [f"f{i}" for i in range(10)])
    ax.set_ylim(0.70, 0.85); ax.set_ylabel("Cohen's κ"); ax.legend(frameon=False)
    ax.set_title("Sleep-EDF-78: per-fold κ (subject-wise CV, from scratch)")
    source(fig, f"kaggle/working/b1/b1_summary.json — spread {min(kf):.3f}–{max(kf):.3f} across 10 folds.")
    finish(fig, "F17_b1_folds")


# ---------------------------------------------------------------- F18 B1 vs published
def f18():
    pub = [  # GPU_PLAN_AND_BENCHMARK.md §2.1 (papers' own tables)
        ("DeepSleepNet", 21e6, 0.785), ("SleepTransformer", 3.7e6, 0.814),
        ("U-Time", 1.1e6, 0.813), ("SeqSleepNet", 0.16e6, 0.826),
        ("TinySleepNet", 1.3e6, 0.831), ("LightSleepX", 0.049e6, 0.826),
        ("LMCSleepNet", 1.49e6, 0.841), ("SalientSleepNet", 0.9e6, 0.841),
        ("MultiChannelSleepNet", 13e6, 0.850),
    ]
    s = load(REPO / "kaggle" / "working" / "b1" / "b1_summary.json", "B1 point missing")
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, p, a in pub:
        ax.scatter(p, a, s=50, color=GREY, zorder=3)
        ax.annotate(lab, (p, a), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
    if s:
        ax.scatter(s["n_parameters"], s["mean_accuracy"], s=160, color=ACCENT, marker="*", zorder=4)
        ax.annotate(f"this work\n{ s['mean_accuracy']:.1%}, 0.176M", (s["n_parameters"], s["mean_accuracy"]),
                    xytext=(0, -26), textcoords="offset points", ha="center", fontsize=9, color=ACCENT)
    ax.set_xscale("log"); ax.set_xlabel("parameters (log)"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.77, 0.87); ax.set_xlim(3e4, 4e7)
    ax.set_title("Sleep-EDF-78: this work vs published baselines")
    source(fig, "b1_summary.json + GPU_PLAN_AND_BENCHMARK.md §2.1 (papers' own tables).")
    finish(fig, "F18_b1_vs_published")


# ---------------------------------------------------------------- F19 B1 per-class vs published
def f19():
    s = load(REPO / "kaggle" / "working" / "b1" / "b1_summary.json", "cannot draw")
    if s is None:
        return
    ours = [s["per_class_f1_mean"][st] for st in STAGES]
    tiny = [0.90, 0.51, 0.90, 0.77, 0.86]  # TinySleepNet-style strong light baseline, §2.1 GPU_PLAN
    xs = np.arange(5); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(xs - w / 2, tiny, w, color=GREY, label="TinySleepNet (1.3M, published)")
    ax.bar(xs + w / 2, ours, w, color=[C[st] for st in STAGES], label="this work (0.176M)")
    ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=8); ax.bar_label(ax.containers[1], fmt="%.2f", fontsize=8)
    ax.set_xticks(xs, STAGES); ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.legend(frameon=False)
    ax.set_title("Per-class F1 on Sleep-EDF-78")
    source(fig, "b1_summary.json (per_recording pooled). N3 headline is a recording-level artefact; epoch-pooled ≈ 0.78.")
    finish(fig, "F19_b1_per_class")


# ---------------------------------------------------------------- F20 verifier ablation
def f20():
    ab = load(REPO / "ROADMAP R2 - VERIFIER ABLATION" / "results" / "ablation.json", "cannot draw")
    if ab is None:
        return
    models, arms, flawed = [], [], []
    for key, v in ab.items():
        model, arm = key.rsplit(" / ", 1)
        if model not in models:
            models.append(model)
        arms.append((model, arm, v["summary"]["released_reports_flawed"]))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    xs = np.arange(3)
    for i, model in enumerate(models):
        vals = [f for m, a, f in arms if m == model]
        ax.bar(xs + (i - 0.5) * 0.36, vals, 0.34,
               color=ACCENT if i == 0 else GREY, label=model)
        ax.bar_label(ax.containers[i], fmt="%d")
    ax.set_xticks(xs, ["A (gate ON)", "B (gate OFF)", "C (no gate,\nno grammar)"])
    ax.set_ylabel("reports released with a flaw"); ax.set_ylim(0, 36)
    ax.legend(frameon=False); ax.axhline(31, color=RED, ls=":", lw=1)
    ax.text(2.9, 30, "all 31 nights flawed", color=RED, fontsize=8, ha="right")
    ax.set_title("The verifier releases nothing incorrect")
    source(fig, "ROADMAP R2 - VERIFIER ABLATION/results/ablation.json — accuracy independent of model quality.")
    finish(fig, "F20_verifier_ablation")


# ---------------------------------------------------------------- F22 N1 confidence
def f22():
    # §6k: accuracy when predicting N1, bucketed by the model's own confidence
    buckets = ["0.0–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    acc = [0.123, 0.392, 0.592, 0.805]
    nf = load(RES / "n1_flag_test_report.json", "test separation not shown")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bars = ax.bar(buckets, acc, 0.55, color=[RED, "#f4a63a", ACCENT, GREEN])
    ax.bar_label(bars, fmt="%.3f"); ax.set_ylabel("accuracy of N1 predictions")
    ax.set_ylim(0, 0.92); ax.set_xlabel("model's own max-probability when predicting N1")
    ax.set_title("The model knows which N1 calls to distrust")
    if nf:
        sep = nf["test"]["decoded (as shipped)"]["separation"]
        lo, hi = nf["separation_ci95_subject_clustered"]
        ax.text(0.98, 0.95, f"frozen flag (p<0.575) separates\n+{sep:.1%}  CI [{lo:+.1%}, {hi:+.1%}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#444")
    source(fig, "PROJECT_REPORT §6k + n1_flag_test_report.json.")
    finish(fig, "F22_n1_confidence")


# ---------------------------------------------------------------- F23 N1 threshold headroom
def f23():
    # §8.4: every decision threshold, best achievable 0.3476 vs current 0.339
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = ax.bar(["current\nthreshold", "best over ANY\nthreshold"], [0.339, 0.3476], 0.5, color=[GREY, ACCENT])
    ax.bar_label(bars, fmt="%.4f"); ax.set_ylabel("N1 F1"); ax.set_ylim(0, 0.45)
    ax.annotate("headroom +0.0086", (0.5, 0.385), ha="center", fontsize=9, color=RED)
    ax.set_title("N1 is representation-bound,\nnot threshold-bound")
    source(fig, "PROJECT_REPORT §8.4 (threshold sweep). Rebalancing cannot fix it — the operating point is already optimal.")
    finish(fig, "F23_n1_headroom")


# ---------------------------------------------------------------- F24 pairwise AUC
def f24():
    sep = load(RES / "separability_diagnostic_E1b_E1a.json", "cannot draw")
    if sep is None:
        return
    pw = sep["E1b"]["pairwise"]
    n1 = {k.replace("N1_vs_", ""): v["roc_auc"] for k, v in pw.items() if k.startswith("N1_vs_")}
    names, vals = zip(*sorted(n1.items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bars = ax.bar(names, vals, 0.55, color=[RED if v < 0.9 else GREY for v in vals])
    ax.bar_label(bars, fmt="%.3f", fontsize=8); ax.set_ylim(0.7, 1.0); ax.set_ylabel("ROC AUC")
    ax.set_xlabel("N1 versus ... (held-out test)")
    ax.set_title("One-vs-rest hides which pair is broken:\nit is N1 vs N2")
    source(fig, "separability_diagnostic_E1b_E1a.json.")
    finish(fig, "F24_pairwise_auc")


ALL = {
    "F1": f1, "F3": f3, "F5": f5, "F6": f6, "F7": f7, "F8": f8, "F9": f9, "F10": f10,
    "F11": f11, "F12": f12, "F13": f13, "F14": f14, "F15": f15, "F16": f16, "F17": f17,
    "F18": f18, "F19": f19, "F20": f20, "F22": f22, "F23": f23, "F24": f24,
}

if __name__ == "__main__":
    wanted = [a.upper() for a in sys.argv[1:]] or list(ALL)
    for k in wanted:
        print(f"[{k}]")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ALL[k]()
            except Exception as e:  # one bad figure must not stop the batch
                import traceback; traceback.print_exc()
                print(f"    !! FAILED: {type(e).__name__}: {e}")
    print(f"\ndone: {len(PARSED)}/{len(wanted)} figures -> {OUT}")
