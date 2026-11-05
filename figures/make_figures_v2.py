"""Figures corrected for release v2 (CURRENT_RELEASE.md) -> figures/output_v2/*.png (300 dpi).

    python figures/make_figures_v2.py            all corrected figures
    python figures/make_figures_v2.py F19 F17    only the listed ones

make_figures.py and figures/output/ are the frozen originals (30 Oct, release v1). This script redraws only
the figures that were wrong or out of date; the rest of output/ stays valid (see output_v2/README.md).
It reuses the original's style and helpers. Every test number comes from an already-logged evaluation file,
except F15, whose own read is logged in figures/TEST_ACCESS_LOG.md.

What changed, per figure:
  F1   legend no longer covers the Train bar; footnote updated (7 logged test reads, not "once").
  F3   x-labels no longer collide.
  F8   labelled as the superseded N3mc architecture; v2 ships with EOG (+0.018 κ, CV, same labels).
  F14  v1-era ladder labelled "stored labels"; new panel: v1 vs v2 on the same corrected labels.
  F15  release v2's confusion matrix (was v1's). One logged test read (figures/TEST_ACCESS_LOG.md); it must
       reproduce the already-logged stage3 evaluation exactly or nothing is written.
  F16  v2 vs v1 per stage, same corrected labels (was: v1 vs the E1b teacher, stored labels).
  F17  bars = final-epoch κ (unbiased); best-epoch κ, chosen on the reported fold, shown as markers.
  F18  our point uses the fold-pooled convention the papers use, at the final epoch; labels separated.
  F19  REMOVED four TinySleepNet per-class values that had no source. Now: our per-class F1 (fold-pooled,
       final epoch) and N1 F1 against every published N1 value in GPU_PLAN_AND_BENCHMARK.md §2.1.
  F20  labelled as the v1 report model; adds v2's test result (P8).
  F22  labelled release v1; v2's own flag (threshold 0.475) was never measured on test.
  F23  footnote no longer cut off.

New for the v2 paper (all from existing result files, no test reads): F25 report verification across releases,
F26 architecture ablation (CV), F27 missing-channel robustness, F28 vs YASA, F29 real recordings (Path B),
F30 metric reliability tiers, F31 signal-quality checks, F32 night-boundary error, F33 system overview.
"""
import json
import sys
import warnings

import numpy as np

import make_figures as M          # style, colours, load(), source(), finish()
from make_figures import plt, STAGES, C, ACCENT, GREY, GREEN, RED, REPO, load, source

M.OUT = M.HERE / "output_v2"
M.OUT.mkdir(exist_ok=True)
finish = M.finish

B1 = REPO / "kaggle" / "working" / "b1"
EVAL_V2 = REPO / "ADDING EOG VERSION - ADVANCED STUDENT" / "results" / "stage3_test_evaluation.json"
P8 = REPO / "ROADMAP P8 - V2 TEST EVALUATION" / "results.json"
ORANGE = "#f4a63a"


def b1_epochs() -> tuple[list[dict], list[dict]]:
    """Per fold: (final epoch, best epoch). Best = the checkpoint B1 kept (best validation macro-F1 on the
    SAME fold it reports - optimistic, FOR JOURNAL F1). Final = the last epoch trained (no selection)."""
    final, best = [], []
    for k in range(10):
        rows = [json.loads(l) for l in (B1 / f"fold_{k}" / "training_metrics.jsonl").read_text().splitlines() if l.strip()]
        final.append(rows[-1])
        best.append(max(rows, key=lambda r: r["macro_f1"]))
    return final, best


def mean(rows, key, stage=None):
    return float(np.mean([r[key][stage] if stage else r[key] for r in rows]))


# ---------------------------------------------------------------- F1 split
def f1():
    splits = [("Train", 54, 15), ("Validation", 12, 4), ("Test", 12, 3)]
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, (name, sc, st) in enumerate(splits):
        ax.barh(i, sc, color=ACCENT, label="SC (sleep cassette)" if i == 0 else None)
        ax.barh(i, st, left=sc, color=ORANGE, label="ST (sleep telemetry)" if i == 0 else None)
        ax.text(sc / 2, i, str(sc), va="center", ha="center", color="w", fontweight="bold")
        ax.text(sc + st / 2, i, str(st), va="center", ha="center", color="w", fontweight="bold")
    ax.set_yticks(range(3), [s[0] for s in splits])
    ax.set_xlabel("Subjects"); ax.set_xlim(0, 86)
    ax.legend(frameon=False, loc="upper right")          # empty area beside the short Test bar
    ax.set_title("Subject-level, cohort-stratified split (69 / 16 / 15 subjects)")
    source(fig, "distillation/make_splits.py. Test split: 7 logged reads, none used for selection (TEST_ACCESS_LOG files).")
    finish(fig, "F1_subject_split")


# ---------------------------------------------------------------- F3 exposure gradient (history)
def f3():
    # PROJECT_REPORT §3 / eval_teacher.json, unchanged numbers; only the tick labels are fixed
    lv = ["Same recording\n(trained on)", "Same subject,\nother night", "Unseen\nsubject"]
    kap, n3, rem = [0.8058, 0.6797, 0.5066], [0.889, 0.837, 0.000], [0.876, 0.718, 0.184]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    b = a1.bar(lv, kap, 0.55, color=[GREY, ORANGE, RED]); a1.bar_label(b, fmt="%.3f")
    a1.axhline(0.6663, color=ACCENT, ls="--"); a1.text(2.45, 0.68, "published κ\n0.6663", color=ACCENT, fontsize=8, ha="right")
    a1.set_ylim(0, 1); a1.set_ylabel("Cohen's κ"); a1.set_title("κ by exposure level"); a1.tick_params(axis="x", labelsize=8)
    xs = np.arange(3); w = 0.36
    a2.bar(xs - w / 2, n3, w, color=C["N3"], label="N3"); a2.bar(xs + w / 2, rem, w, color=C["REM"], label="REM")
    a2.bar_label(a2.containers[0], fmt="%.2f", fontsize=8); a2.bar_label(a2.containers[1], fmt="%.2f", fontsize=8)
    a2.set_xticks(xs, lv, fontsize=8); a2.set_ylim(0, 1); a2.set_ylabel("F1"); a2.legend(frameon=False)
    a2.set_title("Stage F1 by exposure level")
    source(fig, "PROJECT_REPORT.md §3 / eval_teacher.json (inherited model). The published figure blended the first two bars.")
    finish(fig, "F3_exposure_gradient")


# ---------------------------------------------------------------- F8 channel ablation (superseded)
def f8():
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(["test", "validation"], [0.0028, -0.0122], 0.5, color=[GREEN, RED])
    ax.axhline(0, color="#444", lw=1); ax.bar_label(ax.containers[0], fmt="%+.4f")
    ax.set_ylabel("Δ κ vs single-channel student"); ax.set_ylim(-0.027, 0.012)
    ax.set_title("Earlier architecture (N3mc): EOG + Pz-Oz did not help", fontsize=10)
    ax.text(0.5, 0.03, "paired Wilcoxon p = 0.865 (better on 15/29 recordings)\n"
            "Superseded: the v2 architecture uses EOG and gains +0.018 κ\n(5-fold CV, same labels, CI +0.010 to +0.026)",
            transform=ax.transAxes, ha="center", fontsize=8, color="#444")
    source(fig, "PROJECT_REPORT.md §6c (N3mc, Oct). v2 evidence: FOR JOURNAL/README.md F3.")
    finish(fig, "F8_channel_ablation")


# ---------------------------------------------------------------- F14 params vs kappa
def f14():
    ev = load(EVAL_V2, "panel (b) missing")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    pts = [  # PROJECT_REPORT §7 (held-out test, ORIGINAL STORED labels)
        ("inherited\n(honest subj.)", 649229, 0.5066, (-30, -18)), ("E0 teacher", 649229, 0.5122, (30, 2)),
        ("E1a teacher", 649229, 0.5592, (0, 9)), ("E1b teacher", 649229, 0.6055, (0, 9)),
        ("student baseline\n(121K)", 121099, 0.6449, (0, -28)),
        ("student + EOG\n+ Pz-Oz (N3mc)", 151606, 0.7020, (46, -26)), ("ensemble teacher", 410000, 0.7140, (0, 9)),
        # v1 N4kd and the multi-scale student are the same size (139,606) at κ 0.700 / 0.699: one label for both
        ("v1: N4kd, distilled (0.700)\n+ multi-scale, same size (0.699)", 139606, 0.7001, (-30, 13)),
    ]
    for lab, p, k, off in pts:
        a1.scatter(p, k, s=60, color=ACCENT if lab.startswith("v1") else GREY, zorder=3)
        a1.annotate(lab, (p, k), xytext=off, textcoords="offset points", ha="center", fontsize=7.5)
    a1.set_xscale("log"); a1.set_ylim(0.48, 0.76); a1.set_xlim(8e4, 9e5)
    a1.set_xlabel("trainable parameters (log)"); a1.set_ylabel("held-out test κ")
    a1.set_title("(a) Model history: original stored labels", fontsize=10)
    if ev:
        k1, k2 = ev["n4kd_argmax_corrected"]["kappa"], ev["variants"]["all_on"]["argmax"]["kappa"]
        cmp = ev["variants"]["all_on"]["vs_n4kd_argmax"]
        a2.scatter([139606], [k1], s=80, color=GREY, zorder=3); a2.scatter([175599], [k2], s=220, marker="*", color=ACCENT, zorder=3)
        a2.annotate(f"v1 N4kd\nκ {k1:.3f}", (139606, k1), xytext=(0, -30), textcoords="offset points", ha="center", fontsize=8)
        a2.annotate(f"v2 (current)\nκ {k2:.3f}", (175599, k2), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8, color=ACCENT)
        lo, hi = cmp["pooled_dkappa_ci95_subject_bootstrap"]
        a2.text(0.5, 0.06, f"Δκ +{k2 - k1:.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]\nbetter on {cmp['subjects_better']}/{cmp['subjects']} subjects",
                transform=a2.transAxes, ha="center", fontsize=8, color="#444")
        a2.set_xlim(1.2e5, 2.1e5); a2.set_ylim(0.68, 0.78); a2.set_xlabel("trainable parameters"); a2.set_ylabel("held-out test κ")
        a2.set_title("(b) Releases: same corrected labels", fontsize=10)
    fig.suptitle("Held-out test κ against model size. Compare within a panel, never across.", fontsize=10)
    source(fig, "(a) PROJECT_REPORT.md §7, stored labels (one-epoch shift on ~5% of epochs). (b) stage3_test_evaluation.json, corrected labels.")
    finish(fig, "F14_params_vs_kappa")


# ---------------------------------------------------------------- F15 confusion matrix, release v2
def f15():
    """Logged in figures/TEST_ACCESS_LOG.md. Reproduces the logged stage3 evaluation exactly or writes nothing."""
    from sklearn.metrics import cohen_kappa_score
    ev = load(EVAL_V2, "cannot check")
    labels = load(REPO / "ADDING EOG VERSION - ST NIGHTS" / "results" / "corrected_labels_test.json", "cannot draw")
    # Raw (unrounded) v2 outputs on the scored test epochs (make_prob_caches.py). The packets' rounded
    # calibrated probabilities flip 1 of 33,365 argmaxes, so they cannot reproduce the evaluation exactly.
    cache = REPO / "ADDING EOG VERSION - PIPELINE INTEGRATION" / "work_n6adv" / "distillation" / "results" / "probs_student_N6adv_full_s1"
    if ev is None or labels is None:
        return
    y, p = [], []
    for f in sorted(cache.glob("*.npz")):
        d = np.load(f)
        lab = [STAGES.index(s) for s in labels[f.stem].split() if s != "?"]
        assert lab == d["labels"].astype(int).tolist(), f.stem      # cache labels = corrected labels, "?" dropped
        y += lab; p += list(d["probs"].argmax(1))
    y, p = np.array(y), np.array(p)
    ref = ev["variants"]["all_on"]["argmax"]
    acc, kap = float((y == p).mean()), float(cohen_kappa_score(y, p))
    assert len(y) == ref["epochs"] and abs(acc - ref["accuracy"]) < 1e-9 and abs(kap - ref["kappa"]) < 1e-9, \
        f"does not reproduce the logged evaluation: n={len(y)} acc={acc} kappa={kap}"
    cm = np.zeros((5, 5), int)
    np.add.at(cm, (y, p), 1)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=9, color="w" if cmn[i, j] > 0.55 else "#333")
    ax.set_xticks(range(5), STAGES); ax.set_yticks(range(5), STAGES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (corrected labels)"); ax.grid(False)
    ax.set_title("Release v2, held-out test (argmax)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    source(fig, f"{len(y):,} epochs; equals stage3_test_evaluation.json (acc {acc:.4f}, κ {kap:.4f}). Log: figures/TEST_ACCESS_LOG.md")
    finish(fig, "F15_confusion_matrix")


# ---------------------------------------------------------------- F16 per-class F1, v2 vs v1
def f16():
    ev = load(EVAL_V2, "cannot draw")
    if ev is None:
        return
    v2 = [ev["variants"]["all_on"]["argmax"]["per_class"][s]["f1"] for s in STAGES]
    v1 = [ev["n4kd_argmax_corrected"]["per_class"][s]["f1"] for s in STAGES]
    xs = np.arange(5); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(xs - w / 2, v1, w, color=GREY, label="release v1: N4kd (EEG)")
    ax.bar(xs + w / 2, v2, w, color=[C[s] for s in STAGES], label="release v2 (EEG + EOG)")
    ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=7.5); ax.bar_label(ax.containers[1], fmt="%.3f", fontsize=7.5)
    ax.set_xticks(xs, STAGES); ax.set_ylim(0, 1.05); ax.set_ylabel("F1"); ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_title("v2 improves every stage; N1 remains the weakest")
    source(fig, f"stage3_test_evaluation.json: held-out test, {ev['variants']['all_on']['argmax']['epochs']:,} epochs, both models on the same corrected labels.")
    finish(fig, "F16_per_class_f1")


# ---------------------------------------------------------------- F17 B1 per-fold kappa
def f17():
    final, best = b1_epochs()
    kf, kb = [r["kappa"] for r in final], [r["kappa"] for r in best]
    xs = np.arange(10)
    fig, ax = plt.subplots(figsize=(8, 3.9))
    ax.bar(xs, kf, 0.6, color=ACCENT, label=f"final epoch (mean {np.mean(kf):.3f})")
    ax.scatter(xs, kb, marker="_", s=380, color="#222", zorder=4, label=f"best epoch, chosen on the same fold (mean {np.mean(kb):.3f})")
    ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=7.5, padding=-13, color="white")  # inside: clear of the markers
    ax.axhline(np.mean(kf), color=GREEN, ls="--", lw=1)
    ax.set_xticks(xs, [f"f{i}" for i in xs]); ax.set_ylim(0.70, 0.85); ax.set_ylabel("Cohen's κ")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_title("Sleep-EDF-78: per-fold κ (subject-wise 10-fold CV, from scratch)")
    source(fig, f"kaggle/working/b1/fold_*/training_metrics.jsonl. Final-epoch spread {min(kf):.3f}–{max(kf):.3f}; the best-epoch figure is optimistic (FOR JOURNAL F1).")
    finish(fig, "F17_b1_folds")


# ---------------------------------------------------------------- F18 B1 vs published
def f18():
    final, _ = b1_epochs()
    acc = mean(final, "accuracy")
    pub = [  # GPU_PLAN_AND_BENCHMARK.md §2.1 (papers' own tables; secondary source, FOR JOURNAL F8)
        ("DeepSleepNet", 21e6, 0.785, (0, 8)), ("SleepTransformer", 3.7e6, 0.814, (0, 8)),
        ("U-Time", 1.1e6, 0.813, (0, -14)), ("SeqSleepNet", 0.16e6, 0.826, (0, -14)),
        ("TinySleepNet", 1.3e6, 0.831, (0, 8)), ("LightSleepX", 0.049e6, 0.826, (0, 8)),
        ("LMCSleepNet", 1.49e6, 0.841, (34, 6)), ("SalientSleepNet", 0.9e6, 0.841, (-38, 6)),
        ("MultiChannelSleepNet", 13e6, 0.850, (0, 8)),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, p, a, off in pub:
        ax.scatter(p, a, s=50, color=GREY, zorder=3)
        ax.annotate(lab, (p, a), xytext=off, textcoords="offset points", ha="center", fontsize=7)
    ax.scatter(175599, acc, s=180, color=ACCENT, marker="*", zorder=4)
    ax.annotate(f"this work (EEG + EOG)\n{acc:.1%}, 0.176M", (175599, acc), xytext=(0, 14),
                textcoords="offset points", ha="center", fontsize=8.5, color=ACCENT)
    ax.set_xscale("log"); ax.set_xlabel("parameters (log)"); ax.set_ylabel("accuracy (all epochs pooled)")
    ax.set_ylim(0.77, 0.87); ax.set_xlim(3e4, 4e7)
    ax.set_title("Sleep-EDF-78: this work vs published baselines")
    source(fig, "Ours: fold-pooled accuracy at the final epoch, mean of 10 folds. Published: GPU_PLAN_AND_BENCHMARK.md §2.1; inputs differ (EEG only to 3 channels).")
    finish(fig, "F18_b1_vs_published")


# ---------------------------------------------------------------- F19 B1 per-class + N1 vs published
def f19():
    final, _ = b1_epochs()
    ours = [mean(final, "per_class_f1", s) for s in STAGES]
    n1_pub = [  # GPU_PLAN_AND_BENCHMARK.md §2.1, N1 F1 column: the only per-class value the table gives
        ("AttnSleep", 42.0), ("SleepEEGNet", 44.6), ("SeqSleepNet", 46.0), ("DeepSleepNet", 47.0),
        ("LMCSleepNet", 48.5), ("U-Time", 51.0), ("TinySleepNet", 51.0), ("MultiChannelSleepNet", 53.0),
        ("SalientSleepNet", 54.2), ("LightSleepX †", 63.0)]
    rows = sorted(n1_pub + [("this work", ours[1] * 100)], key=lambda r: r[1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [1.15, 1]})
    b = a1.bar(STAGES, ours, 0.6, color=[C[s] for s in STAGES]); a1.bar_label(b, fmt="%.3f", fontsize=8)
    a1.set_ylim(0, 1); a1.set_ylabel("F1"); a1.set_title("(a) This work, per stage", fontsize=10)
    ys = np.arange(len(rows))
    a2.barh(ys, [r[1] for r in rows], color=[ACCENT if r[0] == "this work" else GREY for r in rows])
    a2.set_yticks(ys, [r[0] for r in rows], fontsize=8); a2.set_xlim(0, 70); a2.set_xlabel("N1 F1 (%)")
    for y, (_, v) in zip(ys, rows):
        a2.text(v + 0.8, y, f"{v:.1f}", va="center", fontsize=7.5)
    a2.set_title("(b) N1, the hardest stage, against published", fontsize=10)
    fig.suptitle("Sleep-EDF-78 per-class F1", fontsize=11)
    source(fig, "Ours: fold-pooled, final epoch, mean of 10 folds (training_metrics.jsonl). Published: GPU_PLAN_AND_BENCHMARK.md §2.1. "
                "† reported value, unusually high.")
    finish(fig, "F19_b1_per_class")


# ---------------------------------------------------------------- F20 verifier ablation + P8
def f20():
    """v2 verifier ablation (R2b) + the untrained base model (R2, v1 packets) + v2 on held-out test (P8)."""
    v2 = load(REPO / "ROADMAP R2b - V2 VERIFIER ABLATION" / "results" / "ablation_v2.json", "cannot draw")
    r2 = load(REPO / "ROADMAP R2 - VERIFIER ABLATION" / "results" / "ablation.json", "base-model bars missing")
    p8 = (load(P8, "P8 note missing") or {}).get("summary")
    if v2 is None:
        return
    arms = ("A", "B", "C")
    series = [("Llama-Student v2 (release v2), v2 dev packets", [v2[f"Llama-Student v2 / {a}"]["summary"]["released_reports_flawed"] for a in arms], ACCENT)]
    if r2:
        series.append(("Llama-3.2-3B-Instruct, untrained base (R2, v1 dev packets)",
                       [r2[f"Llama-3.2-3B-Instruct / {a}"]["summary"]["released_reports_flawed"] for a in arms], GREY))
    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    xs = np.arange(3)
    for i, (lab, vals, col) in enumerate(series):
        ax.bar(xs + (i - 0.5) * 0.36, vals, 0.34, color=col, label=lab)
        ax.bar_label(ax.containers[i], fmt="%d")
    ax.set_xticks(xs, ["A (grammar + gate)", "B (grammar,\nno gate)", "C (no grammar,\nno gate)"])
    ax.set_ylabel("reports released with a flaw (of 31)"); ax.set_ylim(0, 40)
    ax.legend(frameon=False, loc="upper left", fontsize=8); ax.axhline(31, color=RED, ls=":", lw=1)
    note = "v2: 0 flawed in every arm on dev (arm C output\nidentical to arm A: the grammar changed nothing)"
    if p8:
        note += (f"\nHeld-out test (P8, gate on): {p8['flawed_reports_released']} of {p8['nights']} flawed released, "
                 f"{p8['nights'] - p8['verified_reports']} blocked")
    ax.text(0.27, 0.5, note, transform=ax.transAxes, ha="center", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))
    ax.set_title("The verifier: a safety net for v2, the decisive control for an untrained model", fontsize=10)
    source(fig, "R2b ablation_v2.json (v2, 31 dev nights); R2 ablation.json (base model, v1 packets); P8 results.json.")
    finish(fig, "F20_verifier_ablation")


# ---------------------------------------------------------------- F22 N1 confidence (v1)
def f22():
    buckets, acc = ["0.0–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"], [0.123, 0.392, 0.592, 0.805]  # §6k
    nf = load(M.RES / "n1_flag_test_report.json", "test separation not shown")
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    bars = ax.bar(buckets, acc, 0.55, color=[RED, ORANGE, ACCENT, GREEN]); ax.bar_label(bars, fmt="%.3f")
    ax.set_ylabel("accuracy of N1 predictions"); ax.set_ylim(0, 1.0)
    ax.set_xlabel("model's own max-probability when predicting N1")
    ax.set_title("Release v1 (N4kd): the model knows which N1 calls to distrust", fontsize=10)
    if nf:
        sep = nf["test"]["decoded (as shipped)"]["separation"]
        lo, hi = nf["separation_ci95_subject_clustered"]
        ax.text(0.02, 0.95, f"v1 flag (p < 0.575) separates\n+{sep:.1%}, CI [{lo:+.1%}, {hi:+.1%}]",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#444")
    source(fig, "PROJECT_REPORT §6k + n1_flag_test_report.json (v1). v2 uses its own flag (threshold 0.475), not measured on test.")
    finish(fig, "F22_n1_confidence")


# ---------------------------------------------------------------- F23 N1 headroom (footnote fix)
def f23():
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    bars = ax.bar(["current\nthreshold", "best over ANY\nthreshold"], [0.339, 0.3476], 0.5, color=[GREY, ACCENT])
    ax.bar_label(bars, fmt="%.4f"); ax.set_ylabel("N1 F1"); ax.set_ylim(0, 0.45)
    ax.annotate("headroom +0.0086", (0.5, 0.385), ha="center", fontsize=9, color=RED)
    ax.set_title("N1 is representation-bound,\nnot threshold-bound")
    source(fig, "PROJECT_REPORT §8.4 (threshold sweep).")
    finish(fig, "F23_n1_headroom")


# ================================================================ new figures for the v2 paper (F25-F33)
# All from existing result files; no test-label reads.

R3 = REPO / "ROADMAP R3 - V2 REPORT MODEL" / "results" / "dev_summary.json"
CV = REPO / "ADDING EOG VERSION" / "kaggle_results" / "advanced_stage1" / "cv_summary.json"
AR = REPO / "agentic" / "reports"


def cp_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval (the convention behind '29/29 -> lower bound 88.1%')."""
    from math import comb

    def tail_ge(j, p):   # P(X >= j), increasing in p
        return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(j, n + 1))

    def solve(j, target):  # p with P(X >= j) = target
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if tail_ge(j, mid) < target else (lo, mid)
        return lo
    lower = 0.0 if k == 0 else solve(k, alpha / 2)
    upper = 1.0 if k == n else solve(k + 1, 1 - alpha / 2)    # P(X <= k) = alpha/2
    return lower, upper


# ---------------------------------------------------------------- F25 report verification, both releases
def f25():
    fz = (load(REPO / "deploy" / "frozen.json") or {}).get("test_results") or {}
    r3, p8 = (load(R3) or {}).get("summary") or {}, (load(P8) or {}).get("summary") or {}
    k1d, n1d = map(int, fz["dev_5_of_5"].split("/")); k1t, n1t = map(int, fz["test_5_of_5"].split("/"))
    rows = [("v1, dev", k1d, n1d, GREY), ("v1, test", k1t, n1t, GREY),
            ("v2, dev", r3["five_of_five"], r3["nights"], ACCENT), ("v2, test (P8)", p8["five_of_five"], p8["nights"], ACCENT)]
    fig, ax = plt.subplots(figsize=(7.4, 4))
    for i, (lab, k, n, col) in enumerate(rows):
        lo, hi = cp_interval(k, n)
        ax.bar(i, k / n, 0.55, color=col)
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color="#222", capsize=6, lw=1.2)
        ax.text(i, 0.05, f"{k}/{n}", ha="center", color="white", fontweight="bold")
        ax.text(i + 0.06, lo, f"≥ {lo:.1%}", ha="left", va="center", fontsize=8, color="#444")
    ax.set_xticks(range(4), [r[0] for r in rows]); ax.set_ylim(0, 1.08)
    ax.set_ylabel("reports verified with all 5 mandatory items")
    ax.set_title("Report generation: verified on every evaluation, no flawed report released", fontsize=10)
    ax.text(0.5, 0.42, "Flawed reports released: 0 in all four\n(the v2 test failure was blocked by the verifier)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"))
    source(fig, "95% Clopper-Pearson. Sources: frozen.json, R3 dev_summary.json, P8 results.json. v1/v2: different packets, never pool.")
    finish(fig, "F25_report_verification")


# ---------------------------------------------------------------- F26 architecture ablation (CV)
def f26():
    cv = load(CV, "cannot draw")
    if cv is None:
        return
    arms = [("base\n(v1 architecture, EEG)", None), ("+ TF + local context\n(EEG only)", cv["arms"]["full_eeg_fixlabels"]),
            ("+ EOG = full\n(v2 architecture)", cv["arms"]["full_fixlabels"])]
    base = cv["arms"]["full_fixlabels"]["base_mean_kappa"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1, 1.1]})
    ks = [base] + [a["arm_mean_kappa"] for _, a in arms[1:]]
    b = a1.bar([a[0] for a in arms], ks, 0.55, color=[GREY, ORANGE, ACCENT]); a1.bar_label(b, fmt="%.4f", fontsize=8)
    a1.set_ylim(0.74, 0.785); a1.set_ylabel("mean κ, 5-fold CV"); a1.tick_params(axis="x", labelsize=8)
    a1.set_title("(a) κ by architecture (same labels, same folds)", fontsize=10)
    ds = [a["delta"]["kappa"] for _, a in arms[1:]]
    for i, d in enumerate(ds):
        a2.errorbar(d["mean"], i, xerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]], fmt="o",
                    color=[ORANGE, ACCENT][i], capsize=6, ms=8)
        a2.text(d["ci95"][1] + 0.0008, i, f"+{d['mean']:.4f}  [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  {d['better']}/5 folds",
                va="center", fontsize=8)
    a2.axvline(0, color="#444", lw=1); a2.axvline(cv["bar_delta_kappa"], color=RED, ls=":", lw=1)
    a2.text(cv["bar_delta_kappa"], 1.45, f"pre-set bar +{cv['bar_delta_kappa']}", color=RED, fontsize=8, ha="center")
    a2.set_yticks([0, 1], ["TF + local\n(EEG only)", "full\n(+ EOG)"], fontsize=8); a2.set_ylim(-0.6, 1.7); a2.set_xlim(-0.004, 0.05)
    a2.set_xlabel("Δκ vs base (95% CI)"); a2.set_title("(b) Gain over the v1 architecture", fontsize=10)
    source(fig, "ADDING EOG VERSION/kaggle_results/advanced_stage1/cv_summary.json (STAGE1_REPORT.md): 5-fold CV, corrected labels.")
    finish(fig, "F26_architecture_ablation")


# ---------------------------------------------------------------- F27 robustness to missing inputs
def f27():
    ev = load(EVAL_V2, "cannot draw")
    if ev is None:
        return
    v = ev["variants"]
    bars = [("v1 N4kd\n(EEG)", ev["n4kd_argmax_corrected"]["kappa"], GREY), ("v2, all inputs", v["all_on"]["argmax"]["kappa"], ACCENT),
            ("v2, EOG missing", v["eog_off"]["argmax"]["kappa"], ORANGE), ("v2, TF branch off", v["tf_off"]["argmax"]["kappa"], "#7aa6e8")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
    b = a1.bar([x[0] for x in bars], [x[1] for x in bars], 0.6, color=[x[2] for x in bars]); a1.bar_label(b, fmt="%.4f", fontsize=8)
    a1.axhline(bars[0][1], color=GREY, ls="--", lw=1); a1.set_ylim(0.68, 0.78); a1.set_ylabel("held-out test κ")
    a1.tick_params(axis="x", labelsize=8); a1.set_title("(a) A missing channel degrades gracefully, still above v1", fontsize=10)
    xs, w = np.arange(2), 0.36
    for j, (name, col) in enumerate([("all_on", ACCENT), ("eog_off", ORANGE)]):
        vals = [v[name]["by_cohort"][c]["kappa"] for c in ("SC", "ST")]
        a2.bar(xs + (j - 0.5) * w, vals, w, color=col, label={"all_on": "all inputs", "eog_off": "EOG missing"}[name])
        a2.bar_label(a2.containers[-1], fmt="%.3f", fontsize=8)
    a2.set_xticks(xs, ["SC (sleep cassette)", "ST (sleep telemetry)"]); a2.set_ylim(0.68, 0.78); a2.legend(frameon=False, fontsize=8)
    a2.set_title("(b) v2 by cohort", fontsize=10)
    source(fig, f"stage3_test_evaluation.json: held-out test, {v['all_on']['argmax']['epochs']:,} epochs, corrected labels; channels switched off at inference.")
    finish(fig, "F27_missing_channel_robustness")


# ---------------------------------------------------------------- F28 same nights vs YASA
def f28():
    r = load(AR / "r1_yasa_comparison.json", "cannot draw")
    if r is None:
        return
    ky = [x["yasa"]["kappa"] for x in r["rows"]]; kp = [x["project"]["kappa"] for x in r["rows"]]
    better = sum(p > y for p, y in zip(kp, ky))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.25]})
    a1.scatter(ky, kp, s=30, color=ACCENT, zorder=3); a1.plot([0.2, 0.95], [0.2, 0.95], color="#888", ls="--", lw=1)
    a1.set_xlim(0.2, 0.95); a1.set_ylim(0.2, 0.95); a1.set_xlabel("YASA κ"); a1.set_ylabel("this work (v2) κ")
    a1.set_title(f"(a) Per night: v2 higher on {better}/{len(kp)}", fontsize=10)
    a1.text(0.62, 0.27, "below the line:\nYASA better", fontsize=8, color="#666")
    xs, w = np.arange(5), 0.38
    pm = r["per_class_f1_mean"]
    a2.bar(xs - w / 2, [pm[s]["yasa"] for s in STAGES], w, color=GREY, label=f"YASA {r['yasa_version']} (pre-trained)")
    a2.bar(xs + w / 2, [pm[s]["project"] for s in STAGES], w, color=[C[s] for s in STAGES], label="this work, release v2")
    a2.bar_label(a2.containers[0], fmt="%.2f", fontsize=7.5); a2.bar_label(a2.containers[1], fmt="%.2f", fontsize=7.5)
    a2.set_xticks(xs, STAGES); a2.set_ylim(0, 1.05); a2.set_ylabel("mean per-night F1"); a2.legend(frameon=False, fontsize=8, loc="upper left")
    a2.set_title(f"(b) Mean κ {r['mean']['kappa']['project']:.3f} vs {r['mean']['kappa']['yasa']:.3f}", fontsize=10)
    source(fig, f"agentic/reports/r1_yasa_comparison.json: {r['nights']} dev nights, same estimated night spans. In-domain advantage: v2 was trained on Sleep-EDF (FOR JOURNAL F10).")
    finish(fig, "F28_vs_yasa")


# ---------------------------------------------------------------- F29 real recordings (Path B)
def f29():
    s1, s2 = load(AR / "p2_staging_scorecard_v1.json"), load(AR / "p2_staging_scorecard_v2.json")
    sc = load(AR / "p2_scorecards_v2_llama.json")
    if not (s1 and s2 and sc):
        return
    k1 = {r["recording_id"]: r["kappa"] for r in s1["rows"]}
    pairs = [(k1[r["recording_id"]], r["kappa"]) for r in s2["rows"] if r["recording_id"] in k1]
    better = sum(b > a for a, b in pairs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.1]})
    a1.scatter([a for a, _ in pairs], [b for _, b in pairs], s=30, color=ACCENT, zorder=3)
    a1.plot([0.3, 0.95], [0.3, 0.95], color="#888", ls="--", lw=1); a1.set_xlim(0.3, 0.95); a1.set_ylim(0.3, 0.95)
    a1.set_xlabel("v1 κ (EEG)"); a1.set_ylabel("v2 κ (EEG + EOG)")
    a1.set_title(f"(a) Staging per night: v2 higher on {better}/{len(pairs)}\nmean κ {s1['pooled']['kappa_mean']:.3f} → {s2['pooled']['kappa_mean']:.3f}", fontsize=10)
    sy = sc["system"]
    outs = [("verified\nreport", sy["VERIFIED_REPORT"], GREEN), ("review\nonly", sy["REVIEW_ONLY"], ORANGE),
            ("fallback\nsummary", sy["FALLBACK_SUMMARY"], RED), ("rejected", sy["REJECTED"], GREY)]
    b = a2.bar([o[0] for o in outs], [o[1] for o in outs], 0.6, color=[o[2] for o in outs]); a2.bar_label(b, fmt="%d")
    a2.set_ylim(0, sy["nights"] + 4); a2.set_ylabel("nights"); a2.tick_params(axis="x", labelsize=8)
    a2.set_title(f"(b) v2 system outcome, {sy['nights']} raw recordings\nmedian {sy['latency_s_median']:.0f} s per report", fontsize=10)
    a2.text(1, sy["REVIEW_ONLY"] + 2.2, "line noise (Q03)", ha="center", fontsize=7.5, color="#666")
    source(fig, "Path B (raw EDF → agent), 31 dev nights cut at lights-off: agentic/reports/p2_staging_scorecard_v1/_v2.json, p2_scorecards_v2_llama.json.")
    finish(fig, "F29_real_recordings")


# ---------------------------------------------------------------- F30 which clinical metrics can be trusted
def f30():
    dm = load(REPO / "ADDING EOG VERSION - PIPELINE INTEGRATION" / "work_n6adv" / "distillation" / "results" /
              "derived_metric_reliability.json", "cannot draw")
    if dm is None:
        return
    col = {"robust": GREEN, "fragile": ORANGE, "unreliable": RED}
    rows = sorted(dm["metrics"].items(), key=lambda kv: kv[1]["mean_relative_error"])
    cap = 1.0
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ys = np.arange(len(rows))
    vals = [min(m["mean_relative_error"], cap) for _, m in rows]
    ax.barh(ys, vals, color=[col[m["tier"]] for _, m in rows])
    for y, (_, m) in zip(ys, rows):
        e = m["mean_relative_error"]
        ax.text(min(e, cap) + 0.015, y, f"{e:.0%}" + (" →" if e > cap else ""), va="center", fontsize=7.5)
    ax.set_yticks(ys, [k.replace("_", " ") for k, _ in rows], fontsize=8)
    ax.axvline(0.10, color=GREEN, ls=":", lw=1); ax.axvline(0.40, color=RED, ls=":", lw=1)
    ax.set_xlim(0, cap + 0.12); ax.set_xlabel("mean relative error against expert scoring (validation)")
    for name, note in (("REM_Latency", "clinical override: produced a false\nnarcolepsy flag on validation nights"),
                       ("Stage_Transition_Entropy", "run-structure metrics are never\nrated robust, whatever the error")):
        y = [k for k, _ in rows].index(name)
        ax.annotate(note, (dm["metrics"][name]["mean_relative_error"] + 0.06, y), xytext=(0.55, y), fontsize=7.5,
                    color="#444", va="center", arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in col.values()]
    ax.legend(handles, ["robust: safe to state", "fragile: state with error bound", "unreliable: caveat or omit"],
              frameon=False, fontsize=8, loc="lower right")
    n = {t: sum(m["tier"] == t for _, m in rows) for t in col}
    ax.set_title(f"Which numbers the report may assert (release v2): {n['robust']} robust, {n['fragile']} fragile, {n['unreliable']} unreliable", fontsize=10)
    source(fig, f"work_n6adv/.../derived_metric_reliability.json: {dm['n_recordings']} validation nights; thresholds 10% / 40% (distillation/metric_reliability.py).")
    finish(fig, "F30_metric_reliability")


# ---------------------------------------------------------------- F31 signal-quality checks
def f31():
    ne = load(REPO / "ROADMAP V5 - LINE NOISE" / "results" / "noise_effect.json")
    qc = load(REPO / "ROADMAP V3 - QUALITY FLAG CI" / "results" / "v3_quality_ci.json")
    if not (ne and qc):
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))
    mk = ne["mean_kappa"]
    for tone, colr, lab in (("49 Hz", RED, "49 Hz tone"), ("50 Hz (Nyquist)", ORANGE, "50 Hz tone")):
        xs = [0.0] + [float(k.split("x")[-1]) for k in mk if k.startswith(tone)]
        ys = [mk["clean x0.0"]] + [mk[k] for k in mk if k.startswith(tone)]
        a1.plot(xs, ys, "o-", color=colr, label=lab)
    a1.set_xlabel("injected tone amplitude (× EEG RMS)"); a1.set_ylabel(f"mean κ ({len(ne['nights'])} nights)")
    a1.set_ylim(0, 0.85); a1.legend(frameon=False, fontsize=8)
    a1.set_title("(a) Mains noise destroys staging → the Q03/W10 check", fontsize=10)
    d = qc["ci"]["dev_W02"]; t = qc["ci"]["train_W02"]
    for i, (lab, c, colr) in enumerate([("train (calibration)", t, GREY), ("dev", d, ACCENT)]):
        a2.errorbar(c["point"], i, xerr=[[c["point"] - c["ci95"][0]], [c["ci95"][1] - c["point"]]], fmt="o", color=colr, capsize=6, ms=8)
        a2.text(c["ci95"][1] + 0.01, i, f"{c['point']:+.3f}  [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]\n{c['n_flagged']} flagged / {c['n_clean']} clean",
                va="center", fontsize=8)
    a2.axvline(0, color="#444", lw=1); a2.set_yticks([0, 1], ["train", "dev"]); a2.set_ylim(-0.7, 1.7); a2.set_xlim(-0.25, 0.2)
    a2.set_xlabel("κ(flagged) − κ(clean), 95% bootstrap CI")
    a2.set_title("(b) W02 flag picks out worse nights on dev", fontsize=10)
    source(fig, "(a) ROADMAP V5 - LINE NOISE/results/noise_effect.json. (b) ROADMAP V3 - QUALITY FLAG CI/results/v3_quality_ci.json "
                "(threshold set on train, so no train effect is expected).")
    finish(fig, "F31_quality_checks")


# ---------------------------------------------------------------- F32 night-boundary estimation (Path B)
def f32():
    s2 = load(AR / "p2_staging_scorecard_v2.json", "cannot draw")
    if s2 is None:
        return
    st = np.array([r["start_err"] for r in s2["rows"]]); en = np.array([r["end_err"] for r in s2["rows"]])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, e, name in ((axs[0], st, "start"), (axs[1], en, "end")):
        bins = np.arange(-130, 135, 10)
        ax.hist(np.clip(e, -129, 129), bins=bins, color=ACCENT if name == "start" else ORANGE)
        ax.axvspan(-30, 30, color=GREEN, alpha=0.08); ax.axvline(0, color="#444", lw=1)
        a = np.abs(e)
        ax.set_title(f"Night {name}: median |error| {np.median(a):.0f}, p90 {np.percentile(a, 90):.0f} epochs", fontsize=10)
        ax.set_xlabel("estimated − expert (epochs of 30 s)")
    axs[0].set_ylabel("nights")
    axs[1].text(0.98, 0.92, "shaded: the ±30-epoch bar\n(end p90 misses it: accepted\ndeviation X1)", transform=axs[1].transAxes,
                ha="right", va="top", fontsize=8, color="#444")
    source(fig, f"agentic/reports/p2_staging_scorecard_v2.json: {len(st)} dev nights, model-only estimator v0.3 vs the expert-derived span.")
    finish(fig, "F32_night_boundaries")


# ---------------------------------------------------------------- F33 system overview (schematic)
def f33():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 3.4)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.4); ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="#555", lw=1))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=7.4, linespacing=1.35)

    def arrow(x1, x2, y=1.95):
        ax.annotate("", (x2, y), (x1, y), arrowprops=dict(arrowstyle="->", color="#444", lw=1.3))

    y, h = 1.3, 1.3
    box(0.1, y, 1.75, h, "1. Recording", "EDF: EEG Fpz-Cz\n+ horizontal EOG\nnight found, quality\nchecked (W02/Q03)", "#eef3fb")
    box(2.15, y, 1.85, h, "2. Staging", "student, 175,599 params\nEEG + EOG + TF branch\ncalibrated 5-class\nprobabilities per 30 s", "#eef3fb")
    box(4.3, y, 1.9, h, "3. Evidence packet", "clinic metrics with\nmeasured error bounds,\nreliability tiers,\nN1 flags, 19 items", "#fdf4e3")
    box(6.5, y, 1.8, h, "4. Report model", "Llama-3.2-3B-Student v2\nlocal, grammar-\nconstrained, writes\nfield-cited claims", "#fdf4e3")
    box(8.6, y, 2.3, h, "5. Verifier (gate)", "30 deterministic rules →\nVERIFIED REPORT\nor FALLBACK SUMMARY\nor REVIEW ONLY", "#e8f5ee")
    for a, b in ((1.85, 2.15), (4.0, 4.3), (6.2, 6.5), (8.3, 8.6)):
        arrow(a, b)
    ax.add_patch(FancyBboxPatch((0.1, 0.2), 10.8, 0.6, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f4f3ef", ec="#999", lw=1, ls="--"))
    ax.text(5.5, 0.5, "Agent (workflow only, never clinical content): run identity, checkpoint/resume, resource checks, "
            "routing, one delivery per night, audit trace", ha="center", va="center", fontsize=8)
    ax.text(5.5, 3.15, "Release v2: one night of EEG + EOG becomes a verified sleep report, fully offline", ha="center", fontsize=11, fontweight="bold")
    finish(fig, "F33_system_overview")


ALL = {"F1": f1, "F3": f3, "F8": f8, "F14": f14, "F15": f15, "F16": f16, "F17": f17, "F18": f18,
       "F19": f19, "F20": f20, "F22": f22, "F23": f23,
       "F25": f25, "F26": f26, "F27": f27, "F28": f28, "F29": f29, "F30": f30, "F31": f31, "F32": f32, "F33": f33}

if __name__ == "__main__":
    final, best = b1_epochs()   # self-check against the numbers verified on 1 Nov (ROADMAP B1 VERIFICATION.md)
    assert abs(mean(best, "kappa") - 0.7753) < 5e-4 and abs(mean(final, "kappa") - 0.7690) < 5e-4, "B1 logs changed"
    assert abs(sum([0.90, 0.51, 0.90, 0.77, 0.86]) / 5 - 0.781) > 0.005   # why F19's old baseline bars were dropped
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
    print(f"\ndone: {len(M.PARSED)}/{len(wanted)} figures -> {M.OUT}")
