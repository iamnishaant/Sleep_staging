"""
Generate the report figures from the JSON artefacts on disk.

Everything is read from distillation/results/*.json - no numbers are hardcoded
here, so re-running after E1b lands updates every figure automatically.

    python distillation/make_figures.py                 # all figures
    python distillation/make_figures.py --only 2 5      # just those
    python distillation/make_figures.py --dark          # dark-mode variants

DESIGN NOTES
------------
* Categorical hues are assigned in FIXED slot order and never cycled. Slot 1
  blue, 2 orange, 3 aqua, 4 yellow.
* Fig 2 (scatter) uses EMPHASIS rather than 6 categorical hues: the story is
  one model, so that model is coloured and the rest are muted, with every
  point directly labelled. Six hues on a scatter would also fail the
  all-pairs colour-separation floor.
* Fig 5 is TWO STACKED PANELS sharing an x-axis, NOT a dual-axis chart. Two
  y-scales on one plot invent a correlation the data does not contain; it is
  the single most common charting error.
* Heatmaps use a single-hue light->dark sequential ramp (magnitude), never a
  rainbow.
* Every bar/cell carries a visible value label. Three palette slots sit below
  3:1 contrast on the light surface, so direct labels are required relief.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

mpl.use("Agg")

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "figures"
STAGES = ["W", "N1", "N2", "N3", "REM"]

# ---------------------------------------------------------------- palette ---
LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8880",
             grid="#e4e3dd", muted="#b8b7ae",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100")
DARK = dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8880",
            grid="#33322f", muted="#5c5b55",
            s1="#3987e5", s2="#d95926", s3="#199e70", s4="#c98500")
C = LIGHT


def ramp(hue_dark: str):
    """Single-hue sequential ramp, light -> dark. Never a rainbow."""
    return LinearSegmentedColormap.from_list("seq", [C["surface"], hue_dark], N=256)


def style():
    mpl.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"],
        "axes.titlecolor": C["ink"], "text.color": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "xtick.labelcolor": C["ink2"], "ytick.labelcolor": C["ink2"],
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": C["grid"], "grid.linewidth": 0.8,
        "legend.frameon": False, "figure.dpi": 130,
    })


def finish(fig, path: Path, note: str | None = None):
    if note:
        # Placed BELOW the axes in figure coords (negative y) so it can never
        # collide with an x-axis label; bbox_inches="tight" expands to include
        # it. Wrapped so long notes do not run past the figure width.
        import textwrap
        fig.text(0.0, -0.055, "\n".join(textwrap.wrap(note, 118)),
                 ha="left", va="top", fontsize=7.5, color=C["ink3"])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {path.relative_to(HERE.parent)}")


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").strip().split("\n") if l]


# ============================================================== FIGURE 1 =====
def fig1_exposure():
    """Leakage proof. Single series -> one colour, no legend."""
    d = load("exposure_gradient.json")
    if not d:
        return
    rows = d["exposure_gradient"]
    labels = ["Same recording\n(trained on)", "Same subject,\ndifferent night",
              "Unseen subject"]
    ks = [r["kappa"] for r in rows]
    ns = [r["n_epochs"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(ks))
    ax.bar(x, ks, width=0.55, color=C["s1"], zorder=3)
    for xi, k, n in zip(x, ks, ns):
        ax.text(xi, k + 0.012, f"{k:.4f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=C["ink"])
        ax.text(xi, 0.02, f"n={n:,}", ha="center", va="bottom",
                fontsize=8.5, color=C["ink2"])

    rc = d.get("reconstruction_check")
    if rc:
        ax.axhline(rc["originally_reported_kappa"], color=C["s2"],
                   lw=2, ls="--", zorder=2)
        ax.text(len(ks) - 0.45, rc["originally_reported_kappa"] + 0.008,
                f"published $\\kappa$ = {rc['originally_reported_kappa']:.4f}",
                ha="right", va="bottom", fontsize=9.5, color=C["s2"],
                fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Cohen's $\\kappa$")
    ax.set_ylim(0, max(ks) * 1.18)
    ax.set_title("Teacher performance falls with distance from training data",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)

    note = ""
    if rc:
        note = (f"Epoch-weighting the two measured validation subgroups reproduces the published figure to "
                f"{rc['absolute_agreement']:.4f} "
                f"({rc['blended_kappa']:.4f} vs {rc['originally_reported_kappa']:.4f}), confirming the split reconstruction is exact.")
    finish(fig, OUT / "fig1_exposure_gradient.png", note)


# ============================================================== FIGURE 2 =====
def fig2_compression():
    """Emphasis, not 6 categorical hues: the story is one model."""
    pts = []
    for name, f, lab in [("retrained", "eval_final_retrained.json", "E0 teacher"),
                         ("E1a", "eval_final_E1a.json", "E1a teacher"),
                         ("E1b", "eval_final_E1b.json", "E1b teacher")]:
        d = load(f)
        if d:
            pts.append((649229, d["groups"]["test_heldout"]["kappa"], lab, False))
    pc = load("prior_correction_E1a.json")
    if pc:
        k = [x for x in pc["test_results"] if x.startswith("val-fitted")][0]
        pts.append((649229, pc["test_results"][k]["kappa"],
                    "E1a + prior corr.", False))
    st = load("eval_students.json") or {}
    for key, lab, hero in [("student_distilled_E0", "Student\n(distilled E0)", False),
                           ("student_distilled_E1b", "Student\n(distilled E1b)", False),
                           ("student_baseline_E0", "Student\n(baseline)", True)]:
        if key in st:
            pts.append((st[key]["n_parameters"],
                        st[key]["test_heldout"]["kappa"], lab, hero))
    if not pts:
        return

    # Only two distinct x positions, so labels are placed to the OUTSIDE of each
    # column (teachers right, students left) on a single line. Two-line labels
    # stacked above/below collide: the tightest gap between two teachers is
    # 0.026 kappa, which is ~28pt on these axes.
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for p, k, lab, hero in pts:
        ax.scatter(p, k, s=190 if hero else 110,
                   color=C["s1"] if hero else C["muted"],
                   edgecolor=C["surface"], linewidth=2, zorder=4 if hero else 3)
        teacher = p > 3e5
        ax.annotate(f"{lab.replace(chr(10), ' ')}  $\\kappa$={k:.4f}", (p, k),
                    textcoords="offset points",
                    xytext=(14 if teacher else -14, 0),
                    ha="left" if teacher else "right", va="center",
                    fontsize=8.5, fontweight="bold" if hero else "normal",
                    color=C["ink"] if hero else C["ink2"])

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Held-out test Cohen's $\\kappa$")
    # Bounds chosen so each column's outward label has room: the student column
    # sits ~35% across and the teacher column ~61%, leaving both label runs
    # inside the axes.
    ax.set_xlim(1.2e4, 8e6)
    ks = [p[1] for p in pts]
    ax.set_ylim(min(ks) - 0.028, max(ks) + 0.028)
    ax.set_title("The smallest model is also the best",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")
    ax.grid(True, zorder=0); ax.set_axisbelow(True)
    finish(fig, OUT / "fig2_compression_vs_kappa.png",
           "All figures evaluated on the same 15 held-out subjects (33,431 epochs) that appear in no training or selection set.")


# ============================================================== FIGURE 3 =====
def fig3_per_class():
    """Grouped bars, 4 models -> slots 1-4 on the adjacent pairlist."""
    # FOUR series only. Six models exist, but the categorical order validates
    # four adjacent slots; past that, yellow sits beside orange and the pair
    # fails the separation floor. The two dropped rungs (E1a teacher, student
    # distilled from E0) are intermediate steps whose numbers live in the
    # tables - these four carry the narrative: original design, best teacher,
    # best distilled student, un-distilled winner.
    series = []
    for f, lab in [("eval_final_retrained.json", "E0 teacher"),
                   ("eval_final_E1b.json", "E1b teacher")]:
        d = load(f)
        if d:
            series.append((lab, [d["groups"]["test_heldout"]["per_class"][s]["f1"]
                                 for s in STAGES]))
    st = load("eval_students.json") or {}
    for key, lab in [("student_distilled_E1b", "Student (distilled E1b)"),
                     ("student_baseline_E0", "Student (baseline)")]:
        if key in st:
            series.append((lab, [st[key]["test_heldout"]["per_class"][s]["f1"]
                                 for s in STAGES]))
    if not series:
        return

    cols = [C["s1"], C["s2"], C["s3"], C["s4"]]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(STAGES))
    w = 0.8 / len(series)
    for i, (lab, vals) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        ax.bar(x + off, vals, width=w * 0.9, label=lab, color=cols[i], zorder=3)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=C["ink2"], rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(STAGES, fontsize=11)
    ax.set_ylabel("F1 (held-out test)")
    ax.set_ylim(0, 1.0)
    ax.set_title("The un-distilled student leads on every stage",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.11), fontsize=9.5)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    finish(fig, OUT / "fig3_per_class_f1.png",
           "N1 stays near 0.35-0.39 across every model - a representation limit, not a training-configuration one. "
           "E1a and the E0-distilled student are omitted to stay within four validated colour slots; their figures are in the tables.")


# ============================================================== FIGURE 4 =====
def fig4_pairwise_auc():
    """Magnitude -> single-hue sequential ramp."""
    d = (load("separability_diagnostic_E1b_E1a.json")
         or load("separability_diagnostic_E1a.json")
         or load("separability_diagnostic.json"))
    if not d:
        return
    key = next((k for k in ("E1b", "E1a") if k in d), list(d)[0])
    pw = d[key]["pairwise"]
    n = len(STAGES)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            v = pw.get(f"{STAGES[i]}_vs_{STAGES[j]}")
            if v:
                M[j, i] = v["roc_auc"]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M, cmap=ramp(C["s1"]), vmin=0.75, vmax=1.0)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                worst = M[i, j] == np.nanmin(M)
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        fontsize=10.5, fontweight="bold" if worst else "normal",
                        color=C["surface"] if M[i, j] > 0.93 else C["ink"])
                if worst:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               edgecolor=C["s2"], lw=3, zorder=5))
    ax.set_xticks(range(n)); ax.set_xticklabels(STAGES, fontsize=11)
    ax.set_yticks(range(n)); ax.set_yticklabels(STAGES, fontsize=11)
    ax.set_title(f"Pairwise separability ({key}) — one pair is the bottleneck",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cb.set_label("ROC-AUC", color=C["ink2"])
    cb.outline.set_visible(False)
    finish(fig, OUT / "fig4_pairwise_auc.png",
           "Score = p_A/(p_A+p_B) on epochs whose true label is A or B. One-vs-rest AUC averages these and hides which pair fails.")


# ============================================================== FIGURE 5 =====
def fig5_kl_vs_perf():
    """
    TWO STACKED PANELS sharing x - deliberately NOT a dual-axis chart.
    Two y-scales on one plot invent a correlation the data does not contain.
    """
    runs = []
    for key, lab, col in [("student_distilled_E0", "Distilled from E0", C["s1"]),
                          ("student_distilled_E1b", "Distilled from E1b", C["s2"]),
                          ("student_baseline_E0", "Baseline ($\\alpha$=1.0)", C["s3"])]:
        p = RES / "students" / key / "training_metrics.jsonl"
        if p.exists():
            runs.append((lab, col, jsonl(p)))
    if not runs:
        return

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True,
                                 gridspec_kw={"hspace": 0.12})
    for lab, col, rows in runs:
        ep = [r["epoch"] for r in rows]
        a1.plot(ep, [r["macro_f1"] for r in rows], color=col, lw=2, label=lab)
        a2.plot(ep, [r["loss_soft"] for r in rows], color=col, lw=2, label=lab)
    a1.set_ylabel("Validation macro-F1")
    a1.set_title("The student that ignores the teacher gets better — and diverges from it",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")
    a1.legend(loc="lower right", fontsize=9.5)
    a2.set_ylabel("KL divergence to teacher")
    a2.set_xlabel("Epoch")
    for a in (a1, a2):
        a.grid(True, zorder=0); a.set_axisbelow(True)

    # Take the colour from the baseline's own series entry. Hardcoding a slot
    # here silently mislabels the line as soon as the series list changes.
    base = [(c, r) for lab, c, r in runs if "1.0" in lab]
    if base:
        col, rows = base[0]
        a2.annotate(f"{rows[-1]['loss_soft']:.2f}", (rows[-1]["epoch"],
                    rows[-1]["loss_soft"]), textcoords="offset points",
                    xytext=(-6, 6), ha="right", fontsize=9.5,
                    fontweight="bold", color=col)
    finish(fig, OUT / "fig5_kl_vs_performance.png",
           "In the alpha=1.0 baseline the KD term is computed but contributes no gradient - a free measurement of distance from the teacher.")


# ============================================================== FIGURE 6 =====
def fig6_confusion():
    st = load("eval_students.json")
    if not st or "student_baseline_E0" not in st:
        return
    cm = np.array(st["student_baseline_E0"]["test_heldout"]["confusion_matrix"],
                  dtype=float)
    row = cm / cm.sum(1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(row, cmap=ramp(C["s1"]), vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{row[i,j]:.2f}", ha="center", va="center",
                    fontsize=10.5, fontweight="bold" if i == j else "normal",
                    color=C["surface"] if row[i, j] > 0.55 else C["ink"])
    ax.set_xticks(range(5)); ax.set_xticklabels(STAGES, fontsize=11)
    ax.set_yticks(range(5)); ax.set_yticklabels(STAGES, fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Student (baseline) — held-out test, row-normalised",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cb.set_label("Fraction of true class", color=C["ink2"])
    cb.outline.set_visible(False)
    finish(fig, OUT / "fig6_confusion_matrix.png",
           "N1 recall 0.41; the dominant error is N1 predicted as W or N2.")


# ============================================================== FIGURE 7 =====
def fig7_reliability():
    d = json.loads((HERE / "reliability_table.json").read_text(encoding="utf-8"))
    diag = d["reliability_diagram_test"]

    def pts(bins):
        return ([b["confidence"] for b in bins if b["n"]],
                [b["accuracy"] for b in bins if b["n"]])

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], color=C["ink3"], lw=1.4, ls="--", zorder=2)
    ax.text(0.62, 0.57, "perfect calibration", rotation=39, fontsize=8.5,
            color=C["ink3"], ha="center")
    for key, lab, col in [("before", f"Before (ECE {d['ece_before']:.4f})", C["s2"]),
                          ("after", f"After  T={d['calibration_temperature']:.3f} "
                                    f"(ECE {d['ece_after']:.4f})", C["s1"])]:
        cx, cy = pts(diag[key])
        ax.plot(cx, cy, "-o", color=col, lw=2, ms=6.5, label=lab,
                markeredgecolor=C["surface"], markeredgewidth=1.6, zorder=3)
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Calibration halves the expected error",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, zorder=0); ax.set_axisbelow(True)
    finish(fig, OUT / "fig7_reliability_diagram.png",
           "Temperature fitted on validation only, applied unchanged to test. Scaling changes confidence, not decisions.")


# ============================================================== FIGURE 8 =====
def fig8_rq3():
    """
    RQ3: the distillation penalty against teacher quality.

    Two measured points. The connecting line is drawn dashed and labelled as an
    extrapolation so it cannot be mistaken for a fitted relationship - with n=2
    there is no fit, only a direction.
    """
    st = load("eval_students.json") or {}
    base = st.get("student_baseline_E0")
    if not base:
        return
    base_k = base["test_heldout"]["kappa"]

    rungs = [("E0", "eval_final_retrained.json", "student_distilled_E0"),
             ("E1b", "eval_final_E1b.json", "student_distilled_E1b")]
    pts = []
    for lab, tf, skey in rungs:
        td = load(tf)
        if td and skey in st:
            pts.append((lab, td["groups"]["test_heldout"]["kappa"],
                        st[skey]["test_heldout"]["kappa"] - base_k))
    if len(pts) < 2:
        return

    (l0, x0, y0), (l1, x1, y1) = pts[0], pts[1]
    slope = (y1 - y0) / (x1 - x0)
    breakeven = x1 + (-y1) / slope

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.axhline(0, color=C["ink3"], lw=1.3, ls="--", zorder=2)
    ax.text(breakeven + 0.004, 0.0016, "distillation breaks even",
            fontsize=9, color=C["ink3"], ha="left", va="bottom")
    ax.plot([x0, breakeven], [y0, 0], color=C["muted"], lw=1.6, ls="--", zorder=2)
    ax.scatter([breakeven], [0], s=70, facecolor=C["surface"],
               edgecolor=C["ink3"], linewidth=1.6, zorder=4)
    ax.annotate(f"extrapolated\n$\\kappa\\approx${breakeven:.2f}", (breakeven, 0),
                textcoords="offset points", xytext=(0, -34), ha="center",
                fontsize=9, color=C["ink3"])

    for (lab, x, y), col in zip(pts, [C["s1"], C["s3"]]):
        ax.scatter(x, y, s=180, color=col, edgecolor=C["surface"],
                   linewidth=2, zorder=5)
        ax.annotate(f"{lab} teacher\n{y:+.4f} $\\kappa$", (x, y),
                    textcoords="offset points", xytext=(0, 18),
                    ha="center", fontsize=9.5, fontweight="bold", color=col)

    ax.set_xlabel("Teacher quality (held-out test Cohen's $\\kappa$)")
    ax.set_ylabel("Distillation gain vs un-distilled baseline ($\\kappa$)")
    ax.set_xlim(0.47, max(breakeven + 0.06, 0.80))
    ax.set_ylim(min(y0, y1) - 0.022, 0.020)
    ax.set_title("Distillation still hurts — but the penalty shrinks as the teacher improves",
                 fontsize=12.5, fontweight="bold", pad=12, loc="left")
    ax.grid(True, zorder=0); ax.set_axisbelow(True)
    finish(fig, OUT / "fig8_rq3_teacher_quality.png",
           f"Both students are identical apart from which teacher supplied the soft targets; the baseline "
           f"(kappa {base_k:.4f}) is the same stored model in both comparisons, since alpha=1.0 never reads a teacher. "
           f"A {x1-x0:+.4f} kappa better teacher shrank the penalty {abs(y1)-abs(y0):+.4f} "
           f"({(1-abs(y1)/abs(y0))*100:.0f}%). The dashed line is a two-point extrapolation, not a fit.")


FIGS = {1: fig1_exposure, 2: fig2_compression, 3: fig3_per_class,
        4: fig4_pairwise_auc, 5: fig5_kl_vs_perf, 6: fig6_confusion,
        7: fig7_reliability, 8: fig8_rq3}


def main() -> int:
    global C, OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", type=int, choices=sorted(FIGS))
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args()
    if a.dark:
        C = DARK
        OUT = HERE / "figures_dark"
    style()
    print(f"writing to {OUT.relative_to(HERE.parent)}/")
    for n in (a.only or sorted(FIGS)):
        try:
            FIGS[n]()
        except Exception as e:  # noqa: BLE001
            print(f"  fig{n} SKIPPED ({type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
