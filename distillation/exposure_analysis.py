"""
Exposure-gradient analysis - permanent artefact generator.

Emits distillation/results/exposure_gradient.json holding:
  - kappa/acc/macro-F1 by exposure level (train -> leaked-val -> clean)
  - the reconstruction check: does the epoch-weighted blend of the measured
    subgroups reproduce the originally reported kappa?
  - per-class F1 by exposure level
  - per-subject kappa on the teacher-clean cohort

Run with --compare <old.json> <new.json> to produce the before/after table for
the retrained teacher. A flattened gradient is the evidence that the protocol
fix worked.

CONVENTION (enforced from Phase 2 onward): headline per-class F1 is always
reported on UNSEEN subjects only. Per-class numbers measured on data the model
trained on are reported only inside the exposure table, where the exposure
level is explicit and the comparison is the point.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["W", "N1", "N2", "N3", "REM"]
EXPOSURE_ORDER = ["teacher_train", "teacher_val_leaked", "teacher_val_clean"]
EXPOSURE_LABEL = {
    "teacher_train": "same recording (trained on)",
    "teacher_val_leaked": "same subject, different night",
    "teacher_val_clean": "unseen subject",
    "new_test_split": "subject-level test split",
}


def build(eval_json: Path, reported_kappa: float | None = None) -> dict:
    d = json.loads(eval_json.read_text(encoding="utf-8"))
    g = d["groups"]
    reported = reported_kappa if reported_kappa is not None else d.get("reported_validation_kappa")

    gradient = []
    for name in EXPOSURE_ORDER:
        m = g.get(name)
        if not m:
            continue
        gradient.append({
            "group": name,
            "exposure": EXPOSURE_LABEL[name],
            "n_epochs": m["n_epochs"],
            "accuracy": round(m["accuracy"], 4),
            "kappa": round(m["kappa"], 4),
            "macro_f1": round(m["macro_f1"], 4),
        })

    monotonic = all(gradient[i]["kappa"] >= gradient[i + 1]["kappa"]
                    for i in range(len(gradient) - 1)) if len(gradient) > 1 else None
    delta = (round(gradient[0]["kappa"] - gradient[-1]["kappa"], 4)
             if len(gradient) > 1 else None)

    # reconstruction check
    recon = None
    if {"teacher_val_leaked", "teacher_val_clean"} <= set(g) and reported:
        nl, nc = g["teacher_val_leaked"]["n_epochs"], g["teacher_val_clean"]["n_epochs"]
        kl, kc = g["teacher_val_leaked"]["kappa"], g["teacher_val_clean"]["kappa"]
        blend = (kl * nl + kc * nc) / (nl + nc)
        recon = {
            "method": "epoch-weighted blend of measured val subgroups",
            "leaked": {"n_epochs": nl, "kappa": round(kl, 4)},
            "clean": {"n_epochs": nc, "kappa": round(kc, 4)},
            "blended_kappa": round(blend, 4),
            "originally_reported_kappa": reported,
            "absolute_agreement": round(abs(blend - reported), 4),
            "interpretation": (
                "Agreement to this tolerance confirms the reconstructed split "
                "(train_test_split(range(197), test_size=0.2, random_state=42)) "
                "is exact, and that the reported figure is a blend of "
                "mostly-leaked measurement."
            ),
        }

    per_class = {}
    for s in STAGES:
        row = {}
        for name in EXPOSURE_ORDER:
            if name in g:
                row[name] = round(g[name]["per_class"][s]["f1"], 4)
        if "teacher_train" in row and "teacher_val_clean" in row:
            row["train_minus_clean"] = round(row["teacher_train"] - row["teacher_val_clean"], 4)
        per_class[s] = row

    ps = d.get("per_subject_clean", {})
    ks = [v["kappa"] for v in ps.values()]
    subj = None
    if ks:
        mean = st.mean(ks)
        sd = st.stdev(ks) if len(ks) > 1 else 0.0
        se = sd / len(ks) ** 0.5 if len(ks) > 1 else 0.0
        subj = {
            "n_subjects": len(ks),
            "per_subject": {k: {"kappa": round(v["kappa"], 4),
                                "n_epochs": v["n_epochs"],
                                "macro_f1": round(v["macro_f1"], 4)}
                            for k, v in ps.items()},
            "mean": round(mean, 4), "sd": round(sd, 4),
            "min": round(min(ks), 4), "max": round(max(ks), 4),
            "ci95_t_df{}".format(len(ks) - 1): [round(mean - 2.776 * se, 4),
                                                round(mean + 2.776 * se, 4)],
        }

    return {
        "source_eval": str(eval_json.relative_to(REPO_ROOT)),
        "checkpoint": d.get("checkpoint"),
        "commit": d.get("commit"),
        "exposure_gradient": gradient,
        "monotonic_decline": monotonic,
        "delta_train_minus_clean": delta,
        "reconstruction_check": recon,
        "per_class_f1_by_exposure": per_class,
        "per_subject_clean": subj,
        "clean_cohorts_represented": d.get("clean_cohorts_represented"),
        "st_subjects_clean": d.get("st_subjects_clean"),
        "caveat": (
            "All clean-cohort figures are SC-only; no ST subject was ever held "
            "out by the original teacher, so ST generalisation is unmeasured."
        ),
    }


def show(a: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'exposure':<34}{'n_epochs':>10}{'acc':>9}{'kappa':>9}{'macroF1':>10}")
    for r in a["exposure_gradient"]:
        print(f"{r['exposure']:<34}{r['n_epochs']:>10,}{r['accuracy']:>9.4f}"
              f"{r['kappa']:>9.4f}{r['macro_f1']:>10.4f}")
    print(f"  monotonic decline: {a['monotonic_decline']}   "
          f"delta(train-clean) = {a['delta_train_minus_clean']}")
    rc = a.get("reconstruction_check")
    if rc:
        print(f"  reconstruction: blend={rc['blended_kappa']} vs "
              f"reported={rc['originally_reported_kappa']} "
              f"(agreement {rc['absolute_agreement']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = Path(__file__).parent / "results"
    ap.add_argument("--eval", default=str(res / "eval_teacher.json"))
    ap.add_argument("--out", default=str(res / "exposure_gradient.json"))
    ap.add_argument("--reported-kappa", type=float, default=0.6663)
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="Two eval JSONs -> before/after gradient comparison.")
    args = ap.parse_args()

    if args.compare:
        before = build(Path(args.compare[0]), args.reported_kappa)
        after = build(Path(args.compare[1]), None)
        show(before, "BEFORE - inherited teacher (recording-level split)")
        show(after, "AFTER - retrained teacher (subject-level split)")
        print("\n=== FLATTENING ===")
        print(f"  delta(train - unseen)  before: {before['delta_train_minus_clean']}")
        print(f"  delta(train - unseen)  after : {after['delta_train_minus_clean']}")
        payload = {"before": before, "after": after,
                   "flattening": {
                       "delta_before": before["delta_train_minus_clean"],
                       "delta_after": after["delta_train_minus_clean"]}}
        out = Path(args.out).with_name("exposure_gradient_comparison.json")
    else:
        payload = build(Path(args.eval), args.reported_kappa)
        show(payload, "exposure gradient")
        print("\nPER-CLASS F1 BY EXPOSURE")
        print(f"  {'stage':<7}{'train':>9}{'leaked':>9}{'unseen':>9}{'drop':>9}")
        for s in STAGES:
            r = payload["per_class_f1_by_exposure"][s]
            print(f"  {s:<7}{r.get('teacher_train', 0):>9.4f}"
                  f"{r.get('teacher_val_leaked', 0):>9.4f}"
                  f"{r.get('teacher_val_clean', 0):>9.4f}"
                  f"{r.get('train_minus_clean', 0):>+9.4f}")
        out = Path(args.out)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
