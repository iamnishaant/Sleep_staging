"""
How trustworthy is each derived sleep metric, given that the stages feeding it
are model predictions rather than expert scoring?

WHY THIS EXISTS
---------------
`physiological_features.extract_features` is exact arithmetic on whatever stage
sequence it is handed. Hand it a perfect hypnogram and every metric is correct.
Hand it a model's output and the errors do not distribute evenly:

  Sleep efficiency        aggregates over the whole night   -> robust
  REM latency             depends on ONE epoch, the first R -> catastrophic

Measured on a single test night, a single spurious REM epoch at position 66
collapsed REM latency from 119.5 minutes to 5.5. A latency under 15 minutes is
a clinical red flag for narcolepsy, so an unqualified packet would hand a
report generator a false alarm and no way to know.

This module measures each metric's error on the VALIDATION split - never test,
consistent with temperature scaling and the night-confidence tiers - and
classifies it. build_packet.py consumes the result so every derived value
travels with its own error bound.

MEASURED ON THE DECODED HYPNOGRAM
---------------------------------
The stage sequence comes from `sequence_decode.load_decoder()`, not from
`probs.argmax(1)`. That is the sequence the packet ships, so it is the sequence
whose derived error must be measured; measuring argmax while shipping decoded
would describe a model that is not the one being used.

CLASSIFICATION
    robust      relative error <= 10%, or absolute error small against clinical
                decision thresholds
    fragile     10-40% - usable with the error bound stated alongside
    unreliable  > 40%, or a known catastrophic failure mode - must not be
                asserted without an explicit caveat

Two metrics carry a CLINICAL override that measured error alone cannot express:
REM_Latency and REM_Periods. The override is driven by evidence measured here
(the false-SOREMP count on validation), not by a hard-coded sentence - an
earlier version stated a held-out-test figure inside an artefact that claims to
be validation-measured, which is both a leak and, after decoding, false.

USAGE
    python distillation/sequence_decode.py     # must run first
    python distillation/metric_reliability.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).parent / "results"
STAGES = ["W", "N1", "N2", "N3", "REM"]
TO_PHYSIO = {"W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "R"}

sys.path.insert(0, str(REPO_ROOT / "code" / "Phase2_47"))
from physiological_features import extract_features            # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from sequence_decode import calibrate, load_decoder            # noqa: E402

# Metrics reported in minutes; everything else is a count, ratio or fraction.
IN_SECONDS = {"Total_Sleep_Time", "Total_Time_In_Bed", "Sleep_Onset_Latency",
              "Wake_After_Sleep_Onset", "REM_Latency", "Mean_Segment_Duration",
              "W_Duration", "N1_Duration", "N2_Duration", "N3_Duration", "R_Duration"}


def robust_rem_latency(seq: list[str], min_run: int = 3) -> float | None:
    """
    REM latency measured to the first SUSTAINED REM period rather than the first
    single epoch.

    Clinical scoring treats an isolated REM epoch surrounded by non-REM as
    noise; requiring 3 consecutive epochs (90 s) is the smallest window that
    survives one misclassification. Returns seconds from sleep onset, or None
    if no sustained REM period exists.
    """
    onset = next((i for i, s in enumerate(seq) if s != "W"), None)
    if onset is None:
        return None
    i = 0
    while i < len(seq):
        if seq[i] == "REM":
            j = i
            while j < len(seq) and seq[j] == "REM":
                j += 1
            if j - i >= min_run:
                return (i - onset) * 30.0
            i = j
        else:
            i += 1
    return None


def classify(name: str, rel_err: float, evidence: dict) -> tuple[str, str]:
    """
    Returns (tier, reason).

    Clinical significance can override the raw percentage, but only on evidence
    measured in this run. `evidence` carries the validation-split facts the
    overrides depend on, so a reason string can never drift away from what was
    actually measured.
    """
    if name == "REM_Latency":
        n_sor, n_rec = evidence["false_soremp"], evidence["n_recordings"]
        if n_sor > 0:
            return ("unreliable",
                    f"depends on a single epoch. Even on the decoded hypnogram it "
                    f"produced a false sleep-onset-REM reading - a narcolepsy red flag - "
                    f"on {n_sor} of {n_rec} validation nights "
                    f"(mean relative error {rel_err:.0%}).")
        return ("fragile",
                f"single-event metric, but no false sleep-onset-REM reading occurred "
                f"on any of {n_rec} validation nights after decoding "
                f"(mean relative error {rel_err:.0%}).")
    if name == "REM_Periods":
        ratio = evidence["ratio"].get(name)
        return (("fragile" if rel_err <= 0.40 else "unreliable"),
                f"counts contiguous runs, so it tracks prediction fragmentation as "
                f"well as physiology. Decoding moved it from "
                f"{evidence['argmax_ratio'].get(name, float('nan')):.2f}x to "
                f"{ratio:.2f}x of the expert count; mean relative error {rel_err:.0%}.")
    if name in ("Stage_Transitions", "Transition_Rate", "Wake_Interruptions_per_Hour",
                "Stage_Transition_Entropy", "Segment_Duration_Variance",
                "Mean_Segment_Duration"):
        tier = "fragile" if rel_err <= 0.40 else "unreliable"
        return (tier, f"derived from run structure, which inherits per-epoch noise; "
                      f"mean relative error {rel_err:.0%} on validation.")
    if rel_err <= 0.10:
        return ("robust", "aggregates over the whole night; per-epoch errors average out.")
    if rel_err <= 0.40:
        return ("fragile", f"mean relative error {rel_err:.0%} on validation.")
    return ("unreliable", f"mean relative error {rel_err:.0%} on validation.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="student_baseline_E0")
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    args = ap.parse_args()

    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    recs = sorted(r for s in sp["splits"]["val"] for r in sp["recordings_by_subject"][s])
    cache = RES / f"probs_{args.model}_val"
    if not cache.exists():
        raise SystemExit(f"Missing {cache}. Run evaluate_student.py first.")

    # The hypnogram the packet ships is decoded, not argmaxed, so that is the
    # sequence whose derived error must be measured. load_decoder raises rather
    # than falling back to argmax - a silent fallback here would mis-state which
    # values are safe to assert, which is the one thing this module exists to
    # get right.
    decode_fn = load_decoder()
    T = decode_fn.artefact["calibration_temperature"]
    print(f"decoder    : {decode_fn.artefact['selected_label']}  {decode_fn.spec}")
    print(f"calibration: T = {T} (decoding runs on calibrated probabilities)")
    print(f"measuring derived-metric error on the VALIDATION split "
          f"({len(recs)} recordings)\n")

    acc: dict[str, list[tuple[float, float]]] = {}
    argmax_acc: dict[str, list[tuple[float, float]]] = {}
    rob_pairs = []
    n_sorem = n_changed = n_epochs_total = n_identical = 0
    for r in recs:
        f = cache / f"{r}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        Pc = calibrate(d["probs"], T)
        idx = decode_fn(Pc)
        arg = Pc.argmax(1)
        n_changed += int((idx != arg).sum())
        n_epochs_total += len(idx)

        pred = [STAGES[i] for i in idx]
        true = [STAGES[i] for i in d["labels"]]
        mp = extract_features([TO_PHYSIO[s] for s in pred])
        mt = extract_features([TO_PHYSIO[s] for s in true])
        if mp is None or mt is None:
            continue
        for k in mt:
            acc.setdefault(k, []).append((float(mp[k]), float(mt[k])))

        # argmax kept alongside purely so the reason strings can state what
        # decoding actually changed, rather than asserting it
        ma = extract_features([TO_PHYSIO[STAGES[i]] for i in arg])
        if ma is not None:
            for k in mt:
                argmax_acc.setdefault(k, []).append((float(ma[k]), float(mt[k])))

        # a false sleep-onset-REM period: predicted REM latency under 15 min
        # when the expert's is over 30. This is the clinical failure mode that
        # keeps REM_Latency out of the assertable set.
        lp, lt = mp["REM_Latency"] / 60.0, mt["REM_Latency"] / 60.0
        if lp >= 0 and lt >= 0 and lp < 15 and lt > 30:
            n_sorem += 1

        a, b = robust_rem_latency(pred), robust_rem_latency(true)
        if a is not None and b is not None:
            rob_pairs.append((a, b))
            # does the sustained definition still say anything the naive one does not?
            if abs(a - mp["REM_Latency"]) < 1e-6:
                n_identical += 1

    def _ratios(store):
        out = {}
        for k, ps in store.items():
            p = np.array([x for x, _ in ps])
            t = np.array([y for _, y in ps])
            out[k] = float(p.mean() / max(abs(t.mean()), 1e-9))
        return out

    evidence = {"false_soremp": n_sorem, "n_recordings": len(rob_pairs) or len(recs),
                "ratio": _ratios(acc), "argmax_ratio": _ratios(argmax_acc)}
    print(f"decoding changed {n_changed:,} of {n_epochs_total:,} epochs "
          f"({100 * n_changed / max(n_epochs_total, 1):.2f}%)")
    print(f"false sleep-onset-REM readings on the decoded hypnogram: "
          f"{n_sorem} of {len(recs)} validation nights\n")

    out = {"model": args.model, "measured_on": "validation split",
           "n_recordings": len(recs),
           "purpose": "Derived metrics are exact arithmetic on the stage sequence. "
                      "When that sequence comes from a model, error does not "
                      "distribute evenly - aggregates survive, single-event "
                      "metrics do not.",
           "hypnogram_source": {
               "decoder": decode_fn.artefact["selected_label"],
               "spec": decode_fn.spec,
               "calibration_temperature": T,
               "epochs_changed_vs_argmax": n_changed,
               "epochs_changed_pct": round(100 * n_changed / max(n_epochs_total, 1), 2),
               "note": "Error is measured on the DECODED hypnogram, which is the one "
                       "the packet ships. Measuring argmax while shipping decoded would "
                       "describe a different model.",
           },
           "false_soremp_validation_nights": n_sorem,
           "tiers": {"robust": "<=10% relative error; safe to assert",
                     "fragile": "10-40%; assert only with the error bound",
                     "unreliable": ">40% or a known catastrophic mode; must carry an explicit caveat"},
           "metrics": {}}

    # argmax error, measured the same way, so every tier change is attributable
    argmax_rel = {}
    for k, ps in argmax_acc.items():
        p = np.array([x for x, _ in ps]); t = np.array([y for _, y in ps])
        argmax_rel[k] = float(np.mean(np.abs(p - t))) / (float(np.mean(np.abs(t))) or 1.0)

    print(f"{'metric':<30}{'pred mean':>11}{'true mean':>11}{'mean abs err':>14}"
          f"{'rel':>8}{'argmax':>9}  tier")
    for k, pairs in sorted(acc.items()):
        p = np.array([x for x, _ in pairs])
        t = np.array([y for _, y in pairs])
        ae = float(np.mean(np.abs(p - t)))
        denom = float(np.mean(np.abs(t))) or 1.0
        re_ = ae / denom
        tier, reason = classify(k, re_, evidence)
        d = 60.0 if k in IN_SECONDS else 1.0
        arg_re = argmax_rel.get(k)
        out["metrics"][k] = {
            "tier": tier, "reason": reason,
            "mean_abs_error": round(ae / d, 4),
            "mean_relative_error": round(re_, 4),
            "mean_relative_error_argmax": round(arg_re, 4) if arg_re is not None else None,
            "predicted_mean": round(float(p.mean()) / d, 4),
            "expert_mean": round(float(t.mean()) / d, 4),
            "predicted_over_expert": round(float(p.mean() / max(abs(t.mean()), 1e-9)), 4),
            "unit": "minutes" if k in IN_SECONDS else "count_or_ratio",
        }
        arrow = "" if arg_re is None else ("  better" if re_ < arg_re - 1e-9
                                           else ("  worse" if re_ > arg_re + 1e-9 else ""))
        print(f"{k:<30}{p.mean()/d:>11.2f}{t.mean()/d:>11.2f}{ae/d:>14.2f}{re_:>8.0%}"
              f"{(arg_re if arg_re is not None else 0):>9.0%}  {tier}{arrow}")

    if rob_pairs:
        a = np.array([x for x, _ in rob_pairs]); b = np.array([y for _, y in rob_pairs])
        ae = float(np.mean(np.abs(a - b))) / 60
        naive = out["metrics"]["REM_Latency"]["mean_abs_error"]
        # A REM-targeted minimum-run decode with L>=3 SUBSUMES this mitigation.
        # The decoder has already removed every REM run shorter than L, so
        # "first REM epoch" and "first sustained REM period" resolve to the same
        # epoch - verified below on every recording rather than assumed.
        #
        # The two error figures are nevertheless NOT comparable, and reporting
        # them side by side without saying so would be the more misleading
        # option: the naive figure scores the prediction against the expert's
        # first REM epoch, this one against the expert's first sustained REM
        # period. Same prediction, different reference, so the larger number
        # here is a stricter target, not a regression.
        spec = decode_fn.spec
        subsumed = (spec.get("method") == "min_run"
                    and "REM" in spec.get("targets", [])
                    and spec.get("length", 0) >= 3)
        rel_rr = ae / max(abs(float(np.mean(b))) / 60, 1e-9)
        out["robust_rem_latency"] = {
            "definition": "time from sleep onset to the first run of >=3 consecutive REM epochs",
            "rationale": "clinical scoring treats an isolated REM epoch as noise; "
                         "3 epochs (90 s) is the smallest window surviving one misclassification",
            "mean_abs_error_min": round(ae, 2),
            "mean_relative_error": round(rel_rr, 4),
            "tier": "fragile" if rel_rr <= 0.40 else "unreliable",
            "n_recordings": len(rob_pairs),
            "predicted_value_identical_to_naive": bool(subsumed and n_identical == len(rob_pairs)),
            "n_recordings_predicted_identical": int(n_identical),
            "not_comparable_to_naive": {
                "naive_mean_abs_error_min": naive,
                "why": "the naive figure is scored against the expert's first REM epoch, "
                       "this one against the expert's first SUSTAINED REM period. The "
                       "prediction is the same sequence in both cases; only the reference "
                       "differs, so the two errors cannot be differenced.",
            },
            "subsumed_by_decoder": bool(subsumed),
            "subsumed_note": (
                f"The decoder enforces a minimum REM run of {spec.get('length')} epochs, so "
                f"this metric no longer mitigates anything - its predicted value is "
                f"identical to the naive metric's on {n_identical} of {len(rob_pairs)} "
                f"validation recordings. Retained because evidence-item ids are a "
                f"permanent contract, but it should be read as a duplicate, not as a "
                f"second opinion." if subsumed else None),
        }
        print(f"\nrobust REM latency (>=3-epoch run): mean abs error {ae:.1f} min "
              f"({rel_rr:.0%} relative)")
        if subsumed:
            print(f"  SUBSUMED by the decoder: predicted value identical to the naive "
                  f"metric on {n_identical}/{len(rob_pairs)} recordings.")
            print(f"  Not comparable to the naive {naive:.1f} min - same prediction, "
                  f"different reference (expert sustained vs expert first REM epoch).")

    counts = {}
    for m in out["metrics"].values():
        counts[m["tier"]] = counts.get(m["tier"], 0) + 1
    print(f"\ntiers: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    p = RES / "derived_metric_reliability.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {p.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
