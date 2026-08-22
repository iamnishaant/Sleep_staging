"""
Evidence packet v0 - one JSON per night, from artefacts that already exist.

This is M1 minus attributions. It is the INTERFACE CONTRACT between the staging
model and everything downstream (report generation, verification, the UI), so
the schema matters more than the code: once the vertical slice is written
against it, changing a field means changing every consumer.

Fields that are not yet populated are present and null rather than absent, so
adding them later is not a breaking change. Two are reserved this way:
`attribution` (pending Gate 3a) and `risk` (pending disorder-detection
integration).

WHAT GOES IN, AND WHERE IT COMES FROM
-------------------------------------
  stage probabilities, hypnogram   cached predictions from evaluate_student.py
  calibrated probabilities         temperature 1.3347, fitted on validation
  derived sleep metrics            code/Phase2_47/physiological_features.py
  per-stage confidence + tier      distillation/reliability_table.json
  night-level confidence + tier    distillation/results/night_confidence.json
  evidence_items                   assembled here, with stable ids
  attribution                      null - Gate 3a has not run
  risk                             null - not integrated

THREE THINGS THAT WOULD OTHERWISE BITE
--------------------------------------
1. `physiological_features.extract_features` expects stage strings using **R**
   for REM, not "REM". Passing our vocabulary straight in silently yields
   REM_Duration = 0 and REM_Latency = -1 for every night, with no error. The
   mapping is applied here and asserted.

2. Every packet carries a `provenance` block with the model checkpoint's
   sha256 and the git commit. Two silent bugs have already been found in this
   pipeline (a leaking split, an inert input branch); if a third turns up, the
   affected packets must be identifiable without re-deriving which run produced
   them.

3. THE HYPNOGRAM IS NO LONGER THE ARGMAX OF THE SHIPPED PROBABILITIES.
   Schema 1.1 decodes it (see sequence_decode.py). Temperature scaling never
   moved a decision, so through schema 1.0 a consumer could argmax the
   probabilities and land on the hypnogram. That is now false, and silently
   false is the dangerous kind - so the packet carries an explicit `decoding`
   block giving the exact spec, and asserts at build time that re-running the
   decoder on the packet's OWN shipped probabilities reproduces its OWN shipped
   hypnogram. That is the s5 self-consistency rule applied to decisions rather
   than to entropy.

USAGE
    python distillation/sequence_decode.py               # must run first
    python distillation/build_packet.py                  # all 29 test nights
    python distillation/build_packet.py --limit 1        # one, to inspect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).parent / "results"
STAGES = ["W", "N1", "N2", "N3", "REM"]

SCHEMA_VERSION = "1.1"

# 1.0 -> 1.1
#   ADDED    `decoding` (object) - how the hypnogram was produced from the
#            probabilities, and the spec needed to reproduce it.
#   CHANGED  `hypnogram.predicted_stages` is now the decoder's output, not
#            argmax of `probabilities.calibrated`. No field was removed or
#            renamed, so a 1.0 consumer still parses a 1.1 packet - but one that
#            recomputes the hypnogram by argmax will now disagree with it, which
#            is exactly why the version moved.
SCHEMA_CHANGELOG = {
    "1.1": {
        "added": ["decoding"],
        "changed": ["hypnogram.predicted_stages is decoded, not argmax",
                    "derived_metrics and evidence_items follow from the decoded hypnogram",
                    "per_stage counts and confidences follow from the decoded hypnogram"],
        "breaking_for": "consumers that recompute the hypnogram as "
                        "probabilities.calibrated.argmax(axis=1)",
    },
}

# physiological_features.py uses R for REM. Ours uses REM. This mismatch is
# silent - it returns zeros rather than raising - so it is mapped explicitly.
TO_PHYSIO = {"W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "R"}

sys.path.insert(0, str(REPO_ROOT / "code" / "Phase2_47"))
from physiological_features import extract_features           # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from sequence_decode import load_decoder                      # noqa: E402


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO_ROOT).stdout.strip() or "unknown"
    except Exception:                                          # noqa: BLE001
        return "unknown"


def calibrate(P: np.ndarray, T: float) -> np.ndarray:
    """
    Temperature scaling. Softmax is invariant to an additive constant, so log(p)
    is a valid logit representation and the missing offset does not matter.

    Note this changes CONFIDENCE, not decisions - argmax is unaffected - so the
    hypnogram is identical before and after. Asserted below.
    """
    z = np.log(np.clip(P, 1e-12, None)) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def robust_rem_latency(seq: list[str], min_run: int = 3) -> float | None:
    """
    REM latency to the first SUSTAINED REM period, in seconds from sleep onset.

    The naive definition uses the first single REM epoch, which one
    misclassification can collapse - measured at 114 minutes of error on one
    held-out night, and a false sleep-onset-REM reading on 4 of 29. Requiring 3
    consecutive epochs is the smallest window that survives one bad epoch.
    Kept identical to the implementation in metric_reliability.py.
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


def runs_of(seq: list[str]) -> list[dict]:
    """Compress the hypnogram to contiguous stage runs - what a reader needs."""
    out, i = [], 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        out.append({"stage": seq[i], "start_epoch": i, "n_epochs": j - i,
                    "start_sec": i * 30, "duration_sec": (j - i) * 30})
        i = j
    return out


MR: dict = {"metrics": {}}          # populated in main() from the artefact
DEC = None                          # decode fn, populated in main()


def build(rec: str, rel: dict, nc: dict, prov: dict, cache: Path) -> dict | None:
    d = np.load(cache / f"{rec}.npz")
    P, y = d["probs"], d["labels"]
    T = prov["calibration_temperature"]
    Pc = calibrate(P, T)

    # Temperature scaling must not change decisions. If it does, something is
    # wrong with the calibration and every confidence below is untrustworthy.
    # This is about SCALING only - decoding changes decisions deliberately.
    assert (Pc.argmax(1) == P.argmax(1)).all(), "calibration changed argmax"

    argmax_pred = Pc.argmax(1)
    pred = DEC(Pc)

    # SELF-CONSISTENCY (EVIDENCE_PACKET.md s5, applied to decisions).
    # The packet ships `probabilities.calibrated` rounded to 5 decimal places.
    # A consumer reconstructs the hypnogram from THAT, not from full-precision
    # floats, so the reproduction has to be checked against the rounded values -
    # otherwise a packet could pass here and still be irreproducible in the
    # hands of the consumer it was written for.
    Pc_shipped = np.round(Pc, 5)
    assert (DEC(Pc_shipped) == pred).all(), (
        f"{rec}: decoding the packet's own rounded probabilities does not "
        f"reproduce its hypnogram")

    pred_stages = [STAGES[i] for i in pred]
    n_changed = int((pred != argmax_pred).sum())
    ent = -(Pc * np.log(np.clip(Pc, 1e-12, None))).sum(1)
    conf = Pc.max(1)

    # ---- derived sleep metrics (R, not REM - see module docstring) ----------
    physio_in = [TO_PHYSIO[s] for s in pred_stages]
    assert "REM" not in physio_in, "REM must be mapped to R before extract_features"
    metrics = extract_features(physio_in)
    if metrics is None:
        return None                                            # night with no sleep

    rrl = robust_rem_latency(pred_stages)

    n = len(pred_stages)
    lo = nc["tier_boundaries_nats"]["high_max"]
    hi = nc["tier_boundaries_nats"]["medium_max"]
    night_H = float(np.mean(ent))
    night_tier = "high" if night_H <= lo else ("medium" if night_H <= hi else "low")

    # ---- per-stage summary -------------------------------------------------
    per_stage = {}
    for i, s in enumerate(STAGES):
        m = pred == i
        per_stage[s] = {
            "n_epochs_predicted": int(m.sum()),
            "fraction_of_night": round(float(m.mean()), 4),
            "mean_calibrated_confidence": round(float(conf[m].mean()), 4) if m.any() else None,
            "model_reliability_tier": rel["per_class"][s]["reliability"],
            "model_f1_on_unseen_subjects": rel["per_class"][s]["f1"],
        }

    # ---- evidence items ----------------------------------------------------
    # Stable, namespaced ids: a downstream claim cites `arch.sleep_efficiency`
    # and that id must mean the same thing in every packet, forever.
    #
    # assertion_level is a property of the DATA, not of prompt wording:
    #   factual           a measured quantity or a model output
    #   associative_only  a population-level association; must never be phrased
    #                     as a statement about this individual
    # Everything here is currently factual. The field exists so that risk
    # evidence, when integrated, arrives already marked.
    #
    # metric_reliability is separate and equally load-bearing. A value can be
    # factual in kind and still not safe to state: extract_features is exact
    # arithmetic, but it is arithmetic on MODEL OUTPUT, and single-event metrics
    # inherit per-epoch errors catastrophically. Measured on validation in
    # metric_reliability.py; `safe_to_assert` is the field a report generator
    # should branch on.
    ev = []

    def item(eid, label, value, unit, level="factual", metric=None, **extra):
        e = {"id": eid, "label": label, "value": value, "unit": unit,
             "assertion_level": level}
        if metric and metric in MR["metrics"]:
            m = MR["metrics"][metric]
            e["metric_reliability"] = m["tier"]
            e["safe_to_assert"] = m["tier"] == "robust"
            e["mean_abs_error"] = m["mean_abs_error"]
            e["error_unit"] = m["unit"]
            e["error_measured_on"] = "validation split"
            if m["tier"] != "robust":
                e["caveat"] = m["reason"]
        else:
            e["safe_to_assert"] = True
        ev.append({**e, **extra})

    item("arch.total_sleep_time", "Total sleep time",
         round(metrics["Total_Sleep_Time"] / 60, 1), "minutes", metric="Total_Sleep_Time")
    item("arch.time_in_bed", "Time in bed",
         round(metrics["Total_Time_In_Bed"] / 60, 1), "minutes", metric="Total_Time_In_Bed")
    item("arch.sleep_efficiency", "Sleep efficiency",
         round(metrics["Sleep_Efficiency"], 4), "fraction", metric="Sleep_Efficiency")
    item("arch.sleep_onset_latency", "Sleep onset latency",
         round(metrics["Sleep_Onset_Latency"] / 60, 1), "minutes", metric="Sleep_Onset_Latency")
    item("arch.waso", "Wake after sleep onset",
         round(metrics["Wake_After_Sleep_Onset"] / 60, 1), "minutes",
         metric="Wake_After_Sleep_Onset")
    item("arch.rem_latency", "REM latency",
         round(metrics["REM_Latency"] / 60, 1) if metrics["REM_Latency"] >= 0 else None,
         "minutes", metric="REM_Latency",
         note=None if metrics["REM_Latency"] >= 0 else "no REM detected")
    # The sustained variant has its OWN measured error - inheriting the naive
    # metric's bound would misreport it in both directions.
    #
    # It is also, since schema 1.1, SUBSUMED: the decoder already enforces a
    # minimum REM run, so this resolves to the same epoch as arch.rem_latency on
    # every validation recording. The id stays because evidence-item ids are a
    # permanent contract, but it now carries `duplicates` so a report generator
    # does not present one number twice as if it were corroboration.
    _rr = MR.get("robust_rem_latency", {})
    _subsumed = bool(_rr.get("subsumed_by_decoder"))
    ev.append({
        "id": "arch.rem_latency_sustained",
        "label": "REM latency to first sustained REM period",
        "value": round(rrl / 60, 1) if rrl is not None else None,
        "unit": "minutes",
        "assertion_level": "factual",
        "metric_reliability": _rr.get("tier", "fragile"),
        "safe_to_assert": _rr.get("tier") == "robust",
        "mean_abs_error": _rr.get("mean_abs_error_min"),
        "error_unit": "minutes",
        "error_measured_on": "validation split",
        "definition": _rr.get("definition",
                              "time from sleep onset to the first run of 3 or more "
                              "consecutive REM epochs"),
        "duplicates": "arch.rem_latency" if _subsumed else None,
        "caveat": (
            "The decoder already enforces a minimum REM run, so this resolves to the "
            "same epoch as arch.rem_latency and is not an independent check. Its error "
            f"({_rr.get('mean_abs_error_min')} min) is measured against the expert's "
            f"first SUSTAINED REM period while arch.rem_latency's "
            f"({MR['metrics']['REM_Latency']['mean_abs_error']} min) is measured against "
            "the expert's first REM epoch - different references, so the two numbers "
            "must not be differenced. Both remain large against clinical thresholds."
        ) if _subsumed else (
            f"more stable than the single-epoch definition "
            f"({_rr.get('mean_abs_error_min')} vs "
            f"{MR['metrics']['REM_Latency']['mean_abs_error']} min mean error), "
            f"but still large relative to clinical decision thresholds."
        ),
        "note": None if rrl is not None else "no sustained REM period detected",
    })
    item("arch.rem_periods", "REM periods", int(metrics["REM_Periods"]), "count",
         metric="REM_Periods")
    item("arch.stage_transitions", "Stage transitions",
         int(metrics["Stage_Transitions"]), "count", metric="Stage_Transitions")
    item("arch.transition_rate", "Transition rate",
         round(metrics["Transition_Rate"], 3), "per hour", metric="Transition_Rate")
    item("arch.light_deep_ratio", "Light-to-deep sleep ratio",
         round(metrics["Light_Deep_Ratio"], 3), "ratio", metric="Light_Deep_Ratio")
    item("arch.wake_interruptions_per_hour", "Wake interruptions",
         round(metrics["Wake_Interruptions_per_Hour"], 3), "per hour",
         metric="Wake_Interruptions_per_Hour")
    for s in STAGES:
        item(f"stage.{s}.fraction", f"{s} as a fraction of the night",
             per_stage[s]["fraction_of_night"], "fraction",
             metric=f"{TO_PHYSIO[s]}_Proportion",
             model_reliability=per_stage[s]["model_reliability_tier"])
    item("night.confidence", "Night-level confidence", night_tier, "tier",
         mean_entropy_nats=round(night_H, 4),
         basis="prediction entropy; Spearman rho -0.75 against per-recording kappa on held-out data")
    item("model.n1_reliability_warning", "N1 is low-reliability in this model",
         rel["per_class"]["N1"]["reliability"], "tier",
         model_f1=rel["per_class"]["N1"]["f1"],
         basis="representation-bound: no decision threshold improves N1 by more than +0.0086")

    return {
        "schema_version": SCHEMA_VERSION,
        "recording_id": rec,
        "subject_id": rec[:5],
        "cohort": rec[:2],
        "n_epochs": n,
        "epoch_seconds": 30,
        "recording_duration_min": round(n * 30 / 60, 1),

        "provenance": prov,

        "hypnogram": {
            "predicted_stages": pred_stages,
            "stage_runs": runs_of(pred_stages),
            "n_runs": len(runs_of(pred_stages)),
            "source": "decoded - see the `decoding` block. NOT the argmax of "
                      "`probabilities.calibrated`.",
        },

        # How the hypnogram was produced from the probabilities. Everything a
        # consumer needs to reproduce it is here plus `probabilities.calibrated`;
        # build_packet asserts that reproduction on the shipped rounded values.
        "decoding": {
            "applied": True,
            "method": DEC.spec["method"],
            "spec": DEC.spec,
            "description": DEC.artefact["selected_label"],
            "input": "probabilities.calibrated",
            "changes_decisions": True,
            "epochs_changed_vs_argmax": n_changed,
            "epochs_changed_pct": round(100.0 * n_changed / max(n, 1), 2),
            "reproducible_from_this_packet": True,
            "why": "The model decides each 30-second epoch independently, which "
                   "sprays isolated REM epochs into non-REM sleep: REM periods came "
                   "out at 2.01x the expert count while total transitions were only "
                   "1.14x. A REM-targeted minimum-run rule removes those epochs and "
                   "the spurious transitions they created.",
            "selection": DEC.artefact["selection"]["objective"],
            "fitted_on": DEC.artefact["selection"]["fitted_on"],
            "artefact": "distillation/results/sequence_decoding.json",
            "consumer_note": "Through schema 1.0 the hypnogram was the argmax of these "
                             "probabilities. From 1.1 it is not. Recomputing it by argmax "
                             "will disagree with `hypnogram.predicted_stages`.",
        },

        "probabilities": {
            "stages": STAGES,
            "calibrated": [[round(float(v), 5) for v in row] for row in Pc],
            "calibration_temperature": T,
            "calibration_note": "Temperature scaling changes confidence, not decisions; "
                                "argmax is identical before and after (asserted at build "
                                "time). The DECODER, by contrast, does change decisions - "
                                "see the `decoding` block.",
            "argmax_stages_note": "argmax of these probabilities is NOT the shipped "
                                  "hypnogram; it differs on "
                                  f"{n_changed} of {n} epochs in this recording.",
        },

        "per_stage": per_stage,
        "derived_metrics": {k: round(float(v), 4) for k, v in metrics.items()},

        # Which derived values may be stated as facts, and which may not.
        # extract_features is exact arithmetic - but on MODEL OUTPUT, and the
        # error does not distribute evenly. Measured on validation.
        "derived_metric_reliability": {
            "measured_on": MR.get("measured_on", "validation split"),
            "safe_to_assert": sorted(k for k, v in MR["metrics"].items()
                                     if v["tier"] == "robust"),
            "assert_with_error_bound": sorted(k for k, v in MR["metrics"].items()
                                              if v["tier"] == "fragile"),
            "do_not_assert_without_caveat": sorted(k for k, v in MR["metrics"].items()
                                                   if v["tier"] == "unreliable"),
            "per_metric": MR["metrics"],
            "note": "Only aggregate metrics survive per-epoch prediction error. "
                    "Single-event metrics (REM latency) and run-structure metrics "
                    "(REM periods, transitions) inherit it directly.",
        },

        "night_confidence": {
            "tier": night_tier,
            "mean_entropy_nats": round(night_H, 4),
            "tier_boundaries_nats": {"high_max": round(lo, 4), "medium_max": round(hi, 4)},
            "boundaries_fitted_on": "validation split entropy tertiles",
            "requires_ground_truth": False,
        },

        "evidence_items": ev,

        # ---- reserved, present-and-null so adding them is not breaking ------
        "attribution": None,
        "attribution_quality": {"status": "not_run", "gate": "3a",
                                "preregistration": "distillation/PREREGISTRATION_gate3a.md"},
        "risk": None,
        "risk_available": False,

        "limitations": [
            "Staging is from a single EEG channel (Fpz-Cz) via 34 derived spectral features. "
            "The raw-waveform pathway contributes nothing to predictions in this model, verified by ablation.",
            "N1 is representation-bound: pairwise N1-vs-N2 AUC ~0.81, unchanged across every model "
            "variant, and no decision threshold improves N1 F1 by more than +0.0086.",
            "Night-level confidence predicts agreement with the expert scorer, which is not the same "
            "as clinical correctness - both can be wrong on the same night.",
            "The hypnogram is decoded with a REM-targeted minimum-run rule, so REM runs shorter "
            "than 3 epochs cannot appear in it by construction. A genuine brief REM intrusion "
            "would therefore be absorbed rather than reported. On validation the rule moved REM "
            "period counts from 2.01x the expert value to 0.83x - closer, but now slightly "
            "UNDER-counting, so REM fragmentation should be read as a lower bound.",
            "No EOG or EMG is used. AASM scoring defines REM partly by eye movements and N1 partly "
            "by slow rolling eye movements, both of which are unavailable to this model.",
        ],

        # held back deliberately: the packet describes what the model SAID, so a
        # consumer cannot accidentally score itself against labels it should not see
        "_ground_truth_withheld": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="student_baseline_E0")
    ap.add_argument("--splits", default=str(Path(__file__).parent / "splits.json"))
    ap.add_argument("--out", default=str(RES / "packets"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    global MR, DEC
    DEC = load_decoder()
    rel = json.loads((Path(__file__).parent / "reliability_table.json").read_text(encoding="utf-8"))
    nc = json.loads((RES / "night_confidence.json").read_text(encoding="utf-8"))
    mrp = RES / "derived_metric_reliability.json"
    if not mrp.exists():
        raise SystemExit(f"Missing {mrp}. Run metric_reliability.py first - without it "
                         f"the packet cannot mark which derived values are safe to state.")
    MR = json.loads(mrp.read_text(encoding="utf-8"))
    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    recs = sorted(r for s in sp["splits"]["test"] for r in sp["recordings_by_subject"][s])
    if args.limit:
        recs = recs[:args.limit]

    ckpt = RES / "students" / args.model / "student_best.pt"
    prov = {
        "model": args.model,
        "model_checkpoint": str(ckpt.relative_to(REPO_ROOT)).replace("\\", "/"),
        "model_sha256": sha256_of(ckpt),
        "model_parameters": 121099,
        "git_commit": git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "calibration_temperature": rel["calibration_temperature"],
        "hypnogram_decoder": DEC.artefact["selected_label"],
        "hypnogram_decoder_spec": DEC.spec,
        "generator": "distillation/build_packet.py",
        "schema_version": SCHEMA_VERSION,
    }

    # The decoder was fitted against a specific temperature. If calibrate.py has
    # since refitted T, the decoder's validation-selected parameters no longer
    # describe these probabilities and every reliability tier downstream is stale.
    T_dec = DEC.artefact["calibration_temperature"]
    if abs(T_dec - rel["calibration_temperature"]) > 1e-3:
        raise SystemExit(
            f"Temperature mismatch: decoder fitted against T={T_dec}, "
            f"reliability_table has T={rel['calibration_temperature']}. "
            f"Re-run sequence_decode.py, then metric_reliability.py.")

    print(f"model      : {args.model}")
    print(f"checkpoint : {prov['model_checkpoint']}")
    print(f"sha256     : {prov['model_sha256'][:16]}...")
    print(f"commit     : {prov['git_commit'][:12]}")
    print(f"temperature: {prov['calibration_temperature']}")
    print(f"decoder    : {DEC.artefact['selected_label']}  {DEC.spec}")
    print(f"schema     : {SCHEMA_VERSION}")

    cache = RES / f"probs_{args.model}"
    if not cache.exists():
        raise SystemExit(f"Missing {cache}. Run evaluate_student.py first.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nbuilding {len(recs)} packets -> {out.relative_to(REPO_ROOT)}/")
    ok = skipped = 0
    for r in recs:
        p = build(r, rel, nc, prov, cache)
        if p is None:
            print(f"  {r}: SKIPPED (no sleep epochs)")
            skipped += 1
            continue
        (out / f"{r}.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
        ok += 1
        print(f"  {r}: {p['n_epochs']:>5} epochs | {len(p['evidence_items'])} evidence items"
              f" | night confidence {p['night_confidence']['tier']}")

    print(f"\n{ok} packets written, {skipped} skipped")
    print(f"schema {SCHEMA_VERSION} | attribution: null (gate 3a not run) | risk: null (not integrated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
