"""
Evidence packet v0 - one JSON per night, from artefacts that already exist.

This is M1 minus attributions. It is the INTERFACE CONTRACT between the staging
model and everything downstream (report generation, verification, the UI), so
the schema matters more than the code: once the vertical slice is written
against it, changing a field means changing every consumer.

Fields that are not yet populated are present and null rather than absent, so
adding them later is not a breaking change. `attribution` was reserved this way
and is now filled from Gate 3a (schema 1.3); `risk` remains reserved, pending
disorder-detection integration.

WHAT GOES IN, AND WHERE IT COMES FROM
-------------------------------------
  stage probabilities, hypnogram   cached predictions from evaluate_student.py
  calibrated probabilities         temperature from reliability_table.json, fitted on validation
  derived sleep metrics            code/Phase2_47/physiological_features.py
  per-stage confidence + tier      distillation/reliability_table.json
  night-level confidence + tier    distillation/results/night_confidence.json
  evidence_items                   assembled here, with stable ids
  attribution                      results/_g3a_n4kd.json, per_recording block
  attribution_quality              same file - the pooled Gate 3a verdict
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

SCHEMA_VERSION = "1.3"

# The N1 flag, fitted on VALIDATION by fit_n1_flag.py. Loaded rather than
# hardcoded so the packet cannot drift from the rule that was actually fitted.
def _load_n1_rule(res_dir):
    p = Path(res_dir) / "n1_flag.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

# 1.0 -> 1.1
#   ADDED    `decoding` (object) - how the hypnogram was produced from the
#            probabilities, and the spec needed to reproduce it.
#   CHANGED  `hypnogram.predicted_stages` is now the decoder's output, not
#            argmax of `probabilities.calibrated`. No field was removed or
#            renamed, so a 1.0 consumer still parses a 1.1 packet - but one that
#            recomputes the hypnogram by argmax will now disagree with it, which
#            is exactly why the version moved.
# 1.2 -> 1.3
#   POPULATED `attribution` (null since 1.0) - this night's per-stage
#            Integrated-Gradients profile over the 34 spectral features.
#   POPULATED `attribution_quality` - the Gate 3a verdict, with status
#            moving from "not_run" to "run", plus the two caveats that
#            bound it.
#   Both fields were reserved present-and-null precisely so this would be
#   additive. A 1.2 consumer that checked `attribution is None` now takes
#   the other branch; one that rendered the field blindly is unaffected.
# 1.1 -> 1.2
#   ADDED    `n1_confidence_flag` (object) - per-epoch flags marking the N1
#            predictions the model is least sure about, with the threshold and
#            the validation numbers behind it.
#   Additive only. A 1.1 consumer parses a 1.2 packet unchanged; it simply does
#   not distinguish flagged N1 epochs from unflagged ones, which is the state
#   every consumer was in before.
SCHEMA_CHANGELOG = {
    "1.3": {
        "added": [],
        "changed": ["attribution (null -> object)",
                    "attribution_quality (status not_run -> run)"],
        "breaking_for": "a consumer that treated `attribution is None` as "
                        "permanent. The field was documented as reserved "
                        "and pending Gate 3a from schema 1.0, so this is "
                        "the change it was reserved for.",
    },
    "1.2": {
        "added": ["n1_confidence_flag"],
        "changed": [],
        "breaking_for": "nothing - additive. Consumers that render all N1 epochs "
                        "identically now have information they can use, and are "
                        "not required to.",
    },
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


# Gate 3a, from the run that produced the committed verdict. Loaded rather
# than recomputed: recomputing IG per packet would take ~6 min per night and
# would let the packet drift from the verdict the report cites.
# The locked test packet directory. Named once, so the guard and the tests
# agree on what "the test set" means rather than each spelling a path.
TEST_PACKET_DIR = (RES / "packets").resolve()


def _gate_artefact_name(split: str) -> str:
    """One artefact per split. A val run yields a VAL cohort verdict, and
    putting the test verdict into dev packets would drop a test-derived
    aggregate into the artefacts being tuned on."""
    return "_g3a_n4kd.json" if split == "test" else f"_g3a_n4kd_{split}.json"


def _load_gate3a(res_dir, model="student_N4kd", split="test"):
    p = Path(res_dir) / _gate_artefact_name(split)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    m = d.get("models", {}).get(model)
    if not m or "per_recording" not in m:
        return None

    # The packet promises `_ground_truth_withheld: true`. A profile grouped by
    # the annotated stage breaks that promise: per-true-stage epoch counts,
    # differenced against this packet's own predicted per-stage counts, give a
    # consumer the night's confusion structure. Hard failure, not a quiet null -
    # a null would leave the leaky file on disk for the next rebuild to ship.
    grouped = m.get("grouped_by")
    if grouped != "predicted":
        raise SystemExit(
            f"{p} groups per-recording attribution by {grouped!r}, not 'predicted'.\n"
            "  A packet that withholds ground truth cannot carry per-true-stage\n"
            "  counts. Re-run gate3a_attribution.py after the grouping fix, or\n"
            "  delete the file to build packets without attribution.")
    return {"gate": d, "model": m}


N1_RULE = _load_n1_rule(RES)

# Populated in main() once --split is known. Loading it at import would bind
# the test artefact before the flag is read, which is the same silent-default
# failure --split exists to remove.
GATE3A = None


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

    # ---- Gate 3a attribution (schema 1.3) ----------------------------------
    # This night's profile, and the cohort-level verdict, kept apart on purpose.
    attribution_block, attribution_quality_block = None, {
        "status": "not_run", "gate": "3a",
        "preregistration": "distillation/PREREGISTRATION_gate3a.md"}
    if GATE3A is not None and rec in GATE3A["model"]["per_recording"]:
        gm, gd = GATE3A["model"], GATE3A["gate"]
        gv = gm["gate_verdict"]
        # Every epoch of this night must appear in exactly one stage bucket.
        # The gate skips trailing chunks shorter than 8 epochs, which silently
        # drops the tail of any recording whose length mod 256 is small - one
        # epoch of SC4021E0-PSG, found by this check rather than by reading it.
        _prof = gm["per_recording"][rec]
        _acc = sum(v["n_epochs"] for v in _prof.values())
        if _acc != len(pred_stages):
            raise SystemExit(
                f"{rec}: attribution profile covers {_acc} epochs but the packet "
                f"ships {len(pred_stages)}. A partial profile presented as the "
                f"night's would misstate which epochs were explained.")
        attribution_block = {
            "method": ("Integrated Gradients, 64 steps, midpoint rule, "
                       "straight-line path"),
            "baseline": ("as pre-registered, NOT all-zeros: the waveform "
                         "baseline is this recording's own mean amplitude and "
                         "the spectral baseline is the per-feature mean over "
                         "the TRAIN split. Attribution is therefore 'relative "
                         "to a featureless night', not 'relative to silence', "
                         "and the numbers are not comparable to a zero-baseline "
                         "IG run."),
            "over": ("the 34 derived spectral features. The raw-waveform branch "
                     "is attributed too, but as 3000 samples it is not "
                     "interpretable feature-by-feature and is summarised as a "
                     "branch share, not listed here."),
            "scope": ("THIS RECORDING. Computed on this night's own epochs, "
                      "grouped by the stage the model predicted."),
            "normalisation": ("shares of this stage's total absolute "
                              "attribution across the 34 spectral features; "
                              "they sum to the spectral branch share, not to 1"),
            "per_stage": gm["per_recording"][rec],
            "cohort_profile": ("distillation/results/_g3a_n4kd.json - the "
                               "pooled 29-recording profile this night should "
                               "be read against"),
            "how_to_read": (
                "These are the features the model leaned on for this night, not "
                "the features a clinician would cite. Attribution explains the "
                "model; it does not validate it. An epoch staged wrongly still "
                "produces a confident-looking attribution profile."),
        }
        attribution_quality_block = {
            "status": "run",
            "gate": "3a",
            "preregistration": gd["preregistration"],
            "model": gm["model"],
            "verdict": "PASS" if gv["PASS"] else "FAIL",
            "criterion": gv["criterion"],
            "predictions_met": gv["predictions_met"],
            "n_met": gv["n_met"],
            "void": gv["void"],
            "completeness_relative_error": gm["completeness"]["relative"],
            # Which split this verdict describes, and over how many recordings.
            # Carried explicitly because the renderer used to assert "test
            # split of 29 recordings" as a literal, which is false for any
            # packet built from another split.
            "evaluated_on_split": gm.get("split", "test"),
            "evaluated_on_n_recordings": gm.get("n_recordings",
                                                gm.get("n_test_recordings")),
            "scope": (
                "COHORT, NOT THIS NIGHT. The verdict was evaluated once on the "
                f"pooled {gm.get('split', 'test')} split of "
                f"{gm.get('n_recordings', gm.get('n_test_recordings'))} "
                "recordings. It is not a quality score for this recording's "
                "attributions, and no per-night version of it was measured."),
            "caveats": [
                ("The top-3 feature set is IDENTICAL across all five stages - "
                 "ratio_delta_beta, ratio_dt_ab and cD1_log_energy, differing "
                 "only in order. The gate asked whether stage-appropriate "
                 "features appear in the top 3, and they do, but they appear "
                 "for every stage. The profile discriminates far less between "
                 "stages than the per-stage predictions passing suggests."),
                ("The spectral/waveform branch split is DIMENSION-BIASED. "
                 "Spectral features take 18.7% of attribution mass on average, "
                 "which reads as a minor branch; but that mass is spread over "
                 "34 dimensions against 3000 waveform samples, so per dimension "
                 "the spectral features carry roughly 20x the attribution. "
                 "Neither number alone describes the split honestly."),
            ],
            "deviation_from_registration": gd["deviation"],
        }

    # ---- N1 confidence flag (roadmap 4.1, schema 1.2) ----------------------
    # N1 is representation-bound: no decision rule improves its F1 by more than
    # +0.0086, and the human-scorer ceiling is itself low. What the model CAN do
    # is say which of its N1 calls are the doubtful ones. The threshold is
    # fitted on VALIDATION by fit_n1_flag.py and loaded, never hardcoded here.
    #
    # Applied to the SHIPPED hypnogram, not to argmax, because the flag has to
    # describe the stages a reader is actually looking at. The threshold was
    # fitted on argmax validation predictions; the decoder changes few epochs
    # and the provenance below says so rather than glossing it.
    n1_flag_block = None
    if N1_RULE is not None:
        thr = N1_RULE["threshold"]
        is_n1 = np.array([sname == "N1" for sname in pred_stages])
        flagged = is_n1 & (conf < thr)
        n1_flag_block = {
            "rule": N1_RULE["rule"],
            "threshold": thr,
            "applied_to": "hypnogram.predicted_stages (the decoded, shipped stages)",
            "flagged_epochs": [int(i) for i in np.flatnonzero(flagged)],
            "n_n1_epochs": int(is_n1.sum()),
            "n_flagged": int(flagged.sum()),
            "fraction_of_n1_flagged": (round(float(flagged.sum() / is_n1.sum()), 4)
                                       if is_n1.sum() else None),
            "fitted_on": N1_RULE["fitted_on"],
            "validation_evidence": {
                "accuracy_unflagged": N1_RULE["validation"]["accuracy_unflagged"],
                "accuracy_flagged": N1_RULE["validation"]["accuracy_flagged"],
                "accuracy_all_n1": N1_RULE["validation"]["accuracy_all_n1"],
            },
            "how_to_read": (
                "A flagged epoch is one the model called N1 with low confidence. On "
                "validation, unflagged N1 calls were right "
                f"{N1_RULE['validation']['accuracy_unflagged']:.0%} of the time and "
                f"flagged ones {N1_RULE['validation']['accuracy_flagged']:.0%}. This "
                "does not make N1 more accurate - it identifies which N1 calls are "
                "worth a human's time. N1 remains the model's weakest class."),
            "provenance_caveat": (
                "The threshold was fitted on argmax validation predictions; it is "
                "applied here to the decoded hypnogram, which differs on "
                f"{n_changed} of {len(pred_stages)} epochs in this recording."),
        }

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
         # Read from night_confidence.json, not hardcoded. This shipped "-0.75"
         # in every packet while the measured value for the current model is
         # -0.508 - a stale statistic asserted to a consumer as the reason to
         # trust the tier.
         basis=f"prediction entropy; Spearman rho "
               f"{nc['correlation']['TEST']['spearman_rho']:+.3f} against per-recording "
               f"kappa on held-out data (n={nc['correlation']['TEST']['n']})")
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

        "n1_confidence_flag": n1_flag_block,
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

        # ---- Gate 3a (schema 1.3); `risk` stays reserved --------------------
        "attribution": attribution_block,
        "attribution_quality": attribution_quality_block,
        "risk": None,
        "risk_available": False,

        "limitations": [
            # Derived from the model, not hardcoded. The claim "the raw-waveform
            # pathway contributes nothing" was true of student_baseline_E0 and is
            # FALSE of the multiscale model, where zeroing the EEG changes 71% of
            # predictions. A hardcoded limitation silently outlives the model it
            # describes, and a false caveat is worse than none.
            ("Staging is from a single EEG channel (Fpz-Cz): the raw waveform plus 34 "
             "spectral features derived from that same channel. No EOG or EMG is used."
             if prov.get("model_eeg_scale", 1.0) != 1.0 else
             "Staging is from a single EEG channel (Fpz-Cz) via 34 derived spectral "
             "features. The raw-waveform pathway contributes nothing to predictions in "
             "this model, verified by ablation."),
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
    ap.add_argument("--split", required=True, choices=["train", "val", "test"],
                    help="which split to build packets for. REQUIRED and with "
                         "no default: a forgotten flag must error rather than "
                         "silently select the locked test split.")
    ap.add_argument("--out", required=True,
                    help="output directory. Writing into results/packets/ "
                         "additionally needs --split test and "
                         "--allow-test-overwrite.")
    ap.add_argument("--allow-test-overwrite", action="store_true",
                    help="second key required to rebuild the 29 locked test "
                         "packets. Regenerating them should take two "
                         "deliberate flags, not one.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # ---- guard FIRST, before anything is loaded or written --------------
    out = Path(args.out).resolve()
    writing_into_test = out == TEST_PACKET_DIR or TEST_PACKET_DIR in out.parents
    if writing_into_test:
        if args.split != "test":
            raise SystemExit(
                f"--split {args.split} would write into the locked test packet "
                f"directory ({TEST_PACKET_DIR}).\n"
                f"  That is the contamination this guard exists to stop. Point "
                f"--out somewhere else.")
        if not args.allow_test_overwrite:
            raise SystemExit(
                f"Refusing to overwrite the 29 locked test packets in "
                f"{TEST_PACKET_DIR}.\n"
                f"  Rebuilding them requires --split test AND "
                f"--allow-test-overwrite, deliberately.")
    elif args.split == "test":
        print(f"NOTE: building TEST packets into {out}, which is not the test "
              f"packet directory. Deliberate? --split test was passed "
              f"explicitly, so proceeding.")

    global MR, DEC, GATE3A
    DEC = load_decoder()
    rel = json.loads((Path(__file__).parent / "reliability_table.json").read_text(encoding="utf-8"))
    nc = json.loads((RES / "night_confidence.json").read_text(encoding="utf-8"))
    mrp = RES / "derived_metric_reliability.json"
    if not mrp.exists():
        raise SystemExit(f"Missing {mrp}. Run metric_reliability.py first - without it "
                         f"the packet cannot mark which derived values are safe to state.")
    MR = json.loads(mrp.read_text(encoding="utf-8"))
    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    if args.split not in sp["splits"]:
        raise SystemExit(f"{args.splits} has no split named {args.split!r}; "
                         f"it has {sorted(sp['splits'])}.")
    recs = sorted(r for s in sp["splits"][args.split]
                  for r in sp["recordings_by_subject"][s])
    GATE3A = _load_gate3a(RES, args.model, args.split)
    if args.limit:
        recs = recs[:args.limit]

    ckpt = RES / "students" / args.model / "student_best.pt"
    prov = {
        "model": args.model,
        "model_checkpoint": str(ckpt.relative_to(REPO_ROOT)).replace("\\", "/"),
        "model_sha256": sha256_of(ckpt),
        # From reliability_table.json, which reads it from the checkpoint. A
        # hardcoded count here would put a wrong number in the one block whose
        # purpose is to identify what produced the packet.
        "model_parameters": rel["n_parameters"],
        "model_encoder": rel.get("model_encoder"),
        "model_eeg_scale": rel.get("model_eeg_scale"),
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
    print(f"split      : {args.split}  ({len(recs)} recordings)")
    print(f"gate 3a    : {_gate_artefact_name(args.split)} "
          f"{'loaded' if GATE3A is not None else 'MISSING - attribution will be null'}")

    # The convention every other tool already uses: probs_{model} is the test
    # split, probs_{model}_{split} is anything else.
    cache = RES / (f"probs_{args.model}" if args.split == "test"
                   else f"probs_{args.model}_{args.split}")
    if not cache.exists():
        raise SystemExit(f"Missing {cache}. Run evaluate_student.py first.")

    out.mkdir(parents=True, exist_ok=True)
    print(f"\nbuilding {len(recs)} {args.split.upper()} packets "
          f"-> {out.relative_to(REPO_ROOT)}/")
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
    _g3a = "gate 3a wired" if GATE3A is not None else "null (gate 3a output missing)"
    print(f"schema {SCHEMA_VERSION} | attribution: {_g3a} | risk: null (not integrated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
