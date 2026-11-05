"""Plain-language values for one evidence packet: graded against sourced, age-specific norms.

Deterministic, no model calls. Every value comes from the packet (or the two lookup tables
build_tables.py makes); every grade comes from a cited rule in reference_ranges.json. Stage
percentages are percent of TOTAL SLEEP TIME (the clinical convention), not of the recording.

    python "PLAIN LANGUAGE REPORT/lay_metrics.py" SC4081E0-PSG       # assess one dev/train night
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RANGES = json.loads((HERE / "reference_ranges.json").read_text(encoding="utf-8"))


def _table(name: str) -> dict:
    p = HERE / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# Release v2 packets (CURRENT_RELEASE.md). Same epoch windows as the v1 packets for all 197 nights,
# so clock_offsets.json and the expert labels apply unchanged.
V2_PACKETS = ROOT / "ADDING EOG VERSION - PIPELINE INTEGRATION" / "work_n6adv_packets"
PACKET_DIRS = {"dev": V2_PACKETS / "packets_dev", "train": V2_PACKETS / "packets_train",
               "test": V2_PACKETS / "packets_test"}


def find_packet(rec: str, splits=("dev", "train")) -> Path | None:
    rec = rec if rec.endswith("-PSG") else rec + "-PSG"
    return next((PACKET_DIRS[s] / f"{rec}.json" for s in splits if (PACKET_DIRS[s] / f"{rec}.json").exists()), None)


SUBJECTS = _table("subjects.json")          # keyed by the first 6 characters: SC4081, ST7081
OFFSETS = _table("clock_offsets.json")      # keyed by recording stem: SC4081E0
# validate_awakenings.py: tier measured on dev with the project's own rule (unreliable, 47% rel. error)
AWAKENINGS_TIER = _table("awakenings_validation.json").get("tier", "not_yet_validated")
# Research packets start 30 min before the FIRST sleep anywhere in a ~20 h recording, which on some
# nights is an evening nap. Allowed gap between the model's first sleep and recorded lights-off:
WINDOW_TOLERANCE_MIN = 30   # = the preprocessing pad, and the NSF 'good' sleep-latency bound

# Our metric -> the packet key whose validation reliability tier applies to it.
PACKET_KEY = {"total_sleep_time_h": "Total_Sleep_Time", "sleep_efficiency_pct": "Sleep_Efficiency",
              "sleep_latency_min": "Sleep_Onset_Latency", "waso_min": "Wake_After_Sleep_Onset",
              "rem_pct_tst": "R_Proportion", "n1_pct_tst": "N1_Proportion", "n2_pct_tst": "N2_Proportion",
              "n3_pct_tst": "N3_Proportion"}


# ── values from the packet ──────────────────────────────────────────────────

def sleep_period(packet: dict) -> tuple[int, int] | None:
    """(first, last) sleep epoch of the decoded hypnogram; None if the night has no sleep."""
    runs = packet["hypnogram"]["stage_runs"]
    sleep = [r for r in runs if r["stage"] != "W"]
    if not sleep:
        return None
    return sleep[0]["start_epoch"], sleep[-1]["start_epoch"] + sleep[-1]["n_epochs"] - 1


def _wake_runs_in_period(packet: dict) -> list[dict]:
    period = sleep_period(packet)
    if period is None:
        return []
    first, last = period
    return [r for r in packet["hypnogram"]["stage_runs"]
            if r["stage"] == "W" and r["start_epoch"] > first and r["start_epoch"] + r["n_epochs"] - 1 < last]


def awakenings_over_5min(packet: dict) -> int:
    """NSF 2017 Table 2: episodes awake for MORE than 5 minutes, between falling asleep and final awakening."""
    return sum(1 for r in _wake_runs_in_period(packet) if r["duration_sec"] > 300)


def values(packet: dict) -> dict:
    dm = packet["derived_metrics"]
    return {
        "total_sleep_time_h": dm["Total_Sleep_Time"] / 3600,
        "sleep_efficiency_pct": dm["Sleep_Efficiency"] * 100,
        "sleep_latency_min": dm["Sleep_Onset_Latency"] / 60,
        "waso_min": dm["Wake_After_Sleep_Onset"] / 60,
        "awakenings_over_5min": awakenings_over_5min(packet),
        "rem_pct_tst": dm["R_Proportion"] * 100,       # derived_metrics proportions are of total sleep time
        "n1_pct_tst": dm["N1_Proportion"] * 100,
        "n2_pct_tst": dm["N2_Proportion"] * 100,
        "n3_pct_tst": dm["N3_Proportion"] * 100,
    }


# ── grading against the sourced rules ───────────────────────────────────────

def age_group(age: int | None) -> str | None:
    if age is None:
        return None
    return next((g for g, (lo, hi) in RANGES["age_groups"].items() if lo <= age <= hi), None)


def _holds(op: str, v: float, ref) -> bool:
    if op == "between":
        return ref[0] <= v <= ref[1]
    return {"<=": v <= ref, ">=": v >= ref, "<": v < ref, ">": v > ref}[op]


def grade(metric: str, value: float, age: int | None) -> str:
    """Only what the source actually says for this age group:
      good / poor          - the value meets that rule
      in_between           - the source sets both a good and a poor level, and the value meets neither
      outside_good_range   - not good, and the source sets no poor level for this group
      not_flagged          - not poor, and the source sets no good level for this group
      no_rule              - the source sets no level at all for this group"""
    group = age_group(age)
    rules = [r for r in RANGES["metrics"][metric]["rules"]
             if group in r["groups"] and age is not None and age <= r.get("max_age", 200)]
    kinds = {r["verdict"] for r in rules}
    for verdict in ("poor", "good"):
        if any(_holds(r["op"], value, r["value"]) for r in rules if r["verdict"] == verdict):
            return verdict
    return {frozenset({"good", "poor"}): "in_between", frozenset({"good"}): "outside_good_range",
            frozenset({"poor"}): "not_flagged"}.get(frozenset(kinds), "no_rule")


# ── clock times ─────────────────────────────────────────────────────────────

def clock_times(packet: dict, edf_start: str, offset_s: float) -> dict:
    """Wall-clock HH:MM of the packet's start, sleep onset, final awakening and end."""
    t0 = datetime.strptime(edf_start, "%H:%M:%S") + timedelta(seconds=offset_s)
    at = lambda epoch: (t0 + timedelta(seconds=30 * epoch)).strftime("%H:%M")  # noqa: E731
    period = sleep_period(packet)
    return {"recording_start": at(0),
            "fell_asleep": at(period[0]) if period else None,
            "woke_up": at(period[1] + 1) if period else None,
            "recording_end": at(packet["n_epochs"])}


def _offset(packet: dict) -> dict | None:
    """The clock offset, only for the window it was measured on (matched by epoch count). A Path B packet
    of the same recording starts at lights-off + the estimated night instead, so it gets none."""
    off = OFFSETS.get(packet["recording_id"].removesuffix("-PSG"))
    return off if off and off.get("n_epochs") == packet["n_epochs"] else None


def window_check(packet: dict) -> dict:
    """Does the packet window describe the night? 'ok' when the model's first sleep is no more than
    WINDOW_TOLERANCE_MIN before recorded lights-off; 'pre_bedtime_sleep' when it is earlier (a nap
    or evening dozing is inside the window); 'unknown' without a clock offset (test split) or
    lights-off time. Uses the model's hypnogram only - no expert labels."""
    rec = packet["recording_id"].removesuffix("-PSG")
    off, lights = _offset(packet), SUBJECTS.get(rec[:6], {}).get("lights_off")
    period = sleep_period(packet)
    if not off or not lights or period is None:
        return {"status": "unknown", "minutes_before_lights_off": None}
    first = datetime.strptime(off["edf_start"], "%H:%M:%S") + timedelta(seconds=off["offset_s"] + 30 * period[0])
    gap = ((datetime.strptime(lights, "%H:%M:%S") - first).total_seconds() / 60 + 720) % 1440 - 720
    return {"status": "pre_bedtime_sleep" if gap > WINDOW_TOLERANCE_MIN else "ok",
            "minutes_before_lights_off": round(gap)}


# ── one night ───────────────────────────────────────────────────────────────

def assess(packet: dict) -> dict:
    rec = packet["recording_id"].removesuffix("-PSG")
    who = SUBJECTS.get(rec[:6], {})
    age = who.get("age")
    tiers = packet["derived_metric_reliability"]["per_metric"]
    window = window_check(packet)
    rows = {}
    for metric, value in values(packet).items():
        key = PACKET_KEY.get(metric)
        rows[metric] = {
            "label": RANGES["metrics"][metric]["label"],
            "value": round(value, 1),
            "unit": RANGES["metrics"][metric]["unit"],
            # norms assume a lights-off -> lights-on window; don't compare when that isn't established
            "grade": grade(metric, value, age) if window["status"] == "ok" else "not_graded",
            "source": RANGES["metrics"][metric]["source"],
            "model_reliability": tiers.get(key, {}).get("tier") if key else AWAKENINGS_TIER,
            "mean_abs_error": tiers.get(key, {}).get("mean_abs_error") if key else None,
        }
    off = _offset(packet)
    return {"recording_id": packet["recording_id"], "age": age, "sex": who.get("sex"),
            "age_group": age_group(age), "condition": who.get("condition"),
            "clock": clock_times(packet, off["edf_start"], off["offset_s"]) if off else None,
            "window": window, "metrics": rows}


def _selfcheck() -> None:
    """Our sleep period must reproduce the packet's own WASO and sleep-onset latency exactly."""
    n = 0
    for d in ("dev", "train"):
        for f in sorted(PACKET_DIRS[d].glob("*.json")):
            p = json.loads(f.read_text(encoding="utf-8"))
            if sleep_period(p) is None:
                continue
            waso = sum(r["duration_sec"] for r in _wake_runs_in_period(p))
            assert waso == p["derived_metrics"]["Wake_After_Sleep_Onset"], (f.name, waso)
            assert sleep_period(p)[0] * 30 == p["derived_metrics"]["Sleep_Onset_Latency"], f.name
            n += 1
    assert grade("sleep_efficiency_pct", 76.3, 25) == "in_between"
    assert grade("sleep_efficiency_pct", 70.0, 25) == "in_between"     # young adults: poor only below 65
    assert grade("sleep_efficiency_pct", 70.0, 40) == "poor"
    assert grade("waso_min", 98.5, 70) == "outside_good_range"         # NSF: no 'poor' WASO bar for 65+
    assert grade("n3_pct_tst", 49.6, 25) == "not_flagged"              # young adults: only 'poor <= 5%' exists
    assert grade("n3_pct_tst", 18.0, 40) == "good"
    assert grade("total_sleep_time_h", 6.5, 70) == "no_rule"           # Watson 2015 covers 18-60 only
    print(f"self-check OK: WASO and sleep-onset reproduced on {n} packets; grading rules behave")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _selfcheck()
    else:
        path = find_packet(sys.argv[1].removesuffix(".json"))
        if path is None:
            raise SystemExit(f"{sys.argv[1]}: not a v2 dev/train packet (the test split is not read here)")
        print(json.dumps(assess(json.loads(path.read_text(encoding="utf-8"))), indent=2))
