"""Plain-language sentences for one night, from fixed templates. No language model.

Every number in the output passes through Text.num(): it comes from the packet, the lay_metrics
assessment or reference_ranges.json. _selfcheck() renders every dev/train packet and fails if any
digit in the text did not come through Text.num() - so the templates cannot introduce a number.

What a reader sees, by the model's own validation tier for each metric:
    robust     -> main section, with the norm comparison
    fragile    -> main section, with the norm comparison AND "could be off by about X"
    unreliable -> "Details for your doctor" only, never graded in the main text

    python "PLAIN LANGUAGE REPORT/plain_text.py" SC4081E0-PSG    # print one night
    python "PLAIN LANGUAGE REPORT/plain_text.py"                 # self-check on all dev/train
"""
from __future__ import annotations

import json
import re
import sys

import lay_metrics as L

R = L.RANGES


def _group_name(t: "Text", group: str | None) -> str:
    if group not in R["age_groups"]:
        return "your age group"
    lo, hi = R["age_groups"][group]
    return f"people aged {t.num(lo)} and over" if hi >= 120 else f"people aged {t.num(lo)}-{t.num(hi)}"

CANNOT_DETECT = ("This test only measured {signals}. It did not measure breathing, blood oxygen, heart rhythm "
                 "or leg movements, so it cannot tell you whether you have sleep apnea (pauses in breathing "
                 "during sleep), restless legs or other conditions that need those measurements.")
DISCLAIMER = ("This summary was written automatically, by fixed rules, from the same numbers as the "
              "clinical report. It describes one night and is not a diagnosis.")


class Text:
    """Collects every formatted number so the self-check can prove no other number was written."""

    def __init__(self):
        self.allowed: set[str] = set()

    def num(self, x, digits: int = 0) -> str:
        s = f"{x:.{digits}f}" if isinstance(x, float) else str(x)
        self.allowed.add(s)
        return s

    def dur(self, minutes: float) -> str:
        m = int(round(minutes))
        if m < 60:
            return f"{self.num(m)} min"
        return f"{self.num(m // 60)} h {self.num(m % 60)} min" if m % 60 else f"{self.num(m // 60)} h"

    def clock(self, hhmm: str) -> str:
        self.allowed.add(hhmm)
        return hhmm


def _rule_phrase(t: Text, rule: dict, unit: str) -> str:
    u = {"hours": " hours", "minutes": " minutes", "%": "%", "% of sleep": "%", "count": ""}[unit]
    v = rule["value"]
    if rule["op"] == "between":
        return f"{t.num(v[0])}-{t.num(v[1])}{u}"
    word = {">=": "{} or more", "<=": "{} or less", "<": "below {}", ">": "more than {}"}[rule["op"]]
    return word.format(f"{t.num(v)}{u}")


def _grade_sentence(t: Text, metric: str, grade: str, group: str | None) -> str:
    spec = R["metrics"][metric]
    rules = [r for r in spec["rules"] if group in r["groups"]]
    good = " or ".join(_rule_phrase(t, r, spec["unit"]) for r in rules if r["verdict"] == "good")
    poor = " or ".join(_rule_phrase(t, r, spec["unit"]) for r in rules if r["verdict"] == "poor")
    who = _group_name(t, group)
    return {
        "good": f"That is in the good range for {who} ({good}).",
        "poor": f"That is in the range experts consider poor for {who} ({poor}).",
        "in_between": f"That is between the good level ({good}) and the poor level ({poor}) for {who}.",
        "outside_good_range": f"That is outside the good range for {who} ({good}); the guideline sets no 'poor' level.",
        "not_flagged": f"The guideline only treats {poor} as a concern for {who}, and this is not.",
        "no_rule": f"The guideline sets no level for {who}, so this number is not compared.",
        "not_graded": "",
    }[grade]


def _error_note(t: Text, row: dict) -> str:
    mae = row["mean_abs_error"]
    if row["model_reliability"] != "fragile" or mae is None:
        return ""
    if row["unit"] == "% of sleep":
        return f"This estimate could be off by about {t.num(mae * 100)} percentage points."
    if row["unit"] in ("minutes", "hours"):
        return f"This estimate could be off by about {t.dur(mae)}."
    return ""


def _what(t: Text, metric: str, row: dict) -> str:
    v = row["value"]
    return {
        "total_sleep_time_h": f"You slept about {t.dur(v * 60)}.",
        "sleep_efficiency_pct": f"You were asleep for {t.num(round(v))}% of the time the recording covered.",
        "sleep_latency_min": f"It took about {t.dur(v)} to fall asleep.",
        "waso_min": f"After first falling asleep, you were awake for about {t.dur(v)} in total.",
        "awakenings_over_5min": (f"You were awake for longer than 5 minutes about {t.num(int(v))} "
                                 f"{'time' if int(v) == 1 else 'times'}."),
        "rem_pct_tst": f"Dreaming (REM) sleep made up about {t.num(round(v))}% of your sleep.",
        "n1_pct_tst": f"Very light (N1) sleep made up about {t.num(round(v))}% of your sleep.",
        "n2_pct_tst": f"Light (N2) sleep made up about {t.num(round(v))}% of your sleep.",
        "n3_pct_tst": f"Deep (N3) sleep made up about {t.num(round(v))}% of your sleep.",
    }[metric]


MAIN = ["total_sleep_time_h", "sleep_efficiency_pct", "waso_min", "sleep_latency_min", "awakenings_over_5min"]
STAGES = ["n3_pct_tst", "rem_pct_tst", "n2_pct_tst", "n1_pct_tst"]


def render(packet: dict) -> dict:
    t, a = Text(), L.assess(packet)
    group, rows, window = a["age_group"], a["metrics"], a["window"]["status"]

    def item(m):
        row = rows[m]
        return {"metric": m, "label": row["label"], "grade": row["grade"], "reliability": row["model_reliability"],
                "text": " ".join(s for s in (_what(t, m, row), _grade_sentence(t, m, row["grade"], group),
                                             _error_note(t, row)) if s)}

    main = [item(m) for m in MAIN if rows[m]["model_reliability"] in ("robust", "fragile")]
    stages = [item(m) for m in STAGES if rows[m]["model_reliability"] in ("robust", "fragile")]
    doctor = [item(m) for m in MAIN + STAGES if rows[m]["model_reliability"] not in ("robust", "fragile")]
    for d in doctor:   # unreliable: say so, never grade
        d["text"] = f"{_what(t, d['metric'], rows[d['metric']])} The system's estimate of this is not reliable enough to compare with the usual range."
        d["grade"] = "not_graded"

    clock = a["clock"]
    headline = _what(t, "total_sleep_time_h", rows["total_sleep_time_h"])
    if clock and window == "ok":
        headline += f" You first fell asleep at about {t.clock(clock['fell_asleep'])} and woke up for the day at about {t.clock(clock['woke_up'])}."

    window_note = {
        "ok": None,
        "pre_bedtime_sleep": "This recording includes sleep well before bedtime (for example an evening nap), "
                             "so the time-based numbers cover the whole recording, not just the night. "
                             "They are not compared with the usual ranges.",
        "unknown": "We could not check when this night's bedtime was, so the numbers are not compared "
                   "with the usual ranges.",
    }[window]

    tier = (packet.get("night_confidence") or {}).get("tier")
    confidence = {"high": "The system was confident about most of this night.",
                  "medium": "The system was fairly confident about this night.",
                  "low": "The system was unsure about much of this night, so treat these numbers with extra caution."
                  }.get(tier, "")

    poor = any(i["grade"] == "poor" for i in main + stages)
    doctor_advice = ("One or more numbers were in the range experts consider poor. One night can be unusual, so "
                     "this is not a diagnosis, but if you often sleep badly or feel very tired during the day, "
                     "talk to a doctor." if poor else
                     "If you often sleep badly or feel very tired during the day, talk to a doctor, even if "
                     "these numbers look fine.")

    model = (packet.get("provenance") or {}).get("model", "")
    # packets name the v2 model "student_N6adv_full_s1" (Path A) or "n6adv_full" (Path B adapter)
    signals = "brain activity and eye movements" if "n6adv" in model.lower() else "brain activity"
    context = ("This night was recorded after taking a sleeping medication (temazepam) as part of a study."
               if a["condition"] == "temazepam" else None)

    out = {
        "recording_id": a["recording_id"],
        "headline": headline,
        "context": context,
        "window_note": window_note,
        "main": main,
        "sleep_types_note": "Experts do not agree that the mix of sleep types measures sleep quality, and it is "
                            "the least certain part of this estimate.",
        "sleep_types": stages,
        "doctor_details": doctor,
        "confidence": confidence,
        "talk_to_doctor": doctor_advice,
        "cannot_detect": CANNOT_DETECT.format(signals=signals),
        "disclaimer": DISCLAIMER,
        "sources": sorted({R["sources"][a["metrics"][m]["source"]] for m in MAIN + STAGES}),
    }
    out["_allowed_numbers"] = sorted(t.allowed)
    return out


def all_text(out: dict) -> str:
    parts = [out["headline"], out["context"], out["window_note"], out["confidence"], out["talk_to_doctor"],
             out["cannot_detect"], out["disclaimer"], out["sleep_types_note"]]
    parts += [i["text"] for k in ("main", "sleep_types", "doctor_details") for i in out[k]]
    return " ".join(p for p in parts if p)


def _selfcheck() -> None:
    n = 0
    for d in ("dev", "train"):
        for f in sorted(L.PACKET_DIRS[d].glob("*.json")):
            out = render(json.loads(f.read_text(encoding="utf-8")))
            text = all_text(out)
            # numbers not glued to letters (so the stage names N1/N2/N3 don't count)
            stray = [x for x in re.findall(r"(?<![A-Za-z])\d+(?::\d+)?", text) if x not in out["_allowed_numbers"]
                     and x != "5"]    # "longer than 5 minutes" is the NSF definition of the measure itself
            assert not stray, (f.name, stray)
            assert "nan" not in text.lower() and "None" not in text, f.name
            n += 1
    print(f"self-check OK: {n} nights rendered; every number in the text came from the packet, "
          f"the assessment or reference_ranges.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _selfcheck()
    else:
        path = L.find_packet(sys.argv[1], ("dev", "train", "test"))
        if path is None:
            raise SystemExit(f"{sys.argv[1]}: no v2 packet")
        out = render(json.loads(path.read_text(encoding="utf-8")))
        out.pop("_allowed_numbers")
        print(json.dumps(out, indent=2))
