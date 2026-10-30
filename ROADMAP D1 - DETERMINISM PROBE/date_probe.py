"""Does the chat template's date explain the day-to-day flips?

    python "ROADMAP D1 - DETERMINISM PROBE/date_probe.py"

The Student GGUF's embedded Llama 3.2 template sets date_string = strftime_now() when none is given, so
the system header carries the generation day. Training pinned "26 Jul 2024". This replays each night with
a copy of student/templates/Llama-3.2-3B-Instruct.pinned.jinja whose date is set to each day the night
was generated (04, 17, 24 Oct 2026) and to the training date. If the date is the cause, each day's
historical output is reproduced byte for byte. Writes results/date_probe.json.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from agentic.core.release import Release  # noqa: E402
from probe import DEV, Backend  # noqa: E402
from reference.prompts import build  # noqa: E402
from report import verify_report  # noqa: E402

PINNED = ROOT / "student" / "templates" / "Llama-3.2-3B-Instruct.pinned.jinja"
DATES = ("04 Oct 2026", "17 Oct 2026", "24 Oct 2026", "26 Jul 2024")
KNOWN = {"SC4582G0-PSG": {"04 Oct 2026": "b687af7bc3bb", "17 Oct 2026": "09480fc1c479", "24 Oct 2026": "b687af7bc3bb"},
         "SC4111E0-PSG": {"04 Oct 2026": "63377b5f3db4", "17 Oct 2026": "63377b5f3db4", "24 Oct 2026": "882724f95b19"}}


def main() -> int:
    release = Release.load("v1")
    base = PINNED.read_text(encoding="utf-8")
    assert base.startswith('{%- set date_string = "26 Jul 2024" %}')
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for date in DATES:
            tpl = Path(tmp) / f"tpl_{date.replace(' ', '_')}.jinja"
            tpl.write_text(base.replace("26 Jul 2024", date, 1), encoding="utf-8", newline="\n")
            for rec, known in KNOWN.items():
                packet = json.loads((DEV / f"{rec}.json").read_text(encoding="utf-8"))
                g = Backend(release, ["--chat-template-file", str(tpl)]).generate(build("P1", packet), Path(tmp))
                sha = hashlib.sha256(g.text.encode("utf-8")).hexdigest()
                codes = [str(v.code) for v in verify_report(g.text, packet).violations]
                match = known.get(date)
                rows.append({"recording_id": rec, "date": date, "sha256": sha, "codes": codes,
                             "historical": match, "reproduces_historical": None if match is None else sha.startswith(match)})
                print(f"{rec} {date}: {sha[:12]} {codes or 'PASS'}"
                      + ("" if match is None else f"  historical {match} -> {'MATCH' if sha.startswith(match) else 'differs'}"),
                      flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "date_probe.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
