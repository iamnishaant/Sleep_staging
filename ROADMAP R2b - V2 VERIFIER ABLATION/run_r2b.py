"""R2b: the verifier ablation on release v2 (PREREGISTRATION.md).

    python "ROADMAP R2b - V2 VERIFIER ABLATION/run_r2b.py"          plan only; runs nothing
    python "ROADMAP R2b - V2 VERIFIER ABLATION/run_r2b.py" --go     generate arm C (resumable), then score
    python "ROADMAP R2b - V2 VERIFIER ABLATION/run_r2b.py" --score  score only

Arm C = release v2's own llama-completion invocation (agentic/report/backends.py) with exactly the
--grammar-file pair removed. Arms A/B reuse R3's grammar-on dev generations. Scoring is R2's code,
imported unchanged. Writes only inside this folder. Dev packets only.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from agentic.core.release import Release  # noqa: E402
from agentic.report.backends import LlamaCompletionBackend  # noqa: E402
from reference.prompts import build  # noqa: E402

PACKETS = ROOT / "ADDING EOG VERSION - PIPELINE INTEGRATION" / "work_n6adv_packets" / "packets_dev"
ARM_AB = ROOT / "ROADMAP R3 - V2 REPORT MODEL" / "results" / "dev"
CACHE = HERE / "cache_arm_c"
WORK = HERE / "work"

spec = importlib.util.spec_from_file_location("r2_score", ROOT / "ROADMAP R2 - VERIFIER ABLATION" / "score_ablation.py")
R2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R2)


class NoGrammar(LlamaCompletionBackend):
    """Release v2's invocation with exactly '--grammar-file <path>' removed (R2's arm C rule)."""

    def command(self, prompt_file):
        cmd = super().command(prompt_file)
        i = cmd.index("--grammar-file")
        return cmd[:i] + cmd[i + 2:]


def packets() -> list[Path]:
    assert "packets_dev" in str(PACKETS)                                  # dev only
    return sorted(PACKETS.glob("*.json"))


def generate() -> None:
    import os
    CACHE.mkdir(exist_ok=True)
    # 2 Nov: two runners once ran at the same time and shared one prompt file, so a generation could have
    # used another night's prompt (see RUN_NOTES.md). A lock refuses a second runner, and each run gets its
    # own work folder.
    lock = HERE / "run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"another runner holds {lock.name} (pid {lock.read_text().strip()}); "
                         "if none is running, delete the file and start again")
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        _generate(WORK / str(os.getpid()))
    finally:
        lock.unlink(missing_ok=True)


def _generate(work: Path) -> None:
    release = Release.load("v2")
    release.check_pins()
    backend = NoGrammar(release, timeout_s=1200)
    work.mkdir(parents=True, exist_ok=True)
    todo = [p for p in packets() if not (CACHE / p.name).exists()]
    print(f"arm C: {len(packets()) - len(todo)} cached, {len(todo)} to generate", flush=True)
    from agentic.core.device import device_health
    for i, p in enumerate(todo, 1):
        # A memory-starved generation would be an infrastructure failure, and the pre-registration forbids
        # re-runs; so wait for RAM instead of producing one (same 5.0 GB floor as P8 and the console).
        while (free := device_health()["free_ram_gb"]) < 5.0:
            print(f"    waiting: {free:.1f} GB free, need 5.0 GB (close other apps)", flush=True)
            time.sleep(60)
        packet = json.loads(p.read_text(encoding="utf-8"))
        t0 = time.monotonic()
        g = backend.generate(build("P1", packet), work)
        (CACHE / p.name).write_text(json.dumps({"text": g.text, "ok": g.ok, "failure": g.failure,
                                                "wall_s": g.wall_s, "hit_token_limit": g.failure == "no_end_marker"}),
                                    encoding="utf-8")
        print(f"[{i:2d}/{len(todo)}] {p.stem}: {g.failure or 'ok'} ({time.monotonic() - t0:.0f} s)", flush=True)


def rows(arm: str) -> list[dict]:
    out = []
    for p in packets():
        packet = json.loads(p.read_text(encoding="utf-8"))
        src = (ARM_AB if arm in "AB" else CACHE) / p.name
        e = json.loads(src.read_text(encoding="utf-8")) if src.exists() else None
        # a generation that ended without its end marker still put its text in front of the reader (ungated arms)
        text = e["text"] if e is not None and (e["ok"] or e.get("failure") == "no_end_marker") else None
        r = R2.score_one(text, packet, gated=(arm == "A"))
        r["recording_id"] = packet["recording_id"]
        r["hit_token_limit"] = bool(e and e.get("hit_token_limit"))
        out.append(r)
    return out


def score() -> None:
    res, lines = {}, ["| Arm | Released | Flawed released | Malformed | Claims | Wrong number | Wrong/missing evidence | "
                      "Unreliable as fact | Reliable hedged | Other | Missing mandatory | Fidelity | Token cap |",
                      "|" + "---|" * 13]
    for arm in ("A", "B", "C"):
        rs = rows(arm)
        s = R2.summarise(rs)
        res[f"Llama-Student v2 / {arm}"] = {"summary": s, "per_night": rs}
        lines.append(f"| {arm} | {s['reports_released']}/{s['nights']} | {s['released_reports_flawed']} | {s['malformed_releases']} | "
                     f"{s['claims_released']} | {s['wrong_number']} | {s['wrong_or_missing_evidence']} | "
                     f"{s['unreliable_stated_as_fact']} | {s['reliable_hedged']} | {s['other_rule_breaks']} | "
                     f"{s['released_missing_mandatory']} | {s['numeric_fidelity']} | {s['token_limit_hits']} |")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "ablation_v2.json").write_text(json.dumps(res, indent=1, default=list), encoding="utf-8")
    (HERE / "results" / "ABLATION_TABLE_V2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true"); ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.go:
        generate()
    if a.go or a.score:
        score()
    else:
        print(f"{len(packets())} dev packets; arm C cached: {len(list(CACHE.glob('*.json'))) if CACHE.exists() else 0}")
