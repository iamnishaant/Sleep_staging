"""Command line.

    python -m agentic manifest [--release v1]
    python -m agentic run-packet <packet.json> [--release v1] [--backend llama|replay] [--runtime DIR]
    python -m agentic run-packets <dir> [...same options]
    python -m agentic run-edf <recording.edf> [--crop-start-s S --crop-hours H] [...same options]

`--backend replay` serves the frozen evaluation's cached outputs (fast; for tests and plumbing checks).
`--backend llama` runs llama-completion exactly as the frozen evaluation did.
`--backend oracle` writes the project's deterministic oracle claims (tests / plumbing only, never results).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import DEFAULT_RUNTIME_DIR, REPO_ROOT
from .core.hashing import sha_bytes
from .core.policy import Policy
from .core.release import Release
from .core.runner import Runner
from .core.store import RunStore
from .report.backends import LlamaCompletionBackend, OracleBackend, ReplayBackend
from .workflows import edf_report, packet_report

REPLAY_CACHE = REPO_ROOT / "candidates" / "cache" / "Llama-Student" / "P1"


def make_runner(release_name: str, policy_path: str | None, backend: str, runtime: str | Path,
                workflow=packet_report.WORKFLOW) -> Runner:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))          # the frozen pipeline is imported, never edited
    release, policy = Release.load(release_name), Policy.load(policy_path)
    if backend == "llama":
        report_backend = LlamaCompletionBackend(release, timeout_s=float(policy.get("report.timeout_s")))
    elif backend == "replay":
        report_backend = ReplayBackend(REPLAY_CACHE)
    elif backend == "oracle":
        report_backend = OracleBackend()
    else:
        raise SystemExit(f"unknown backend {backend!r}")
    return Runner(workflow, release, policy, RunStore(runtime), {"report_backend": report_backend})


def run_packet(runner: Runner, path: Path) -> dict:
    data = path.read_bytes()
    return runner.run(sha_bytes(data), {"packet.json": data})


def run_edf(runner: Runner, path: Path, crop: dict | None = None) -> dict:
    data = path.read_bytes()
    seed = {"recording.edf": data, "meta.json": json.dumps({"recording_id": path.stem}).encode()}
    if crop:
        seed["crop.json"] = json.dumps(crop, sort_keys=True).encode()
    return runner.run(sha_bytes(data + seed.get("crop.json", b"")), seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agentic", description="Deterministic agent around the frozen sleep-report pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="print the execution manifest of a release")
    m.add_argument("--release", default="v1")
    for name in ("run-packet", "run-packets", "run-edf"):
        s = sub.add_parser(name)
        s.add_argument("path")
        s.add_argument("--release", default="v1")
        s.add_argument("--policy", default=None)
        s.add_argument("--backend", choices=["llama", "replay", "oracle"], default="llama")
        s.add_argument("--runtime", default=str(DEFAULT_RUNTIME_DIR))
        if name == "run-edf":
            s.add_argument("--crop-start-s", type=float, default=None)
            s.add_argument("--crop-hours", type=float, default=None)
    a = ap.parse_args(argv)

    if a.cmd == "manifest":
        r = Release.load(a.release)
        r.check_pins()
        print(json.dumps({"sha256": r.manifest_sha256, **r.manifest()["core"]}, indent=1))
        return 0
    if a.cmd == "run-edf":
        runner = make_runner(a.release, a.policy, a.backend, a.runtime, edf_report.WORKFLOW)
        crop = {"start_s": a.crop_start_s, "hours": a.crop_hours} if a.crop_start_s is not None else None
        rec = run_edf(runner, Path(a.path), crop)
        print(f"{rec['recording_id']}: {rec['outcome']} {rec['reasons'] or ''} -> outbox/{rec['run_id']}.html")
        return 0
    runner = make_runner(a.release, a.policy, a.backend, a.runtime)
    paths = [Path(a.path)] if a.cmd == "run-packet" else sorted(Path(a.path).glob("*.json"))
    for p in paths:
        rec = run_packet(runner, p)
        print(f"{rec['recording_id']}: {rec['outcome']} {rec['reasons'] or ''} -> outbox/{rec['run_id']}.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
