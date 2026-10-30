"""P4: fault-injection scenarios S1-S15 (03_EDGE_SIMULATION_PI5.md §5 L3) + the I1-I8 invariant checker.

    python -m agentic.tools.p4_faults

Signal faults are real EDF files written from one dev night (SC4081E0, cropped at lights-off to 12 h),
so they go through the real ingest path. Report-model faults are injected backends. The oracle backend
stands in for the model where the scenario is about the workflow, not about generation quality.
Writes agentic/reports/p4_faults.json.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .. import PACKAGE_DIR, REPO_ROOT
from ..cli import make_runner, run_edf, run_packet
from ..core import atomic
from ..core.crash import ENV
from ..core.release import Release
from ..report.backends import Generation, OracleBackend
from ..workflows import edf_report
from .p2_parity import RAW, find_pair, lights_off_epoch

REC = "SC4081E0-PSG"
ROOT = PACKAGE_DIR / "runtime" / "p4"
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]


# ---------------------------------------------------------------------------------- inputs
def _base_raw():
    import mne
    psg, _ = find_pair(RAW, REC)
    raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False).pick(CHANNELS)
    c0 = lights_off_epoch(REC, psg) * 30
    return raw.crop(tmin=c0, tmax=c0 + 12 * 3600 - 1 / raw.info["sfreq"])


def _export(raw, name: str) -> Path:
    import mne
    p = ROOT / "inputs" / f"{name}-PSG.edf"
    p.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(p), raw, fmt="edf", overwrite=True, verbose=False)
    return p


def make_inputs() -> dict[str, Path]:
    names = {"base": "S01_base", "flat2h": "S03_flat2h", "no_eog": "S04_no_eog", "no_fpz": "S05_no_fpz",
             "128hz": "S06_128hz", "2h": "S07_2h", "corrupt": "S08_corrupt", "linenoise": "S14_linenoise"}
    existing = {k: ROOT / "inputs" / f"{v}-PSG.edf" for k, v in names.items()}
    if all(p.exists() for p in existing.values()):
        return existing                      # same bytes -> same run IDs -> finished scenarios are not redone
    raw = _base_raw()
    sf = raw.info["sfreq"]
    out = {"base": _export(raw, "S01_base")}
    r = raw.copy()
    i0 = int(3 * 3600 * sf)
    r._data[r.ch_names.index("EEG Fpz-Cz"), i0:i0 + int(2 * 3600 * sf)] = 0.0          # electrode off for 2 h
    out["flat2h"] = _export(r, "S03_flat2h")
    out["no_eog"] = _export(raw.copy().drop_channels(["EOG horizontal"]), "S04_no_eog")
    out["no_fpz"] = _export(raw.copy().drop_channels(["EEG Fpz-Cz"]), "S05_no_fpz")
    out["128hz"] = _export(raw.copy().resample(128, verbose=False), "S06_128hz")
    out["2h"] = _export(raw.copy().crop(tmax=2 * 3600 - 1 / sf), "S07_2h")
    bad = ROOT / "inputs" / "S08_corrupt-PSG.edf"
    data = bytearray(out["base"].read_bytes())
    data[0:256] = b"\x00" * 256                                                       # destroy the header
    bad.write_bytes(bytes(data))
    out["corrupt"] = bad
    r = raw.copy()
    t = np.arange(r.n_times) / sf
    r._data[r.ch_names.index("EEG Fpz-Cz")] += 20e-6 * np.sin(2 * np.pi * 49.0 * t)    # strong line noise
    out["linenoise"] = _export(r, "S14_linenoise")
    return out


# ---------------------------------------------------------------------------------- backends
class CrashOnce(OracleBackend):
    name = "crash-once"

    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate(self, prompt, workdir):
        self.calls += 1
        return Generation("", False, "crash", 0.0, 1, "simulated crash") if self.calls == 1 else super().generate(prompt, workdir)


class Hang:
    name = "hang"

    def generate(self, prompt, workdir):
        return Generation("", False, "timeout", 1200.0, None, "simulated hang")


class Garbage(OracleBackend):
    name = "garbage"

    def generate(self, prompt, workdir):
        return Generation('[{"claim_id": "c1", "claim_type": "value"', True, None, 0.0, 0)


# ---------------------------------------------------------------------------------- invariants
NETWORK = re.compile(r"^\s*(import|from)\s+(urllib|requests|socket|http|httpx|aiohttp)\b", re.M)


def static_invariants() -> dict:
    agent_code = [p for p in PACKAGE_DIR.rglob("*.py") if not ({"tests", "tools"} & set(p.relative_to(PACKAGE_DIR).parts))]
    net = [str(p.relative_to(PACKAGE_DIR)) for p in agent_code if NETWORK.search(p.read_text(encoding="utf-8"))]
    # reading EXPERT hypnograms (files or annotations); the packet's own predicted "hypnogram" field is fine
    hyp_read = re.compile(r"Hypnogram\.edf|read_annotations|hypnogram_labels")
    hyp = [str(p.relative_to(PACKAGE_DIR)) for p in agent_code if hyp_read.search(p.read_text(encoding="utf-8"))]
    return {"I4_no_network_imports": net == [], "I4_offenders": net,
            "I5_workflow_never_reads_hypnograms": hyp == [], "I5_offenders": hyp}


def run_invariants(runtime: Path, run_id: str, manifest_sha: str) -> dict:
    run = runtime / "runs" / run_id
    rec = atomic.read_json(run / "DELIVERED") if (run / "DELIVERED").exists() else None
    if rec is None:
        return {"I6_terminated_and_delivered": False}
    steps = {d.name.split("_", 1)[1]: d for d in run.iterdir() if d.is_dir() and "_" in d.name}
    vh = atomic.read_json(steps["vhtml"] / "vhtml.json") if "vhtml" in steps else None
    ver = atomic.read_json(steps["verify"] / "verify.json") if "verify" in steps else None
    traced = {(json.loads(l)["step"], json.loads(l).get("event")) for l in (runtime / "trace.jsonl").read_text().splitlines()
              if json.loads(l)["run_id"] == run_id}
    out = rec["outcome"]
    return {
        "I1_no_unverified_text": out != "VERIFIED_REPORT" or bool(ver and ver["ok"] and vh and not vh["unexplained_numbers"]),
        "I2_numbers_have_provenance": out not in ("VERIFIED_REPORT", "FALLBACK_SUMMARY") or bool(vh and not vh["unexplained_numbers"]),
        "I3_frozen_components": atomic.read_json(run / "run.json")["execution_manifest_sha256"] == manifest_sha,
        "I6_terminated_and_delivered": True,
        "I7_every_step_traced": all((name, "done") in traced for name in steps),
        "I8_failure_visible": (out == "VERIFIED_REPORT" and not any(r[0] in "WQ" for r in rec["reasons"]))
                              or (runtime / "review" / f"{run_id}.json").exists(),
    }


# ---------------------------------------------------------------------------------- scenarios
def main(argv=None) -> int:
    inputs = make_inputs()
    manifest_sha = Release.load("v1").manifest_sha256
    dev_packet = REPO_ROOT / "distillation/results/phase2_dev_packets" / f"{REC}.json"
    hot = {"device_health": lambda: {"soc_temp_c": 82.0, "free_ram_gb": 10.0}, "sleep": lambda s: None}

    def edf(name, path, backend=None, services=None, expect=("VERIFIED_REPORT", [])):
        rt = ROOT / "runs" / name
        r = make_runner("v1", None, "oracle", rt, edf_report.WORKFLOW)
        if backend is not None:
            r.services["report_backend"] = backend
        r.services.update(services or {})
        rec = run_edf(r, path)
        return rt, rec, expect

    scenarios = {
        "S01 clean night, no hypnogram": lambda n: edf(n, inputs["base"]),
        "S02 prebuilt packet (Path A)": None,
        "S03 EEG flat for 2 h": lambda n: edf(n, inputs["flat2h"], expect=("REVIEW_ONLY", ["Q01"])),
        "S04 EOG channel missing (v1 uses EEG only)": lambda n: edf(n, inputs["no_eog"]),
        "S05 Fpz-Cz missing": lambda n: edf(n, inputs["no_fpz"], expect=("REVIEW_ONLY", ["W05"])),
        "S06 recorded at 128 Hz": lambda n: edf(n, inputs["128hz"]),
        "S07 only 2 h recorded": lambda n: edf(n, inputs["2h"], expect=("REJECTED", ["R03"])),
        "S08 corrupt header": lambda n: edf(n, inputs["corrupt"], expect=("REJECTED", ["R01"])),
        "S09 report model crashes once": lambda n: edf(n, inputs["base"], CrashOnce()),
        "S10 report model hangs": lambda n: edf(n, inputs["base"], Hang(), expect=("FALLBACK_SUMMARY", ["F02"])),
        "S11 report model writes invalid claims": lambda n: edf(n, inputs["base"], Garbage(), expect=("FALLBACK_SUMMARY", ["F03"])),
        "S12 device too hot": lambda n: edf(n, inputs["base"], services=hot, expect=("FALLBACK_SUMMARY", ["F01"])),
        "S13 power cut during report generation": None,
        # policy 0.2 (V5): 20 uV at 49 Hz gives ratio ~0.45, between warn (0.081) and review (1.0)
        "S14 strong 49 Hz line noise": lambda n: edf(n, inputs["linenoise"], expect=("VERIFIED_REPORT", ["W10"])),
        "S15 two nights of one subject (trend)": None,
    }
    rows = []
    for name, fn in scenarios.items():
        if name.startswith("S02"):
            rt = ROOT / "runs" / name
            r = make_runner("v1", None, "replay", rt)
            rec, expect = run_packet(r, dev_packet), ("VERIFIED_REPORT", [])
        elif name.startswith("S13"):
            rt = ROOT / "runs" / name
            import shutil
            shutil.rmtree(rt, ignore_errors=True)          # the crash must happen on a fresh run every time
            # this scenario tests crash recovery, not RAM: a policy copy without the free-RAM floor, so the
            # harness's own memory use cannot push the subprocess into the resource wait
            import yaml
            pol = yaml.safe_load((PACKAGE_DIR / "config" / "policy.yaml").read_text(encoding="utf-8"))
            pol["resources"]["min_free_ram_gb"]["value"] = 0
            pol_path = ROOT / "policy_no_ram_floor.yaml"
            pol_path.write_text(yaml.safe_dump(pol), encoding="utf-8")
            cmd = [sys.executable, "-m", "agentic", "run-edf", str(inputs["base"]), "--backend", "oracle",
                   "--runtime", str(rt), "--policy", str(pol_path)]
            first = subprocess.run(cmd, cwd=REPO_ROOT, env=dict(os.environ, **{ENV: "generate:during"}), capture_output=True, text=True)
            second = subprocess.run(cmd, cwd=REPO_ROOT, env={k: v for k, v in os.environ.items() if k != ENV}, capture_output=True, text=True)
            pages = list((rt / "outbox").glob("*.html"))
            rec = atomic.read_json(next((rt / "outbox").glob("*.json"))) if pages else {"outcome": None, "reasons": [], "run_id": None}
            expect = ("VERIFIED_REPORT", [])
            rec["_crash_exit"], rec["_deliveries"] = first.returncode, len(pages)
        elif fn is None:
            rows.append({"scenario": name, "status": "DEFERRED", "note": "needs the history store (Phase 6)"})
            print(f"{name:<45} DEFERRED (Phase 6: history store)", flush=True)
            continue
        else:
            rt, rec, expect = fn(name)
        inv = run_invariants(rt, rec["run_id"], manifest_sha) if rec.get("run_id") else {}
        ok = (rec["outcome"], rec["reasons"]) == expect and all(inv.values())
        if name.startswith("S13"):
            ok = ok and rec["_crash_exit"] == 137 and rec["_deliveries"] == 1
        rows.append({"scenario": name, "expected": list(expect), "got": [rec["outcome"], rec["reasons"]],
                     "invariants": inv, "status": "PASS" if ok else "FAIL"})
        print(f"{name:<45} expected {expect[0]:<17} {expect[1]!s:<8} got {rec['outcome']:<17} {rec['reasons']!s:<8} "
              f"invariants {'ok' if inv and all(inv.values()) else inv}  {rows[-1]['status']}", flush=True)
    static = static_invariants()
    summary = {"scenarios": len(rows), "pass": sum(r["status"] == "PASS" for r in rows),
               "fail": sum(r["status"] == "FAIL" for r in rows), "deferred": sum(r["status"] == "DEFERRED" for r in rows),
               "static_invariants": static}
    atomic.write_json(PACKAGE_DIR / "reports" / "p4_faults.json", {"summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))
    return 0 if summary["fail"] == 0 and all(v for k, v in static.items() if not k.endswith("offenders")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
