"""Edge simulation measurements. Runs INSIDE the container (python 3.12), started by run_all.ps1.

    python /sim/measure.py check              environment, release pins, no network, no secrets
    python /sim/measure.py pathb [--backend oracle|llama] [--nights N]
                                              raw recordings end to end; staging compared with the reference PC
    python /sim/measure.py patha [--nights N] report model on the dev evidence packets; text compared with the reference PC
    python /sim/measure.py summary            everything measured so far -> /out/SUMMARY.md

Every step writes /out/<step>.json and can be re-run: the agent re-delivers a finished night without
recomputing it (same run id), so an interrupted step resumes. Paths are the container mounts; for a test
outside Docker set SIM_REPO / SIM_DATA / SIM_REF / SIM_OUT.
"""
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

REPO = Path(os.environ.get("SIM_REPO", "/repo"))
DATA = Path(os.environ.get("SIM_DATA", "/data"))
REF = Path(os.environ.get("SIM_REF", "/sim/ref"))
OUT = Path(os.environ.get("SIM_OUT", "/out"))
PACKETS = REPO / "ADDING EOG VERSION - PIPELINE INTEGRATION" / "work_n6adv_packets" / "packets_dev"
ARCH = platform.machine()                                     # x86_64 or aarch64


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def tracked(cmd: list[str], log: Path) -> dict:
    """Run one agent command; wall time and the peak memory of its whole process tree (agent + llama)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    t0, peak = time.monotonic(), 0
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                peak = max(peak, sum(x.memory_info().rss for x in [proc, *proc.children(recursive=True)]))
            except psutil.Error:
                pass
            time.sleep(0.5)
    text = log.read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"^(\S+): (\S+) (.*?)-> outbox/(\w+)\.html", text, re.M)
    rec, outcome, reasons, run_id = m[-1] if m else (None, None, "", None)
    return {"exit": p.returncode, "wall_s": round(time.monotonic() - t0, 1), "peak_gb": round(peak / 1e9, 2),
            "outcome": outcome, "reasons": reasons.strip(" []'\""), "run_id": run_id,
            "error_tail": None if p.returncode == 0 else text[-800:]}


def stats(rows: list[dict]) -> dict:
    w = [r["wall_s"] for r in rows if r.get("exit") == 0]
    pk = [r["peak_gb"] for r in rows if r.get("exit") == 0]
    out = {"nights": len(rows), "failed_runs": sum(r.get("exit") != 0 for r in rows)}
    if w:
        out |= {"wall_s_median": float(np.median(w)), "wall_s_p95": float(np.percentile(w, 95)),
                "peak_gb_max": max(pk), "peak_gb_median": float(np.median(pk))}
    for o in ("VERIFIED_REPORT", "FALLBACK_SUMMARY", "REVIEW_ONLY", "REJECTED"):
        out[o] = sum(r.get("outcome") == o for r in rows)
    return out


# ---------------------------------------------------------------- check
def check() -> int:
    res = {"arch": ARCH, "python": sys.version.split()[0], "cpus_visible": os.cpu_count()}
    for f, key in (("/sys/fs/cgroup/cpu.max", "cgroup_cpu_max"), ("/sys/fs/cgroup/memory.max", "cgroup_memory_max")):
        res[key] = Path(f).read_text().strip() if Path(f).exists() else None
    res["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
    exe = os.environ.get("AGENTIC_LLAMA_BIN", "")
    try:
        v = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
        res["llama_version"] = (v.stdout + v.stderr).strip().splitlines()[-1:] or None
        res["llama_ok"] = v.returncode == 0
    except Exception as e:  # noqa: BLE001
        res["llama_version"], res["llama_ok"] = f"unavailable: {e}", False
    m = subprocess.run([sys.executable, "-m", "agentic", "manifest", "--release", "v2"],
                       capture_output=True, text=True, cwd=REPO)
    res["release_pins_ok"] = m.returncode == 0
    res["release_manifest"] = (json.loads(m.stdout).get("sha256", "")[:16] if m.returncode == 0 else m.stderr[-500:])
    # P1: no network. Both a raw connection and a DNS lookup must fail.
    net = {}
    for name, fn in (("tcp_1.1.1.1_53", lambda: socket.create_connection(("1.1.1.1", 53), timeout=3)),
                     ("dns_pypi.org", lambda: socket.gethostbyname("pypi.org"))):
        try:
            fn(); net[name] = "REACHABLE"
        except OSError as e:
            net[name] = f"blocked ({type(e).__name__})"
    res["network"] = net
    res["network_off"] = all(v.startswith("blocked") for v in net.values())
    # P4: no secrets in what the container can see (the mounted bundle and the image).
    key = re.compile(rb"AIza[0-9A-Za-z_\-]{35}")
    hits = []
    for root in (REPO, Path("/sim"), Path("/opt"), Path("/root")):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.name == ".env":
                hits.append(str(p))
            elif p.is_file() and p.stat().st_size < 20_000_000:
                try:
                    if key.search(p.read_bytes()):
                        hits.append(str(p))
                except OSError:
                    pass
    res["secret_hits"] = hits
    res["no_secrets"] = not hits
    save(f"check_{ARCH}", res)
    print(json.dumps(res, indent=1))
    return 0 if res["release_pins_ok"] and res["network_off"] and res["no_secrets"] else 1


# ---------------------------------------------------------------- Path B: raw recordings
def find_edf(rec: str) -> Path | None:
    for sub in ("sleep-cassette", "sleep-telemetry"):
        p = DATA / sub / f"{rec}.edf"
        if p.exists():
            return p
    return None


_o = REF / "outcomes_llama.json"
REF_OUTCOMES = json.loads(_o.read_text()) if _o.exists() else {}   # this PC's delivery outcome per night (llama backend)


def pathb(backend: str, nights: int | None) -> int:
    crops = json.loads((REF / "crops.json").read_text())         # the reference PC's exact night crops
    recs = sorted(crops)[:nights] if nights else sorted(crops)
    rt = OUT / f"rt_pathb_{backend}_{ARCH}"
    rows = []
    for i, rec in enumerate(recs, 1):
        edf = find_edf(rec)
        if edf is None:
            rows.append({"recording_id": rec, "exit": None, "error_tail": "EDF missing"}); continue
        c = crops[rec]
        r = tracked([sys.executable, "-m", "agentic", "run-edf", str(edf), "--release", "v2", "--backend", backend,
                     "--runtime", str(rt), "--crop-start-s", str(c["start_s"]), "--crop-hours", str(c["hours"])],
                    OUT / "logs" / f"pathb_{backend}_{ARCH}_{rec}.log")
        r["recording_id"] = rec
        run = rt / "runs" / (r["run_id"] or "-")
        if not r["run_id"]:                    # crashed before delivery: staging output is still in its run folder
            hits = [m.parent.parent for m in (rt / "runs").glob("*/input/meta.json")
                    if json.loads(m.read_text()).get("recording_id") == rec]
            run = max(hits, key=lambda d: d.stat().st_mtime) if hits else run
        par = {}
        for k in ("pass1", "pass2"):           # pass1 = raw model output (every night); pass2 only past the quality gates
            got, ref = sorted(run.glob(f"*_stage_{k}/{k}.npy")), REF / f"{k}_reference" / f"{rec}.npy"
            if got and ref.exists():
                a, b = np.load(got[0]), np.load(ref)
                same = a.shape == b.shape
                par[k] = {"max_abs_diff": float(np.abs(a - b).max()) if same else None,
                          "argmax_identical": bool(same and (a.argmax(-1) == b.argmax(-1)).all())}
            elif got or ref.exists():
                par[k] = {"max_abs_diff": None, "argmax_identical": False, "note": "present on one PC only"}
        if par:
            r["parity"] = par
        if backend == "llama" and REF_OUTCOMES.get(rec):
            r["outcome_matches_reference"] = r.get("outcome") == REF_OUTCOMES[rec]
        rows.append(r)
        print(f"[{i}/{len(recs)}] {rec}: {r.get('outcome')} {r['wall_s']} s, peak {r['peak_gb']} GB, parity {r.get('parity')}", flush=True)
        save(f"pathb_{backend}_{ARCH}", {"backend": backend, "arch": ARCH, "summary": stats(rows), "rows": rows})
    par = [p for r in rows for p in r.get("parity", {}).values()]
    s = stats(rows) | {"parity_nights": sum("parity" in r for r in rows), "parity_arrays_compared": len(par),
                       "parity_max_abs_diff": max((p["max_abs_diff"] for p in par if p["max_abs_diff"] is not None), default=None),
                       "parity_hypnograms_identical": sum(p["argmax_identical"] for p in par),
                       "E1_pass": bool(par) and all(p["argmax_identical"] and p["max_abs_diff"] is not None
                                                     and p["max_abs_diff"] <= 1e-4 for p in par)}
    if backend == "llama":
        s["outcomes_matching_reference"] = sum(bool(r.get("outcome_matches_reference")) for r in rows)
    save(f"pathb_{backend}_{ARCH}", {"backend": backend, "arch": ARCH, "summary": s, "rows": rows})
    print(json.dumps(s, indent=1))
    return 0


# ---------------------------------------------------------------- Path A: report model on packets
def patha(nights: int | None) -> int:
    packets = sorted(PACKETS.glob("*.json"))[:nights] if nights else sorted(PACKETS.glob("*.json"))
    rt = OUT / f"rt_patha_{ARCH}"
    rows = []
    for i, p in enumerate(packets, 1):
        r = tracked([sys.executable, "-m", "agentic", "run-packet", str(p), "--release", "v2", "--backend", "llama",
                     "--runtime", str(rt)], OUT / "logs" / f"patha_{ARCH}_{p.stem}.log")
        r["recording_id"] = p.stem
        raw = sorted((rt / "runs" / (r["run_id"] or "-")).glob("*_generate/raw.txt"))
        ref = REF / "reports_reference" / f"{p.stem}.json"
        if raw and ref.exists():
            r["text_identical_to_reference"] = raw[0].read_text(encoding="utf-8") == json.loads(ref.read_text())["text"]
        rows.append(r)
        print(f"[{i}/{len(packets)}] {p.stem}: {r.get('outcome')} {r['wall_s']} s, peak {r['peak_gb']} GB, "
              f"same text as reference: {r.get('text_identical_to_reference')}", flush=True)
        save(f"patha_{ARCH}", {"arch": ARCH, "summary": stats(rows), "rows": rows})
    s = stats(rows) | {"text_identical_to_reference": sum(bool(r.get("text_identical_to_reference")) for r in rows)}
    save(f"patha_{ARCH}", {"arch": ARCH, "summary": s, "rows": rows})
    print(json.dumps(s, indent=1))
    return 0


# ---------------------------------------------------------------- summary
def summary() -> int:
    lines = ["# Edge simulation results", ""]
    for f in sorted(OUT.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        lines += [f"## {f.stem}", "", "```", json.dumps(d.get("summary", d), indent=1, default=str)[:3000], "```", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "check"
    args = sys.argv[2:]
    opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default  # noqa: E731
    n = int(opt("--nights")) if opt("--nights") else None
    if step == "check":
        raise SystemExit(check())
    if step == "pathb":
        raise SystemExit(pathb(opt("--backend", "oracle"), n))
    if step == "patha":
        raise SystemExit(patha(n))
    if step == "summary":
        raise SystemExit(summary())
    raise SystemExit(f"unknown step {step!r}")
