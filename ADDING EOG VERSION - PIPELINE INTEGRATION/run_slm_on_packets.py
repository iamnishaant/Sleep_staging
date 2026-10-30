"""
Run the FROZEN report model on a directory of evidence packets and score every output
with the frozen verifier - without touching deploy/audit_log.jsonl, deploy/reports or
deploy/human_review (deploy.server.run_pipeline writes to all three).

    # --cli (recommended, reproduces the frozen evaluation): llama-completion, one process per packet
    # without --cli: a running llama-server (does NOT reproduce the frozen outputs exactly)
    python run_slm_on_packets.py work_n6adv_packets/packets_dev  results/slm_dev
    python run_slm_on_packets.py work_n6adv_packets/packets_test results/slm_test --allow-test

What is reused UNCHANGED from the project: deploy.server._build_prompt, the generation
settings in deploy/frozen.json (temperature 0, seed 0, n_predict 1024), the component
integrity check (model/grammar/verifier/serializer hashes), report.evaluate.score_output
(verifier + coverage + numeric fidelity), report.render.render_report.
Only the HTTP call differs (it has a timeout), and outputs go under this folder.

Resumable: a packet whose raw output file exists is not re-run.
"""
import sys
sys.dont_write_bytecode = True
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
LOG = HERE / "TEST_ACCESS_LOG.md"

from deploy import server as S                      # noqa: E402  (read-only use)
from report.evaluate import score_output            # noqa: E402
from report import verify_report, render_report     # noqa: E402


def call_llm(prompt, url="http://127.0.0.1:8080/v1/chat/completions", timeout=1800):
    data = {"messages": [{"role": "user", "content": prompt}], "temperature": S.GEN["temperature"],
            "seed": S.GEN["seed"], "n_predict": S.GEN["n_predict"]}
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()
    return text, round(time.monotonic() - t0, 1)


def call_cli(packet, out_dir):
    """EXACTLY the frozen evaluation's path (candidates/run.py): a fresh llama-completion process per
    packet, the model's embedded chat template, --jinja -cnv -st, the frozen grammar, -n 3000 -c 12288,
    --temp 0 --seed 0. Reproduces the cached outputs byte for byte; llama-server does not (8 of 31 dev
    nights differ)."""
    from candidates import run as C
    from reference.prompts import build
    (out_dir / "prompts").mkdir(exist_ok=True)
    pf = out_dir / "prompts" / f"{packet['recording_id']}.txt"
    pf.write_text(build(C.PROMPT_ID, packet), encoding="utf-8", newline="\n")
    r = C.exec_llama(C.command(C.MODELS["Llama-Student"], pf, "constrained", None))
    text, note = C.extract_output(r["stdout"])
    if r["exit_status"] != 0 or note != "ok":
        raise SystemExit(f"{packet['recording_id']}: llama-completion exit {r['exit_status']}, {note}")
    return text, r["wall_s"]


def main(argv):
    skip = {argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--limit"}
    pos = [a for a in argv if not a.startswith("--") and a not in skip]
    if len(pos) != 2:
        raise SystemExit(__doc__)
    pkt_dir, out_dir = Path(pos[0]).resolve(), Path(pos[1]).resolve()
    if "test" in pkt_dir.name and "--allow-test" not in argv:
        raise SystemExit("test packets need --allow-test (and are logged)")
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    S.verify_component_integrity()                      # frozen model / grammar / verifier / serializer
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)
    files = sorted(pkt_dir.glob("*.json"))[:limit]
    if "test" in pkt_dir.name:
        if not LOG.exists():
            LOG.write_text("# Test-set access log (ADDING EOG VERSION - PIPELINE INTEGRATION)\n\n", encoding="utf-8")
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"\n## SLM RUN ON TEST PACKETS — {dt.datetime.now():%Y-%m-%d %H:%M}\n\n"
                     f"Frozen report model on {len(files)} test packets from {pkt_dir.name}; scored by the frozen verifier.\n")
    rows = []
    for k, f in enumerate(files, 1):
        packet = json.loads(f.read_text(encoding="utf-8"))
        rec = packet["recording_id"]
        raw_p, wall = out_dir / "raw" / f"{rec}.txt", None
        if raw_p.exists():
            raw = raw_p.read_text(encoding="utf-8")
        else:
            raw, wall = (call_cli(packet, out_dir) if "--cli" in argv
                         else call_llm(S._build_prompt(packet)))
            raw_p.write_text(raw, encoding="utf-8")
        res = score_output(raw, packet)
        d = res.as_dict(); d["llm_wall_s"] = wall
        if res.passed:
            v = verify_report(raw, packet)
            (out_dir / "reports" / f"{rec}.html").write_text(render_report(v, packet), encoding="utf-8")
        rows.append(d)
        print(f"[{k:2d}/{len(files)}] {rec}: {'PASS' if res.passed else 'FAIL'} "
              f"mandatory {d['mandatory_coverage']} claims {d['n_claims']} violations {d['violations']}"
              f"{'' if wall is None else f' ({wall}s)'}", flush=True)
        (out_dir / "results.json").write_text(json.dumps(rows, indent=1))
    n = len(rows)
    full = sum(1 for r in rows if r["passed"] and r["mandatory_coverage"] == 1.0)
    print(f"\n{n} packets | verifier PASS {sum(r['passed'] for r in rows)} | at 5/5 mandatory and 0 violations {full} | "
          f"violations total {sum(len(r['violations']) for r in rows)}")


if __name__ == "__main__":
    main(sys.argv[1:])
