"""Flash-Lite feasibility pilot: ONE dev packet under P1, at the reference run's fixed settings.

EXPLORATORY. It informs the choice of reference model; it is not a measurement,
and nothing from it enters the reference set. Its cache and log live under
reference/pilot/, apart from the reference run, and every attempt is logged
under the candidate's own model name, so none of it counts against the
reference model's daily budget.

    python -m reference.pilot            the plan; sends nothing
    python -m reference.pilot --go       at most MAX_ATTEMPTS attempts, then score

The packet is dev SC4111E0 - the local runs' night, and one the reference
model has already answered under P1, so the two are directly comparable.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from report import render_report, verify_report
from report.evaluate import PACKET_DIRS, evaluate
from report.render import RenderRefused
from report.verify_structure import verify_structure
from .config import CANDIDATE_MODELS
from .run import DEV_MANIFEST, Runner, dev_jobs, load_cached
from .schema import validate

HERE = Path(__file__).resolve().parent
PILOT_DIR = HERE / "pilot"
MAX_ATTEMPTS = 4                     # the pilot's own cap, as its daily budget
PILOT_MODEL = CANDIDATE_MODELS["flash_lite"]
PACKET = "SC4111E0-PSG"
PROMPT = "P1"


def runner() -> Runner:
    jobs = [(p, pk) for p, pk in dev_jobs() if p == PROMPT and pk["recording_id"] == PACKET]
    assert len(jobs) == 1
    return Runner(model=PILOT_MODEL, max_attempts_today=MAX_ATTEMPTS, jobs=jobs,
                  cache_dir=PILOT_DIR / "cache", log_dir=PILOT_DIR / "logs")


def score(entry: dict) -> dict:
    pk = json.loads((PACKET_DIRS["val"] / f"{PACKET}.json").read_text(encoding="utf-8"))
    text = entry["text"]
    out = {"finish_reason": entry["finish_reason"], "usage": entry["usage"]}
    try:
        claims = json.loads(text)
    except ValueError as e:
        claims, out["json"] = None, f"FAILED: {e}"
    if claims is not None:
        out["json"] = f"parsed: {len(claims)} claims"
        out["schema"] = validate(claims) or "valid"
        out["layer1"] = sorted({str(v.code) for v in verify_structure(claims)}) or "clean"
    r = verify_report(text, pk)
    out["per_rule"] = dict(Counter(r.codes))
    out["verified"] = f"{len(r.enriched)} of {len(r.claims) if isinstance(r.claims, list) else 0}"
    cl = [c for c in (r.claims or []) if isinstance(c, dict)] if isinstance(r.claims, list) else []
    out["hedged_value"] = sum(c.get("claim_type") == "hedged_value" for c in cl)
    try:
        render_report(r, pk)
        out["renders"] = True
    except RenderRefused:
        out["renders"] = False

    ev = PILOT_DIR / "eval"
    (ev / "outputs").mkdir(parents=True, exist_ok=True)
    rows = [row for row in json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
            if row["recording_id"] == PACKET]
    (ev / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    (ev / "outputs" / f"{PACKET}.json").write_text(text, encoding="utf-8")
    d = evaluate(ev / "outputs", ev / "manifest.json").as_dict()
    pr, ov = d["per_recording"][0], d["strata"]["overall"]
    out |= {k: pr[k] for k in ("mandatory_coverage", "mandatory_missing", "cited_mandatory",
                               "discretionary_coverage", "cited_discretionary",
                               "oracle_recovery", "unrecovered_available")}
    out["numeric_fidelity"] = f"{pr['fidelity_matched']}/{pr['fidelity_total']}"
    out["numeric_fidelity_rate"] = ov["numeric_fidelity"]
    return out


def attempts() -> dict:
    p = PILOT_DIR / "logs" / "run.jsonl"
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    return {"attempts": len(rows), "by_status": dict(Counter(r["http_status"] for r in rows)),
            "retry_causes": [r["retry_cause"] for r in rows if r.get("retry")],
            "thinking_tokens": [r.get("thinking_tokens") for r in rows if r["http_status"] == 200]}


def main(argv: list[str]) -> int:
    rn = runner()
    print(f"pilot: {PROMPT} on {PACKET}, model {PILOT_MODEL}, at most {MAX_ATTEMPTS} attempts")
    print(f"resolved request: {json.dumps({k: v for k, v in rn.request_kwargs().items() if k != 'schema'})}"
          f" + the Item 0 schema")
    if "--go" in argv:
        s = rn.run()
        print(f"run: sent {s['sent']}, attempts {s['attempts']}, stopped {s['stopped']}, "
              f"last error {s['last_error']}")
    print(f"log: {attempts()}")
    _, _, _, key = next(rn.keyed_jobs())
    entry = load_cached(rn.cache_dir, key)
    if entry is None:
        print("no cached response - nothing to score")
        return 1
    for k, v in score(entry).items():
        print(f"  {k:<24} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
