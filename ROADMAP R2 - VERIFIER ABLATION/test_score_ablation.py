"""Self-check of score_one on a real cached output and corrupted copies of it.

    python -m pytest "ROADMAP R2 - VERIFIER ABLATION/test_score_ablation.py" -q
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from candidates.run import CACHE_DIR, CandidateRun  # noqa: E402
from score_ablation import score_one  # noqa: E402


def _clean_pair():
    run = CandidateRun("Llama-Student", cache_dir=CACHE_DIR, exec_fn=None, arm="constrained")
    for pk, _, key in run.keyed():
        e = run.cached(key)
        if e and e["exit_status"] == 0 and score_one(e["text"], pk, gated=True)["released"]:
            return e["text"], pk
    raise AssertionError("no verifier-passing cached output found")


def test_clean_output_is_released_and_not_flawed():
    text, pk = _clean_pair()
    a, b = score_one(text, pk, gated=True), score_one(text, pk, gated=False)
    assert a["released"] and not a["flawed"]
    assert b["released"] and not b["flawed"] and b["mandatory_missing"] == 0


def test_wrong_number_is_counted_and_gate_withholds():
    text, pk = _clean_pair()
    claims = json.loads(text)
    c = next(c for c in claims if c["claim_type"] in ("value", "hedged_value"))
    c["value"] = c["value"] + 1
    bad = json.dumps(claims)
    assert score_one(bad, pk, gated=False)["wrong_number"] == 1
    assert score_one(bad, pk, gated=False)["flawed"]
    assert not score_one(bad, pk, gated=True)["released"]


def test_malformed_text_is_a_malformed_release():
    _, pk = _clean_pair()
    s = score_one("Total sleep time was about seven hours.", pk, gated=False)
    assert s["malformed"] and s["flawed"] and s["released"]


def test_missing_mandatory_item_is_flawed():
    text, pk = _clean_pair()
    claims = [c for c in json.loads(text) if "arch.total_sleep_time" not in c.get("cites", [])]
    s = score_one(json.dumps(claims), pk, gated=False)
    assert s["mandatory_missing"] >= 1 and s["flawed"]
