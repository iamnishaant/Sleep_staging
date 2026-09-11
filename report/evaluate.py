"""Score a directory of model outputs against the packets a manifest names.

    evaluate(outputs_dir, manifest_path) -> EvaluationReport

Runs before any model exists: it is calibrated on oracle witnesses, which must
score perfectly, and on synthetic corruptions, which must score exactly as
predicted. If it cannot score a known-perfect input perfectly, nothing it
reports later is trustworthy.

MANIFEST-DRIVEN, NEVER GLOB-DRIVEN. The manifest decides which recordings are
in scope. `outputs_dir` is only ever read at `<recording_id>.json` for ids the
manifest lists - a stray file there cannot enter the evaluation, for the same
reason tests/_packets.py stopped globbing in 2A.

STRATIFIED BY NIGHT-CONFIDENCE TIER, which is the reason this step exists. Dev
is 11/10/10 high/medium/low; test is 12/4/13. Night tier correlates with
staging kappa (high 0.7223, medium 0.5984, low 0.5471), so dev is measurably the
easier split and every Phase 2 selection decision is made on it. That the
reporting task is tier-invariant is an argument; this makes it a measurement.
Every metric is reported overall and per tier, with n beside every cell and a
flag on any cell whose stratum has fewer than SMALL_N packets.

NONE, NOT ZERO, WHERE A METRIC HAS NO DENOMINATOR. numeric_fidelity with no
value claims, oracle_recovery with nothing available, unsupported_claim_rate
with no claims. A model that emits only observations has not transcribed
perfectly; it has not transcribed. A null leaves the output out of the mean; a
zero or a one would misreport it.

CO-OCCURRENCE IS DATA. One output can violate several rules, so violation counts
legitimately exceed failed-output counts. Every violation is kept. Recording
only the first per output would make per-rule rates sum neatly to the failure
rate - and destroy the per-rule breakdown the 30 codes exist to provide.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import verify_report
from .claim_schema import evidence_index
from .coverage import cited_ids, coverage
from .oracle import oracle
from .violations import V

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "distillation" / "results"

# Manifest `split` value -> packet directory. The dev manifest's rows say
# "val" because the dev set IS the upstream validation split.
PACKET_DIRS = {"test": RES / "packets", "val": RES / "phase2_dev_packets"}

TIERS = ("high", "medium", "low")
STRATA = ("overall",) + TIERS
SMALL_N = 5                              # packets, the unit of independence
ALL_CODES = tuple(str(c) for c in V)     # all 30, in enum order

VALUE_TYPES = ("value", "hedged_value")

# ---------------------------------------------------------------------------
# unsupported_claim_rate is a NAMED AGGREGATE of codes the verifier already
# produces - not a new semantic definition alongside the verifier's, and never
# classified from message text. One place, identical for dev and test.
#
# The line drawn: a claim is UNSUPPORTED when it asserts content the cited
# evidence does not back. That takes in
#   - citing something that is not evidence        L1.unknown_evidence_id
#   - citing evidence this packet does not have     L2.cited_id_not_in_packet
#   - a number the packet does not hold             L2.value_mismatch
#   - a unit the packet does not state              L2.unit_mismatch
#   - a quantity from nowhere it cites              L2.uncited_quantity
#   - an observation the packet contradicts         L2.text_key_predicate_false
#   - an observation without its evidence cited     L2.text_key_dependency_missing
#   - an association without associative evidence   L2.not_associative_evidence
#
# and leaves out three other kinds of failure, each measured elsewhere:
#   - FORM: every other L1 code. A malformed claim asserts nothing.
#   - CONFIDENCE LEVEL: unsafe_item_not_hedged, safe_item_hedged. The fact is
#     supported; it is stated at the wrong confidence.
#   - PRESENTATION / COMBINATION: stage_tier_unavailable, rem_latency_double_
#     count, rem_error_differenced, missing_review_flag, bad_subject_for_type,
#     tier_item_not_valuable. rem_error_differenced asserts a derived quantity,
#     but it always co-fires with uncited_quantity, so leaving it out loses
#     nothing and keeps the set to one idea.
# ---------------------------------------------------------------------------
UNSUPPORTED_CODES = frozenset({
    str(V.UNKNOWN_EVIDENCE_ID),
    str(V.CITED_ID_NOT_IN_PACKET),
    str(V.VALUE_MISMATCH),
    str(V.UNIT_MISMATCH),
    str(V.UNCITED_QUANTITY),
    str(V.TEXT_KEY_PREDICATE_FALSE),
    str(V.TEXT_KEY_DEPENDENCY_MISSING),
    str(V.NOT_ASSOCIATIVE_EVIDENCE),
})


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _mean(xs):
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else None


def _div(a, b):
    return (a / b) if b else None


class OutputResult:
    """One recording's output, scored."""

    __slots__ = ("recording_id", "subject_id", "cohort", "tier", "present",
                 "n_claims", "codes", "schema_valid", "passed", "coverage",
                 "oracle_recovery", "unrecovered_available",
                 "fidelity_matched", "fidelity_total", "unsupported_claims")

    @property
    def failed(self) -> bool:
        return self.present and bool(self.codes)

    def as_dict(self) -> dict:
        r = lambda x: None if x is None else round(x, 6)
        return {
            "recording_id": self.recording_id,
            "subject_id": self.subject_id,
            "cohort": self.cohort,
            "tier": self.tier,
            "present": self.present,
            "n_claims": self.n_claims,
            "violations": list(self.codes),
            "schema_valid": self.schema_valid,
            "passed": self.passed,
            "mandatory_coverage": r(self.coverage.mandatory),
            "mandatory_missing": sorted(self.coverage.mandatory_missing),
            "discretionary_coverage": r(self.coverage.discretionary),
            "oracle_recovery": r(self.oracle_recovery),
            "unrecovered_available": self.unrecovered_available,
            "fidelity_matched": self.fidelity_matched,
            "fidelity_total": self.fidelity_total,
            "unsupported_claims": self.unsupported_claims,
        }


def score_output(raw, packet: dict) -> OutputResult:
    """Score one output. `raw` is the file's text, or None if there was none.

    A MISSING output is not an empty one. An empty array is a model that chose
    to say nothing; a missing file is a model that produced nothing usable. It
    scores zero coverage and does not pass, but it has no violations, so it is
    counted separately as `n_missing` rather than folded into the failures.
    """
    res = OutputResult()
    res.recording_id = packet.get("recording_id")
    res.subject_id = packet.get("subject_id")
    res.cohort = packet.get("cohort")
    res.tier = packet.get("night_confidence", {}).get("tier")
    res.present = raw is not None

    o = oracle(packet)
    if raw is None:
        res.n_claims, res.codes = 0, ()
        res.schema_valid = res.passed = False
        res.coverage = coverage(packet, [])
        res.oracle_recovery = o.recovery(set(), packet)
        res.unrecovered_available = o.unrecovered_available(set(), packet)
        res.fidelity_matched = res.fidelity_total = 0
        res.unsupported_claims = 0
        return res

    r = verify_report(raw, packet)
    claims = [c for c in (r.claims or []) if isinstance(c, dict)] \
        if isinstance(r.claims, list) else []
    res.n_claims = len(claims)
    res.codes = tuple(str(v.code) for v in r.violations)
    res.schema_valid = not any(c.startswith("L1.") for c in res.codes)
    res.passed = not res.codes

    res.coverage = coverage(packet, r.enriched)
    cited = cited_ids(r.enriched)
    res.oracle_recovery = o.recovery(cited, packet)
    res.unrecovered_available = o.unrecovered_available(cited, packet)

    # numeric fidelity: value/hedged_value claims comparable to a packet value,
    # compared with the same exactness rule 2 uses - no tolerance of any kind.
    idx = evidence_index(packet)
    matched = total = 0
    for c in claims:
        cites = c.get("cites")
        if (c.get("claim_type") in VALUE_TYPES and isinstance(cites, list)
                and len(cites) == 1 and cites[0] in idx
                and _is_number(c.get("value"))):
            total += 1
            matched += c["value"] == idx[cites[0]].get("value")
    res.fidelity_matched, res.fidelity_total = matched, total

    flagged = {v.claim_id for v in r.violations
               if str(v.code) in UNSUPPORTED_CODES and v.claim_id is not None}
    res.unsupported_claims = sum(1 for c in claims if c.get("claim_id") in flagged)
    return res


def stratum_metrics(results) -> dict:
    """Every metric, over one stratum. Counts are exact; rates may be None."""
    results = list(results)
    n = len(results)
    present = [x for x in results if x.present]
    schema_valid = [x for x in results if x.schema_valid]
    per_rule = {}
    for code in ALL_CODES:
        count = sum(x.codes.count(code) for x in results)
        outputs = sum(1 for x in results if code in x.codes)
        per_rule[code] = {"count": count, "outputs": outputs,
                          "output_rate": _div(outputs, n)}
    recov = [x.oracle_recovery for x in results if x.oracle_recovery is not None]
    fm = sum(x.fidelity_matched for x in results)
    ft = sum(x.fidelity_total for x in results)
    uc = sum(x.unsupported_claims for x in results)
    nc = sum(x.n_claims for x in results)
    return {
        "n": n,
        "small_n": n < SMALL_N,
        "n_missing": n - len(present),
        "n_passed": sum(1 for x in results if x.passed),
        "n_failed": sum(1 for x in results if x.failed),
        "n_schema_valid": len(schema_valid),
        "total_violations": sum(len(x.codes) for x in results),
        # validity
        "schema_validity_rate": _div(len(schema_valid), n),
        "policy_pass_rate": _div(sum(1 for x in schema_valid if x.passed),
                                 len(schema_valid)),
        "overall_pass_rate": _div(sum(1 for x in results if x.passed), n),
        # coverage
        "mandatory_coverage": _mean(x.coverage.mandatory for x in results),
        "n_mandatory_full": sum(1 for x in results
                                if x.coverage.mandatory_complete),
        "discretionary_coverage": _mean(x.coverage.discretionary for x in results),
        "oracle_recovery": _mean(recov),
        "oracle_recovery_n": len(recov),
        "unrecovered_available": _mean(x.unrecovered_available for x in results),
        # fidelity
        "numeric_fidelity": _div(fm, ft),
        "numeric_fidelity_claims": ft,
        "unsupported_claim_rate": _div(uc, nc),
        "unsupported_claims": uc,
        "n_claims": nc,
        # per rule, all 30
        "per_rule": per_rule,
    }


class EvaluationReport:
    def __init__(self, results, manifest_path, outputs_dir):
        self.results = list(results)
        self.manifest_path = str(manifest_path)
        self.outputs_dir = str(outputs_dir)
        self.strata = {"overall": stratum_metrics(self.results)}
        for t in TIERS:
            self.strata[t] = stratum_metrics(x for x in self.results if x.tier == t)

    def as_dict(self) -> dict:
        def clean(v):
            if isinstance(v, float):
                return round(v, 6)
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            return v
        return {
            "manifest": Path(self.manifest_path).name,
            "n_recordings": len(self.results),
            "small_n_threshold": SMALL_N,
            "unsupported_codes": sorted(UNSUPPORTED_CODES),
            "strata": {k: clean(v) for k, v in self.strata.items()},
            "per_recording": [x.as_dict() for x in self.results],
        }

    def to_json(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path

    # ------------------------------------------------------------------ table
    ROWS = (
        ("schema_validity_rate", "schema validity"),
        ("policy_pass_rate", "policy pass (of schema-valid)"),
        ("overall_pass_rate", "overall pass"),
        ("mandatory_coverage", "mandatory coverage"),
        ("n_mandatory_full", "  packets at 5/5"),
        ("discretionary_coverage", "discretionary coverage (/14)"),
        ("oracle_recovery", "oracle recovery"),
        ("unrecovered_available", "unrecovered available (mean)"),
        ("numeric_fidelity", "numeric fidelity"),
        ("unsupported_claim_rate", "unsupported claim rate"),
        ("total_violations", "violations (count)"),
        ("n_failed", "failed outputs"),
        ("n_missing", "missing outputs"),
    )

    def _cell(self, stratum: str, key: str) -> str:
        m = self.strata[stratum]
        v = m[key]
        if v is None:
            val = "   -  "
        elif isinstance(v, float):
            val = f"{v:6.3f}"
        else:
            val = f"{v:6d}"
        flag = " !" if m["small_n"] else "  "
        return f"{val} n={m['n']:<3}{flag}"

    def format_table(self) -> str:
        w = 30
        head = f"{'metric':<{w}}" + "".join(f"{s:>16}" for s in STRATA)
        lines = [f"evaluation of {len(self.results)} outputs "
                 f"({Path(self.manifest_path).name})", "", head, "-" * len(head)]
        for key, label in self.ROWS:
            lines.append(f"{label:<{w}}" +
                         "".join(f"{self._cell(s, key):>16}" for s in STRATA))
        lines += ["", "violations by rule (count; every co-occurring violation "
                      "is kept)", f"{'code':<{w}}" +
                  "".join(f"{s:>16}" for s in STRATA),
                  "-" * len(head)]
        for code in ALL_CODES:
            row = f"{code:<{w}}"
            for s in STRATA:
                m = self.strata[s]
                c = m["per_rule"][code]["count"]
                flag = " !" if m["small_n"] else "  "
                row += f"{c:>6d} n={m['n']:<3}{flag}".rjust(16)
            lines.append(row)
        lines += [
            "",
            f"!  stratum has fewer than {SMALL_N} packets - too few to quote "
            f"without this warning attached.",
            "n  counts PACKETS, the unit of independence. Claim-level metrics "
            "are flagged by their stratum's packet count, because a stratum's "
            "claims are clustered within its packets.",
            "-  no denominator: None, not zero.",
            f"unsupported = {', '.join(sorted(UNSUPPORTED_CODES))}",
        ]
        return "\n".join(lines)


def evaluate(outputs_dir, manifest_path, packets_dir=None) -> EvaluationReport:
    """Evaluate every recording the manifest names, and nothing else."""
    outputs_dir, manifest_path = Path(outputs_dir), Path(manifest_path)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for row in rows:
        rec, split = row["recording_id"], row["split"]
        if packets_dir is None and split not in PACKET_DIRS:
            raise ValueError(f"{manifest_path.name}: unknown split {split!r} "
                             f"for {rec}; known: {sorted(PACKET_DIRS)}")
        pdir = Path(packets_dir) if packets_dir is not None else PACKET_DIRS[split]
        packet = json.loads((pdir / f"{rec}.json").read_text(encoding="utf-8"))
        if packet.get("recording_id") != rec:
            raise ValueError(f"{pdir / rec}.json holds "
                             f"{packet.get('recording_id')!r}, not {rec!r}")
        out = outputs_dir / f"{rec}.json"
        raw = out.read_text(encoding="utf-8") if out.exists() else None
        results.append(score_output(raw, packet))
    return EvaluationReport(results, manifest_path, outputs_dir)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outputs_dir")
    ap.add_argument("manifest")
    ap.add_argument("--json", help="write the JSON artefact here")
    a = ap.parse_args(argv)
    rep = evaluate(a.outputs_dir, a.manifest)
    print(rep.format_table())
    if a.json:
        print(f"\nwritten -> {rep.to_json(a.json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
