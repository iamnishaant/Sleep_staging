"""The reference model's structured-output schema, GENERATED from the claim schema.

Local candidates decode under report/claims.gbnf. The reference model decodes
under the provider's `responseSchema`, an OpenAPI 3.0 subset. Both are generated
from the same constants - report/claim_schema.py, and the id and unit tuples
report/gbnf.py derives from it - so they cannot drift apart silently.
tests/test_reference_schema.py asserts the committed response_schema.json is
byte-identical to this generator's output, and compares its acceptance with the
grammar's, claim by claim.

What this schema does NOT do is encode Layer 2 policy. A `value` claim may cite
any of the 17 numeric ids, safe or unsafe, exactly as the grammar allows. Packet
membership, value equality, safe/unsafe hedging, key dependencies and predicates
stay with the common deterministic verifier: the shared enforcement point that
keeps the comparison with the local candidates fair. Where this dialect is less
expressive than GBNF, the gap is documented (PHASE2_NOTES) and left to that
verifier - never closed by adding policy here, and never by weakening the GBNF.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from report.claim_schema import (BASE_FIELDS, CLAIM_ID_RE, CLAIM_TYPES,
                                 EVIDENCE_ID_VOCAB, REASON_KEYS, TEXT_KEYS)
from report.gbnf import NUMERIC_IDS, NUMERIC_UNITS, TIER_IDS

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "response_schema.json"

# Which ids each claim type may cite: the same per-type scopes claims.gbnf
# encodes as numeric-cites, tier-cites and any-cites. A new claim type fails
# here rather than inheriting a scope by default, and the agreement test checks
# every (type, id) pair against the grammar itself.
CITE_SCOPE: dict[str, tuple[str, ...]] = {
    "value": NUMERIC_IDS,
    "hedged_value": NUMERIC_IDS,
    "observation": EVIDENCE_ID_VOCAB,
    "review_flag": TIER_IDS,
    "population_association": EVIDENCE_ID_VOCAB,
}
if set(CITE_SCOPE) != set(CLAIM_TYPES):
    raise AssertionError(f"cite scopes {sorted(CITE_SCOPE)} do not match claim "
                         f"types {sorted(CLAIM_TYPES)}")


def _extra_field(name: str) -> dict:
    if name == "value":
        return {"type": "NUMBER"}
    if name == "unit":
        return {"type": "STRING", "enum": list(NUMERIC_UNITS)}
    if name == "text_key":
        return {"type": "STRING", "enum": sorted(TEXT_KEYS)}
    if name == "reason_key":
        return {"type": "STRING", "enum": sorted(REASON_KEYS)}
    raise KeyError(f"no schema for claim field {name!r} - extend _extra_field")


def _branch(name: str) -> dict:
    spec = CLAIM_TYPES[name]
    if spec.subject is None:
        raise AssertionError(f"{name}: no fixed subject to express as an enum")
    cites = {"type": "ARRAY",
             "items": {"type": "STRING", "enum": list(CITE_SCOPE[name])},
             "minItems": spec.cites_min}
    if spec.cites_max is not None:
        cites["maxItems"] = spec.cites_max
    props = {
        "claim_id": {"type": "STRING", "pattern": CLAIM_ID_RE.pattern},
        "claim_type": {"type": "STRING", "enum": [name]},
        "cites": cites,
        "subject": {"type": "STRING", "enum": [spec.subject]},
    }
    for field in spec.extra_fields:
        props[field] = _extra_field(field)
    order = list(BASE_FIELDS) + list(spec.extra_fields)      # the GBNF field order
    return {"type": "OBJECT", "properties": props, "required": order,
            "propertyOrdering": order}


def build_response_schema() -> dict:
    """A top-level array of claims, each matching exactly one per-type branch."""
    return {"type": "ARRAY",
            "items": {"anyOf": [_branch(name) for name in CLAIM_TYPES]}}


def schema_text() -> str:
    return json.dumps(build_response_schema(), indent=2) + "\n"


def write_schema(path: Path = SCHEMA_PATH) -> Path:
    path.write_text(schema_text(), encoding="utf-8", newline="\n")
    return path


# ==========================================================================
# A local validator for the dialect subset used above.
#
# Semantics follow the documented OpenAPI meaning of each keyword. In
# particular objects are OPEN: this dialect has no way to say "no other
# properties", so a property the branch does not declare is not an error here.
# That is the precise sense in which field exclusion is weaker than in GBNF -
# and why Layer 1's unknown-field check, not this schema, is what enforces it.
# ==========================================================================
_PY_TYPES = {"ARRAY": list, "OBJECT": dict, "STRING": str, "BOOLEAN": bool}


def _type_ok(value, t: str) -> bool:
    if t == "NUMBER":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "INTEGER":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, _PY_TYPES[t])


def validate(instance, schema: dict | None = None, path: str = "$") -> list[str]:
    """Every error from checking `instance` against `schema` (default: ours)."""
    if schema is None:
        schema = build_response_schema()
    if "anyOf" in schema:
        if all(validate(instance, branch, path) for branch in schema["anyOf"]):
            return [f"{path}: matches no anyOf branch"]
        return []
    t = schema["type"]
    if not _type_ok(instance, t):
        return [f"{path}: expected {t}, got {type(instance).__name__}"]
    errs: list[str] = []
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(f"{path}: {instance!r} not in enum")
    if "pattern" in schema and not re.search(schema["pattern"], instance):
        errs.append(f"{path}: {instance!r} does not match {schema['pattern']}")
    if t == "ARRAY":
        if len(instance) < schema.get("minItems", 0):
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        for i, item in enumerate(instance):
            errs += validate(item, schema["items"], f"{path}[{i}]")
    if t == "OBJECT":
        for key in schema.get("required", []):
            if key not in instance:
                errs.append(f"{path}: missing {key}")
        for key, sub in schema.get("properties", {}).items():
            if key in instance:
                errs += validate(instance[key], sub, f"{path}.{key}")
    return errs
