"""Layer 1 - structural checks. No packet required.

Closed-world throughout: an unknown claim type, an unknown field, an unknown
evidence id, an unknown enum value are all hard rejects. The point is not to
catch a model behaving badly; it is that a claim outside the schema has no
meaning the renderer could give it, so there is nothing to salvage.

Why the evidence-id check lives HERE and not in Layer 2: `cites: ["attribution"]`
must fail as a name the schema does not have, not as a policy the verifier
enforces. If it reached Layer 2 the safety boundary would have been implemented
as a blocklist. `tests/test_adversarial.py` asserts the Layer 1 code
specifically for exactly that reason.

Layer 1 owns cross-claim state as well - `claim_id` uniqueness - because the
GBNF grammar is context-free and cannot.
"""
from __future__ import annotations

import json

from .claim_schema import (BASE_FIELDS, CLAIM_ID_RE, CLAIM_TYPES,
                           EVIDENCE_ID_VOCAB,
                           FORBIDDEN_FIELDS, REASON_KEYS, SUBJECTS, TEXT_KEYS)
from .violations import V, Violation

_VOCAB = frozenset(EVIDENCE_ID_VOCAB)


def _type_ok(value, expected) -> bool:
    # bool is a subclass of int in Python, and `True == 1`. A claim whose
    # value is `true` must not sail through a numeric check.
    if isinstance(value, bool) and expected is not bool:
        return False
    return isinstance(value, expected)


def parse(raw: str) -> tuple[list | None, list[Violation]]:
    """Parse model output. Returns (claims, violations); claims is None on failure."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, [Violation(V.MALFORMED_JSON, error=type(exc).__name__)]
    if not isinstance(obj, list):
        return None, [Violation(V.NOT_AN_ARRAY, got=type(obj).__name__)]
    return obj, []


def verify_structure(claims: list) -> list[Violation]:
    """Every structural violation in the claim set, not just the first."""
    out: list[Violation] = []
    if not isinstance(claims, list):
        return [Violation(V.NOT_AN_ARRAY, got=type(claims).__name__)]

    seen_ids: set[str] = set()

    for pos, claim in enumerate(claims):
        if not isinstance(claim, dict):
            out.append(Violation(V.NOT_AN_OBJECT, position=pos,
                                 got=type(claim).__name__))
            continue

        cid = claim.get("claim_id")
        cid_str = cid if isinstance(cid, str) else None

        # ---- base fields present and correctly typed --------------------
        for field, ftype in BASE_FIELDS.items():
            if field not in claim:
                out.append(Violation(V.MISSING_BASE_FIELD, cid_str,
                                     field=field, position=pos))
            elif not _type_ok(claim[field], ftype):
                out.append(Violation(V.BAD_FIELD_TYPE, cid_str, field=field,
                                     expected=ftype.__name__,
                                     got=type(claim[field]).__name__))

        # ---- claim_id shape, then uniqueness -----------------------------
        # Shape keeps Layer 1 and the grammar describing the same space;
        # uniqueness is cross-claim state, which a context-free grammar
        # cannot express at all.
        if cid_str is not None and not CLAIM_ID_RE.match(cid_str):
            out.append(Violation(V.BAD_FIELD_TYPE, None, field="claim_id",
                                 expected="c<digits>", got=cid_str))
        if cid_str is not None:
            if cid_str in seen_ids:
                out.append(Violation(V.DUPLICATE_CLAIM_ID, cid_str))
            seen_ids.add(cid_str)

        ctype = claim.get("claim_type")
        spec = CLAIM_TYPES.get(ctype) if isinstance(ctype, str) else None
        if spec is None:
            out.append(Violation(V.UNKNOWN_CLAIM_TYPE, cid_str,
                                 claim_type=ctype,
                                 permitted=sorted(CLAIM_TYPES)))
            continue                    # per-type checks are meaningless now

        # ---- forbidden verifier-derived fields --------------------------
        for field in sorted(set(claim) & FORBIDDEN_FIELDS):
            out.append(Violation(V.FORBIDDEN_DERIVED_FIELD, cid_str, field=field))

        # ---- closed world: no field outside this type's schema ----------
        for field in sorted(set(claim) - spec.all_fields - FORBIDDEN_FIELDS):
            out.append(Violation(V.UNKNOWN_FIELD, cid_str, field=field,
                                 claim_type=spec.name))

        # ---- required per-type fields, correctly typed -------------------
        for field, ftype in spec.extra_fields.items():
            if field not in claim:
                out.append(Violation(V.MISSING_REQUIRED_FIELD, cid_str,
                                     field=field, claim_type=spec.name))
            elif not _type_ok(claim[field], ftype):
                names = (ftype.__name__ if isinstance(ftype, type)
                         else "|".join(t.__name__ for t in ftype))
                out.append(Violation(V.BAD_FIELD_TYPE, cid_str, field=field,
                                     expected=names,
                                     got=type(claim[field]).__name__))

        # ---- subject enum -----------------------------------------------
        subj = claim.get("subject")
        if isinstance(subj, str) and subj not in SUBJECTS:
            out.append(Violation(V.UNKNOWN_SUBJECT, cid_str, subject=subj,
                                 permitted=sorted(SUBJECTS)))

        # ---- cites: arity, then a closed vocabulary ---------------------
        cites = claim.get("cites")
        if isinstance(cites, list):
            n = len(cites)
            if n < spec.cites_min or (spec.cites_max is not None
                                      and n > spec.cites_max):
                out.append(Violation(V.BAD_CITES_ARITY, cid_str,
                                     claim_type=spec.name, got=n,
                                     min=spec.cites_min, max=spec.cites_max))
            for ref in cites:
                if not isinstance(ref, str):
                    out.append(Violation(V.BAD_FIELD_TYPE, cid_str,
                                         field="cites[]", expected="str",
                                         got=type(ref).__name__))
                elif ref not in _VOCAB:
                    # `attribution`, `attribution_quality`, `per_stage`, ...
                    # are not denied here; they are simply not names the
                    # schema has.
                    out.append(Violation(V.UNKNOWN_EVIDENCE_ID, cid_str,
                                         cited=ref))

        # ---- key enums ---------------------------------------------------
        tk = claim.get("text_key")
        if spec.name == "observation" and isinstance(tk, str) and tk not in TEXT_KEYS:
            out.append(Violation(V.UNKNOWN_TEXT_KEY, cid_str, text_key=tk,
                                 permitted=sorted(TEXT_KEYS)))
        rk = claim.get("reason_key")
        if spec.name == "review_flag" and isinstance(rk, str) and rk not in REASON_KEYS:
            out.append(Violation(V.UNKNOWN_REASON_KEY, cid_str, reason_key=rk,
                                 permitted=sorted(REASON_KEYS)))

    return out
