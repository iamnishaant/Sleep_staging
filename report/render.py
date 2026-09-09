"""Deterministic rendering. The model contributes no prose, only a key.

PURE: same enriched claims in, byte-identical prose out. No clock, no locale, no
dict-ordering dependence, no randomness. `tests/test_render.py` renders twice
and compares bytes.

The renderer receives claims already ENRICHED by the verifier - the claim plus
the packet fields the verifier looked up. That is what lets a hedged value carry
its measured error and its caveat even though the model never saw either, and it
is why a caveat cannot be softened or dropped: the model was never holding it.

One display convention, and it is the only place any conversion happens: items
whose packet unit is `fraction` are shown as percentages, because a clinical
reader expects "92.7%" and not "0.9273". The verifier compares raw packet units
with no conversion whatsoever (rule 3); this is presentation, applied after
verification, and pinned by a byte-exact test so it cannot drift.
"""
from __future__ import annotations

STAGE_LABEL = {"stage.W.fraction": "Wake", "stage.N1.fraction": "N1",
               "stage.N2.fraction": "N2", "stage.N3.fraction": "N3",
               "stage.REM.fraction": "REM"}

TIER_WORD = {"high": "high", "medium": "medium", "low": "low"}


def _num(x) -> str:
    """Deterministic number formatting. No locale, no thousands separators."""
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x == int(x):
            return str(int(x))
        s = f"{x:.4f}".rstrip("0").rstrip(".")
        return s
    return str(x)


def _quantity(value, unit) -> str:
    if unit == "fraction":
        return f"{value * 100:.1f}%"
    if unit in ("count", "ratio"):
        # "0 ratio" reads as a typo; the label already says it is a ratio.
        return _num(value)
    if unit == "tier":
        return str(value)
    return f"{_num(value)} {unit}"


def _error_phrase(ev) -> str | None:
    mae, unit = ev.get("mean_abs_error"), ev.get("unit")
    if mae is None:
        return None
    where = ev.get("error_measured_on") or "validation"
    if unit == "fraction":
        return f"mean absolute error {mae * 100:.1f} percentage points on the {where}"
    if unit in ("minutes", "per hour"):
        return f"mean absolute error {_num(mae)} {unit} on the {where}"
    return f"mean absolute error {_num(mae)} on the {where}"


def _tier_clause(ev) -> str:
    """Stage claims always carry the model's reliability tier (policy rule 6)."""
    tier = ev.get("model_reliability")
    stage = STAGE_LABEL.get(ev.get("id"), ev.get("label"))
    return f"{stage} is a {TIER_WORD.get(tier, tier)}-reliability stage for this model"


# --------------------------------------------------------------------------
# one template per claim type
# --------------------------------------------------------------------------
def render_value(claim) -> str:
    ev = claim["_evidence"][0]
    return f"{ev['label']} was estimated at {_quantity(claim['value'], claim['unit'])}."


def render_hedged_value(claim) -> str:
    """Value, then measured error, then caveat. Never a bare number."""
    ev = claim["_evidence"][0]
    parts = [f"{ev['label']} was estimated at "
             f"{_quantity(claim['value'], claim['unit'])}"]
    err = _error_phrase(ev)
    if err:
        parts.append(f", with {err}")
    parts.append(".")
    text = "".join(parts)
    if ev.get("id") in STAGE_LABEL:
        text += f" {_tier_clause(ev)}."
    caveat = ev.get("caveat")
    if caveat:
        # Caveats in the packet are grammatically heterogeneous: some are verb
        # phrases ("depends on a single epoch"), some noun phrases ("mean
        # relative error 85% on validation."), one is a full sentence. An
        # earlier template prefixed "This figure ", which produced "This figure
        # mean relative error 85% on validation." A labelled clause is the only
        # form that composes correctly with all three, and it also makes the
        # caveat visually unmissable, which is the point of carrying it.
        text += f" Caveat: {caveat}"
        if not text.endswith("."):
            text += "."
    return text


OBSERVATION_TEXT = {
    "tier_is_low": ("Overall prediction confidence for this night is in the "
                    "low tier, so the whole recording warrants review."),
    "n1_reliability_is_low": ("N1 is a low-reliability stage for this model; "
                              "N1 figures in this report should be read with "
                              "that in mind."),
}

REVIEW_TEXT = {
    "low_night_confidence": ("This recording falls in the low night-confidence "
                             "tier. Review the full hypnogram before relying "
                             "on any figure below."),
    "n1_low_reliability": ("N1 detection is the weakest part of this model. "
                           "Review N1-scored epochs individually."),
}


def render_observation(claim) -> str:
    return OBSERVATION_TEXT[claim["text_key"]]


def render_review_flag(claim) -> str:
    return REVIEW_TEXT[claim["reason_key"]]


def render_population_association(claim) -> str:
    labels = ", ".join(e["label"] for e in claim["_evidence"])
    return (f"In population studies, {labels} has been reported as associated "
            f"with sleep-disorder risk. This is an association across groups "
            f"and is not a statement about this individual.")


RENDERERS = {
    "value": render_value,
    "hedged_value": render_hedged_value,
    "observation": render_observation,
    "review_flag": render_review_flag,
    "population_association": render_population_association,
}


def render_claim(claim) -> str:
    return RENDERERS[claim["claim_type"]](claim)


# --------------------------------------------------------------------------
# The attribution footer.
#
# `attribution_quality` is NOT citeable by any claim type in Phase 1, so no
# claim can produce this text and no model can phrase it. It is emitted
# straight from the packet when present, and it states the cohort scope
# explicitly, because a reader who sees "PASS" next to one night's report will
# otherwise take it as a statement about that night.
# --------------------------------------------------------------------------
def render_attribution_footer(packet: dict) -> str | None:
    aq = packet.get("attribution_quality") or {}
    if aq.get("status") != "run":
        return None
    return ("Explainability gate 3a: "
            f"{aq.get('verdict')} ({aq.get('n_met')} of 5 pre-registered "
            "predictions met). This verdict was measured once over the pooled "
            "test split of 29 recordings and describes the COHORT, not this "
            "recording. No per-night version of it was measured.")


def render_report(enriched_claims, packet: dict) -> str:
    """Assemble the report. Banner first, then claims in the order given."""
    lines: list[str] = []

    banner = [c for c in enriched_claims
              if c["claim_type"] == "review_flag"
              and "night.confidence" in c.get("cites", [])]
    if banner:
        lines.append("REVIEW REQUIRED")
        for c in banner:
            lines.append(render_claim(c))
        lines.append("")

    body = [c for c in enriched_claims if c not in banner]
    for c in body:
        lines.append(render_claim(c))

    footer = render_attribution_footer(packet)
    if footer:
        lines.append("")
        lines.append(footer)

    return "\n".join(lines)
