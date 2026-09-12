"""Deterministic rendering, in register A (clinician). The model contributes no
prose, only keys.

REGISTER A, decided 12 September 2026 (report/PHASE2_NOTES.md): a reader who can
act on a hypnogram. The review instruction and the tier stay; the tier's
mechanics do not - they remain in the packet, which is where an auditor looks.

PROVENANCE. A rendered report must be derivable from its packet alone - the
renderer-side counterpart of rule 11. Anything that reaches the output comes
from exactly one of three places:

    a packet field            read from the cited evidence item, or from
                              attribution_quality for the footer
    a declared constant       WORDING or a LABEL - never a fact. A constant that
                              asserts a threshold, a statistic or a domain claim
                              is a packet fact in disguise.
    a rule consequence        text whose truth is guaranteed by a verifier rule
                              that already passed - e.g. REVIEW REQUIRED, which
                              exists because rule 10 forces the flag

The renderer opens no file. tests/test_render.py instruments `open` across all
60 packets, and a separate test fails on any digit in an output constant, since
a number the packet did not supply is the commonest disguised fact.

PURE: same enriched claims in, byte-identical prose out. No clock, no locale, no
dict-ordering dependence, no randomness.

The renderer receives claims already ENRICHED by the verifier - the claim plus
the packet fields the verifier looked up. That is what lets a hedged value carry
its measured error and its caveat even though the model never saw either, and it
is why a caveat cannot be softened or dropped: the model was never holding it.

Display conventions, the only conversions anywhere, applied after verification:
a `fraction` is shown as a percentage to one decimal; an error bound is shown on
the scale of the value it measures (see _ERROR_DISPLAY, the one declared
inference). The verifier compares raw packet values with no conversion at all.
"""
from __future__ import annotations

from .claim_schema import REASON_KEYS, TEXT_KEYS

# ---- LABELS: names for things, asserting nothing ---------------------------
STAGE_LABEL = {"stage.W.fraction": "Wake", "stage.N1.fraction": "N1",
               "stage.N2.fraction": "N2", "stage.N3.fraction": "N3",
               "stage.REM.fraction": "REM"}

# attribution_quality.evaluated_on_split -> the word a clinician reads. An
# unlisted split is shown as the packet spells it, never guessed.
SPLIT_WORD = {"val": "validation", "test": "test", "train": "training"}


class UndeclaredUnit(ValueError):
    """An error bound whose unit pairing the renderer has not declared."""


class MissingField(ValueError):
    """The packet lacks a field a template needs. The renderer does not guess."""


def _num(x) -> str:
    """Deterministic number formatting. No locale, no thousands separators."""
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x == int(x):
            return str(int(x))
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def _minutes(x) -> str:
    """WORDING: grammatical number. The packet's unit is the string "minutes";
    "1 minutes" reads as a typo in a clinical report, and pinning it would
    lock the typo in. Singular only for exactly one - and the digit is still
    the packet's, formatted, never a literal."""
    return f"{_num(x)} minute" if x == 1 else f"{_num(x)} minutes"


def _quantity(value, unit) -> str:
    if unit == "fraction":
        return f"{value * 100:.1f}%"
    if unit == "minutes":
        return _minutes(value)
    if unit in ("count", "ratio"):
        # "0 ratio" reads as a typo; the label already names the quantity.
        return _num(value)
    if unit == "tier":
        return str(value)
    return f"{_num(value)} {unit}"


# ---- THE ONE DECLARED INFERENCE --------------------------------------------
# The packet's `error_unit` is "minutes" for minute items and "count_or_ratio"
# for everything else. The renderer shows an error on the scale of the value it
# measures - a fraction's error in percentage points, a per-hour rate's error
# per hour - because a mean absolute error is in the units of its quantity.
# That is read from neither field alone, so it is declared here as a table
# keyed on the (unit, error_unit) PAIR, and any pairing outside the table
# raises rather than being guessed at.
_ERROR_DISPLAY = {
    ("minutes", "minutes"): _minutes,
    ("fraction", "count_or_ratio"): lambda e: f"{e * 100:.1f} percentage points",
    ("per hour", "count_or_ratio"): lambda e: f"{_num(e)} per hour",
    ("count", "count_or_ratio"): _num,
    ("ratio", "count_or_ratio"): _num,
}


def _error_phrase(ev) -> str | None:
    mae = ev.get("mean_abs_error")
    if mae is None:
        return None
    pair = (ev.get("unit"), ev.get("error_unit"))
    if pair not in _ERROR_DISPLAY:
        raise UndeclaredUnit(f"{ev.get('id')}: no declared display for "
                             f"(unit, error_unit) = {pair}")
    phrase = f"mean absolute error {_ERROR_DISPLAY[pair](mae)}"
    # Read, with NO default. An earlier version fell back to "validation" when
    # the field was absent - a default asserting where an error was measured.
    where = ev.get("error_measured_on")
    return f"{phrase} on the {where}" if where else phrase


def _tier_clause(ev) -> str:
    """Stage claims always carry the model's reliability tier (policy rule 6).

    Read from the evidence item's own `model_reliability`, not inferred from
    anything. Rule 6 guarantees it for a verified claim; if it is somehow
    absent, the renderer refuses rather than printing a tier it does not have.
    """
    tier = ev.get("model_reliability")
    if tier is None:
        raise MissingField(f"{ev.get('id')}: no model_reliability")
    return f"{STAGE_LABEL[ev['id']]} is a {tier}-reliability stage for this model"


def _cited(claim, eid) -> dict:
    for ev in claim["_evidence"]:
        if ev.get("id") == eid:
            return ev
    raise MissingField(f"{claim.get('claim_id')}: {eid} not among its evidence")


# ---- one template per claim type -------------------------------------------
def render_value(claim) -> str:
    ev = claim["_evidence"][0]
    return f"{ev['label']}: {_quantity(claim['value'], claim['unit'])}."


def render_hedged_value(claim) -> str:
    """Value, then measured error, then caveat. Never a bare number."""
    ev = claim["_evidence"][0]
    text = f"{ev['label']}: {_quantity(claim['value'], claim['unit'])}"
    err = _error_phrase(ev)
    if err:
        text += f", {err}"
    text += "."
    if ev.get("id") in STAGE_LABEL:
        text += f" {_tier_clause(ev)}."
    caveat = ev.get("caveat")
    if caveat:
        # VERBATIM - the packet's own words, nothing appended or trimmed. An
        # earlier version added a full stop when a caveat lacked one; none of
        # the 60 packets' caveats does, so it never fired, but a caveat the
        # renderer can edit is not the packet's caveat any more.
        text += f" Caveat: {caveat}"
    return text


# Observations DESCRIBE and give no instruction; the banner instructs. Every
# placeholder is filled from the evidence item the key requires (rule 7 has
# already confirmed it is cited and its predicate holds). Register A keeps the
# tier and drops its mechanics, so these state what the packet says and nothing
# about how it was computed.
OBSERVATION_TEMPLATE = {
    "tier_is_high": "{label} is in the {value} tier.",
    "tier_is_medium": "{label} is in the {value} tier.",
    "tier_is_low": "{label} is in the {value} tier.",
    # The packet's own label for this item is the whole statement.
    "n1_reliability_is_low": "{label}.",
}

REVIEW_TEMPLATE = {
    "low_night_confidence": ("This recording falls in the {value} "
                             "night-confidence tier. Review the full hypnogram "
                             "before relying on any figure below."),
    # Was "N1 detection is the weakest part of this model" - a comparison across
    # stages that the cited evidence does not make. Now the evidence's own label.
    "n1_low_reliability": ("{label}. Review N1-scored epochs individually "
                           "before relying on them."),
}


def render_observation(claim) -> str:
    ev = _cited(claim, TEXT_KEYS[claim["text_key"]].requires)
    return OBSERVATION_TEMPLATE[claim["text_key"]].format(label=ev["label"],
                                                          value=ev["value"])


def render_review_flag(claim) -> str:
    ev = _cited(claim, REASON_KEYS[claim["reason_key"]].requires)
    return REVIEW_TEMPLATE[claim["reason_key"]].format(label=ev["label"],
                                                       value=ev["value"])


def render_population_association(claim) -> str:
    """Unreachable today - no packet has associative evidence.

    Was "In population studies, X has been reported as associated with
    sleep-disorder risk": a literature claim and an association target, neither
    of which the packet states. What rule 12 does guarantee is that the cited
    item is associative, so that is all this says.
    """
    labels = ", ".join(e["label"] for e in claim["_evidence"])
    return (f"{labels}: associative evidence. It describes an association "
            f"across a population and is not a statement about this recording.")


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
# `attribution_quality` is not citeable by any claim type, so no claim produces
# this text and no model can phrase it. Every fact in it is read: the gate id,
# the verdict, how many predictions were met OUT OF HOW MANY (was a hardcoded
# "of 5"), whether they were pre-registered (was asserted unconditionally), the
# split and its size (was once "test split of 29" for every packet).
#
# The cohort-scope sentence is emitted only when the packet itself declares
# cohort scope. Two sentences that used to follow - "measured once" and "no
# per-night version was measured" - are gone: the packet states them only in
# free prose, not as fields, so a constant repeating them was asserting facts
# the renderer could not read.
# --------------------------------------------------------------------------
def render_attribution_footer(packet: dict) -> str | None:
    aq = packet.get("attribution_quality") or {}
    if aq.get("status") != "run":
        return None
    gate, verdict = aq.get("gate"), aq.get("verdict")
    n_met, predictions = aq.get("n_met"), aq.get("predictions_met")
    split, n_recs = aq.get("evaluated_on_split"), aq.get("evaluated_on_n_recordings")
    if None in (gate, verdict, n_met, split, n_recs) or not predictions:
        return None
    if not str(aq.get("scope", "")).startswith("COHORT"):
        return None
    registered = "pre-registered " if aq.get("preregistration") else ""
    return (f"Explainability check (gate {gate}): {verdict}, {n_met} of "
            f"{len(predictions)} {registered}predictions met, across {n_recs} "
            f"{SPLIT_WORD.get(split, split)} recordings. It describes the model "
            f"across that cohort, not this recording.")


# --------------------------------------------------------------------------
# The gate.
#
# A report renders only from a CLEAN verification result: exactly
# len(result.violations) == 0. Not schema-valid, not policy-pass, not "some
# claims survived", not a coverage threshold. Failing claims are never filtered
# out and the survivors rendered - mandatory coverage is a property of the SET,
# so a rendered subset is an apparently authoritative report that silently
# omits facts the contract requires. Fail closed.
# --------------------------------------------------------------------------
class RenderRefused(ValueError):
    """render_report was handed a verification result with violations.

    Carries the COMPLETE violation list, in the verifier's order, so a caller
    can report every one of them - not just the first.
    """

    def __init__(self, violations):
        self.violations = list(violations)
        self.codes = [str(v.code) for v in self.violations]
        super().__init__(f"refusing to render: {len(self.violations)} "
                         f"violation(s) {self.codes}")


def render_report(result, packet: dict) -> str:
    """Render a clean verification result, or raise RenderRefused.

    `result` is the VerifyReport that report.verify_report returned. The
    renderer does not verify: it consumes the result already computed, so
    there is one verification path, and what it renders is the result's own
    enriched claims - never a second argument that could disagree with what
    was verified. `packet` is still passed because the result does not carry
    it and the footer reads it.
    """
    from . import VerifyReport       # deferred: report/__init__ imports this module
    if not isinstance(result, VerifyReport):
        # A Layer 2 PolicyResult has `violations` and `enriched` too, but it
        # has not been through Layer 1 - accepting it would open a bypass.
        raise TypeError(f"render_report takes the VerifyReport from "
                        f"report.verify_report, not {type(result).__name__}")
    if len(result.violations) != 0:
        raise RenderRefused(result.violations)
    return _assemble(result.enriched, packet)


def render_unverified(enriched_claims, packet: dict) -> str:
    """ESCAPE HATCH: render with no gate. For tests and debugging only.

    Never call this from report/ - tests/test_render.py fails if any module
    there does. It is not a second renderer: it and render_report share
    _assemble, so a clean set gives identical bytes through either.
    """
    return _assemble(enriched_claims, packet)


def _assemble(enriched_claims, packet: dict) -> str:
    """Assemble the report. Banner first, then claims in the order given.

    REVIEW REQUIRED is a rule consequence: it appears exactly when a review
    flag on night.confidence is present, which rule 10 forces on every
    low-confidence night.
    """
    lines: list[str] = []

    banner = [c for c in enriched_claims
              if c["claim_type"] == "review_flag"
              and "night.confidence" in c.get("cites", [])]
    if banner:
        lines.append("REVIEW REQUIRED")
        for c in banner:
            lines.append(render_claim(c))
        lines.append("")

    banner_ids = {id(c) for c in banner}
    body = [c for c in enriched_claims if id(c) not in banner_ids]
    for c in body:
        lines.append(render_claim(c))

    footer = render_attribution_footer(packet)
    if footer:
        lines.append("")
        lines.append(footer)

    return "\n".join(lines)
