"""V-HTML: every number on a delivered page must have registered provenance (ACCEPTANCE.md §5).

Allowed sources:
  1. a direct packet field (numeric leaves, and numbers written inside packet text such as caveats);
  2. a direct quality-sidecar field (same rule; Phase 2);
  3. a registered derivation of such a value (policy.yaml vhtml.derivations): fraction -> percent with
     one decimal, fraction error -> percentage points, number formatting (int as is, float to <= 4
     decimals with trailing zeros stripped);
  4. a whitelisted constant passed in by the caller (e.g. the epoch length).

Per-epoch arrays (probabilities, hypnogram) and provenance metadata are NOT sources: they are long
lists of numbers that would make almost any number look "allowed".
"""
from __future__ import annotations

import html as _html
import re

EXCLUDED_KEYS = frozenset({"probabilities", "hypnogram", "provenance"})
_NUM_IN_TEXT = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?")


def num_format(x) -> str:
    """Identical to report/render.py _num (registered derivation `num_format`)."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x == int(x):
            return str(int(x))
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def _forms(v) -> set[str]:
    """All registered textual forms of one numeric value."""
    forms = {num_format(v), str(v)}
    if isinstance(v, float):
        forms.add(f"{v * 100:.1f}")                    # fraction_percent / fraction_error_pp
        forms.add(num_format(round(v * 100, 4)))
    return forms


def _walk(node, out: set[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k not in EXCLUDED_KEYS:
                _walk(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk(v, out)
    elif isinstance(node, bool) or node is None:
        return
    elif isinstance(node, (int, float)):
        out |= _forms(node)
    elif isinstance(node, str):
        for tok in _NUM_IN_TEXT.findall(node):
            out.add(tok)


def allowed_numbers(packet: dict, sidecar: dict | None = None, constants: set[str] | None = None) -> set[str]:
    allowed: set[str] = set()
    _walk(packet, allowed)
    if sidecar:
        _walk(sidecar, allowed)
    return allowed | set(constants or ())


def page_numbers(page_html: str) -> list[str]:
    text = re.sub(r"<(style|script)\b.*?</\1>", " ", page_html, flags=re.S | re.I)
    text = _html.unescape(re.sub(r"<[^>]+>", " ", text))
    return _NUM_IN_TEXT.findall(text)


def check(page_html: str, packet: dict, sidecar: dict | None = None, constants: set[str] | None = None) -> list[str]:
    """Numbers on the page without provenance (empty list = pass)."""
    allowed = allowed_numbers(packet, sidecar, constants)
    return sorted({n for n in page_numbers(page_html) if n not in allowed and n.lstrip("-") not in allowed})
