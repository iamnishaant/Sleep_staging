"""The deterministic fallback summary: shown when no verified report can be produced.

It contains NO generated text. Every number is a packet value under a registered derivation
(see vhtml.py), so the same V-HTML check applies to it. Only evidence items the packet marks
safe_to_assert are shown as plain values; the rest are counted, not stated.
"""
from __future__ import annotations

import html

from ..core.outcomes import REASONS
from .vhtml import num_format


def _value(item: dict) -> str:
    v, unit = item.get("value"), item.get("unit")
    if v is None:
        return "not available"
    if unit == "fraction" and isinstance(v, (int, float)):
        return f"{v * 100:.1f}%"
    if unit == "minutes":
        return f"{num_format(v)} minute" if v == 1 else f"{num_format(v)} minutes"
    if unit in ("count", "ratio", "tier") or unit is None:
        return num_format(v)
    return f"{num_format(v)} {unit}"


def render_fallback(packet: dict, reasons: list[str]) -> str:
    e = html.escape
    items = packet.get("evidence_items", [])
    safe = [i for i in items if i.get("safe_to_assert") is True]
    withheld = len(items) - len(safe)
    tier = packet.get("night_confidence", {}).get("tier", "unknown")
    rows = "\n".join(f"<tr><td>{e(str(i.get('label', i.get('id'))))}</td><td>{e(_value(i))}</td></tr>" for i in safe)
    why = "\n".join(f"<li><code>{e(c)}</code> {e(REASONS.get(c, ''))}</li>" for c in reasons)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Automatic sleep summary</title></head>
<body>
<h1>Automatic summary (full report pending review)</h1>
<p>Recording <code>{e(str(packet.get('recording_id', '')))}</code>. Night confidence: <b>{e(str(tier))}</b>.</p>
<p>The written report for this night could not be verified, so it is not shown. This summary lists only
values that the evidence packet marks as safe to state. Nothing here is generated text.</p>
<table><thead><tr><th>Measure</th><th>Value</th></tr></thead><tbody>
{rows}
</tbody></table>
{'<p>Some further measures are withheld because they are not reliable enough to state on their own.</p>' if withheld else ''}
<h2>Why this is a summary</h2>
<ul>
{why}
</ul>
<p>This is an offline sleep-wellness and reporting prototype, not a diagnosis. A reviewer has been notified.</p>
</body></html>
"""
