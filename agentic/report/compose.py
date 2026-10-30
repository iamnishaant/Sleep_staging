"""Thin wrappers over the frozen report layer (reference/prompts.py, report/). Nothing here changes
their behaviour; it only gives the agent one place to call them from."""
from __future__ import annotations


def build_prompt(packet: dict, prompt_id: str) -> str:
    from reference.prompts import build
    return build(prompt_id, packet)


def verify(raw: str, packet: dict):
    """The frozen two-layer verifier. Returns report.VerifyReport (.ok, .violations, ...)."""
    from report import verify_report
    return verify_report(raw, packet)


def render(result, packet: dict) -> str:
    """The frozen renderer. Raises report.render.RenderRefused if it will not render."""
    from report import render_report
    return render_report(result, packet)


def render_refused_type():
    from report.render import RenderRefused
    return RenderRefused
