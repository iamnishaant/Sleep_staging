"""The four final outcomes of a night, and the reason codes the agent attaches to its decisions."""
from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    VERIFIED_REPORT = "VERIFIED_REPORT"      # SLM claims passed the verifier AND the page passed V-HTML
    FALLBACK_SUMMARY = "FALLBACK_SUMMARY"    # deterministic summary from packet values + review item
    REVIEW_ONLY = "REVIEW_ONLY"              # nothing safe to show; notice + review item
    REJECTED = "REJECTED"                    # input below the minimum requirements


# Reason codes (policy.yaml uses the same codes). R = input refused, W = warning (report still goes out,
# with a review item), Q = quality, F = report fallback, V = validation, I = integrity.
REASONS = {
    "R01": "recording unreadable",
    "R03": "recording shorter than the minimum duration",
    "R04": "no sustained sleep found in the recording (study period not found)",
    "R05": "recording longer than one night (policy ingest.max_duration_h)",
    "W02": "part of the night has unusable signal (above the warning threshold)",
    "W05": "a channel this release's staging model requires is missing from the recording",
    "W08": "the estimated night boundaries were unstable between the two staging passes",
    "Q01": "too much of the night has unusable signal (above the review threshold)",
    "W10": "line noise near 50 Hz in the EEG (above the warning threshold); staging may be degraded",
    "Q03": "line noise near 50 Hz is too strong to stage the night reliably (above the review threshold)",
    "R10": "packet schema does not match the release",
    "R11": "packet unreadable or not a JSON object",
    "R12": "the packet builder's self-check failed (hypnogram not reproducible from the packet's rounded probabilities)",
    "F01": "the device stayed too hot or short of memory to run the report model (policy resources.*)",
    "F02": "report generation failed after the allowed retries (crash, timeout or no end marker)",
    "F03": "generated claims failed the frozen verifier",
    "F04": "renderer refused the verified claims",
    "V01": "final page failed V-HTML (a number without registered provenance)",
    "V02": "the evidence packet changed after it was loaded or built",
    "I01": "report binary, model or grammar changed since the run started",
}
