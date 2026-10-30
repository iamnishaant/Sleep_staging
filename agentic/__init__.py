"""Deterministic agentic layer around the frozen sleep-report pipeline.

The agent decides what to do (check, route, retry, fall back, deliver, audit). It never decides
what is true: stages come from the staging model, numbers from the evidence packet, and every
generated claim must pass the frozen verifier. See agentic/README.md.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CONFIG_DIR = PACKAGE_DIR / "config"
DEFAULT_RUNTIME_DIR = PACKAGE_DIR / "runtime"
