"""The llama-completion invocation must be exactly the frozen evaluation's."""
from pathlib import Path

import pytest

from agentic.core.release import Release
from agentic.report.backends import LlamaCompletionBackend, extract_output


def test_command_identical_to_frozen_evaluation():
    try:
        from candidates import run as C            # Windows-only import chain (deploy/measure.py)
    except Exception as e:                          # pragma: no cover
        pytest.skip(f"frozen evaluation module not importable here: {e}")
    rel = Release.load("v1")
    prompt = Path("prompt.txt")
    ours = LlamaCompletionBackend(rel, 10).command(prompt)
    theirs = C.command(C.MODELS["Llama-Student"], prompt, "constrained", None)
    norm = lambda cmd: [str(Path(x)).lower() if ("\\" in x or "/" in x) else x for x in cmd]
    assert norm(ours[1:]) == norm(theirs[1:])
    assert Path(ours[0]).name == Path(theirs[0]).name


def test_extract_output_matches_frozen_rule():
    assert extract_output('[{"a":1}] [end of text]\n', "[end of text]") == ('[{"a":1}]', True)
    assert extract_output("cut off", "[end of text]") == ("cut off", False)
