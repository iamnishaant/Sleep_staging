"""Test-only crash injection. `crash_point("deliver:before_marker")` kills the process immediately
(no cleanup, like kill -9) when the environment variable AGENTIC_CRASH_AT names that point.
In normal operation the variable is unset and this does nothing."""
from __future__ import annotations

import os

ENV = "AGENTIC_CRASH_AT"
EXIT_CODE = 137


def crash_point(name: str) -> None:
    if os.environ.get(ENV) == name:
        os._exit(EXIT_CODE)
