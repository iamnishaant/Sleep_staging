"""Launcher for the Somnia console."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description="Somnia · Sleep-Staging console")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args(argv)

    print(f"\n  Somnia console  ->  http://{args.host}:{args.port}\n", flush=True)
    uvicorn.run("ui.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
