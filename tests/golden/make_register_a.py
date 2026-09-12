"""Generate the six register-A golden reports. Never run by the test suite.

The pins in tests/test_render.py compare rendered output byte-for-byte against
these files. Regenerating them is a DELIBERATE act - it is how a wording change
is accepted - so this script refuses to overwrite anything unless called with
--write, and prints a diff of what would change otherwise. A test that could
regenerate its own expectations would pin nothing.

One packet per (split, tier), six in all. The claim set is the oracle witness:
the canonical complete report, which exercises every template path a real
report can reach.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # tests/
sys.path.insert(0, str(HERE.parent.parent))    # repo root

from _packets import packet_with_tier          # noqa: E402
from report import verify_report               # noqa: E402
from report.oracle import oracle               # noqa: E402
from report.render import render_report        # noqa: E402

OUT = HERE / "register_a"
CASES = [(split, tier) for split in ("test", "dev")
         for tier in ("high", "medium", "low")]


def golden_path(split, tier, rec) -> Path:
    return OUT / f"{split}_{tier}_{rec}.txt"


def render_case(split, tier):
    rec, pk = packet_with_tier(tier, split)
    r = verify_report(oracle(pk).witness, pk)
    assert not r.violations, (rec, r.codes)
    return rec, render_report(r.enriched, pk) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually overwrite the golden files")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    changed = 0
    for split, tier in CASES:
        rec, text = render_case(split, tier)
        p = golden_path(split, tier, rec)
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        if old == text:
            print(f"  unchanged  {p.name}")
            continue
        changed += 1
        print(f"  {'WRITE' if a.write else 'would change'}  {p.name}")
        if not a.write:
            sys.stdout.writelines(difflib.unified_diff(
                old.splitlines(True), text.splitlines(True), "golden", "rendered"))
        else:
            p.write_text(text, encoding="utf-8", newline="\n")
    if changed and not a.write:
        print(f"\n{changed} file(s) differ. Re-run with --write to accept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
