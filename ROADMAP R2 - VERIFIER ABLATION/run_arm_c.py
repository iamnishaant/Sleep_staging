"""R2 arm C: Llama-3.2-3B-Student on the 31 dev packets with the grammar removed.

    python "ROADMAP R2 - VERIFIER ABLATION/run_arm_c.py"          the plan; runs nothing
    python "ROADMAP R2 - VERIFIER ABLATION/run_arm_c.py" --go     every uncached packet

Reuses candidates.run.CandidateRun unchanged (its "unconstrained" arm removes exactly
--grammar-file). Cache and log go to this folder so no existing file is written.
Dev packets only: CandidateRun refuses anything else.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from candidates.run import CandidateRun  # noqa: E402

MODEL = "Llama-Student"
CACHE = HERE / "cache"
LOG = HERE / "logs" / "runs.jsonl"


def make_run(**kw) -> CandidateRun:
    return CandidateRun(MODEL, cache_dir=CACHE, log_path=LOG, arm="unconstrained", **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    run = make_run()
    pending = run.pending()
    print(f"{MODEL}, unconstrained, dev: {len(run.jobs) - len(pending)} cached, {len(pending)} to generate")
    if a.go and pending:
        print(run.run(progress=lambda m: print(m, flush=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
