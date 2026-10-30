"""
    cd "ADDING EOG VERSION - ERROR ANALYSIS"
    python -m pytest -q -p no:cacheprovider test_analyse_errors.py
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_errors as A                                     # noqa: E402

HAVE_CACHES = all(p.exists() for p in A.CACHES.values())


def test_boundary_distance_edges_and_interior():
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    #             run W      run N1  run N2 (ends at recording end -> no boundary there)
    assert A.boundary_distance(y).tolist() == [2, 1, 0, 0, 0, 0, 1, 2, 3, 4]


def test_single_run_has_no_boundary():
    assert (A.boundary_distance(np.zeros(5, int)) == A.FAR).all()


def test_run_context():
    y = np.array([0, 1, 1, 0, 4, 4, 4])
    length, prev, nxt, ordinal = A.run_context(y)
    assert length.tolist() == [1, 2, 2, 1, 3, 3, 3]
    assert prev.tolist() == [-1, 0, 0, 1, 0, 0, 0]
    assert nxt.tolist() == [1, 0, 0, 4, -1, -1, -1]
    assert ordinal.tolist() == [0, 0, 0, 1, 0, 0, 0]          # second W run is ordinal 1


def test_calibration_never_moves_argmax():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(5), size=200)
    assert (A.calibrate(P, 1.2847).argmax(1) == P.argmax(1)).all()


def test_min_run_absorbs_isolated_rem_only():
    P = np.full((7, 5), 0.01)
    seq = [2, 2, 2, 4, 2, 2, 1]                                 # one isolated REM, one N1 at the end
    for t, s in enumerate(seq):
        P[t, s] = 0.96
    out = A.min_run_filter(P, 3, ("REM",))
    assert out.tolist() == [2, 2, 2, 2, 2, 2, 1]               # REM absorbed, short N1 untouched


@pytest.mark.skipif(not HAVE_CACHES, reason="validation caches not present")
def test_reproduces_the_shipped_validation_kappa():
    D, recs, T, spec, dec = A.load()
    import json
    test_subjects = set(json.loads((A.DIST / "splits.json").read_text())["splits"]["test"])
    assert len(recs) == 31 and not {r[:5] for r in recs} & test_subjects   # validation only
    res = A.analyse(D, recs, dec)
    assert res["reproduction"]["matches"]
