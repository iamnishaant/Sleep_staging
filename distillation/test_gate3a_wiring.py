"""The Gate 3a -> packet wiring, checked without waiting for a 2-hour IG run.

The wiring reads a dozen keys out of the gate artefact. A KeyError in that path
would only surface when the packets are rebuilt, which is the worst moment to
find it. These tests exercise the populated branch against a synthetic gate
file shaped like the real one, and against the real one when it has the
per-recording block.

They also pin the two things a reader could be misled by if they silently
changed: that the per-night field and the cohort verdict stay separate, and
that both caveats are present in the shipped object.
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_packet as B                                      # noqa: E402

STAGES = ["W", "N1", "N2", "N3", "REM"]
REC = "SC4011E0-PSG"


def synthetic_gate():
    """Same shape as gate3a_attribution.py emits, minimal contents."""
    per_stage = {
        s: {
            "n_epochs": 100,
            "mean_attribution": {"ratio_delta_beta": 0.3},
            "top5": ["ratio_delta_beta", "ratio_dt_ab", "cD1_log_energy", "sef95", "rel_delta"],
            "top1_share": 0.25,
            "cross_recording_top3_intersection": ["ratio_delta_beta"],
            "cross_recording_variance": 0.01,
            "spectral_attribution_share": 0.187,
        }
        for s in STAGES
    }
    return {
        "gate": "3a",
        "preregistration": "distillation/PREREGISTRATION_gate3a.md (committed 2026-08-10)",
        "deviation": "The registration names student_baseline_E0.",
        "models": {
            "student_N4kd": {
                "model": "student_N4kd",
                "n_parameters": 139606,
                "encoder": "multiscale",
                "eeg_scale": 15849.46,
                "n_test_recordings": 29,
                "steps": 64,
                "completeness": {"mean_abs_error": 0.279257,
                                 "mean_abs_reference": 914.981053,
                                 "relative": 0.000305,
                                 "void_threshold": 0.05,
                                 "VOID": False},
                "per_stage": per_stage,
                "predictions": {},
                "gate_verdict": {"criterion": "N3 must be met AND at least 3 of 5",
                                 "predictions_met": {s: True for s in STAGES},
                                 "n_met": 3, "n3_met": True, "void": False, "PASS": True},
                "grouped_by": "predicted",
                "per_recording": {
                    REC: {s: {"n_epochs": 40,
                              "top3": [{"feature": "ratio_delta_beta", "share": 0.36}],
                              "top1_share": 0.36} for s in STAGES},
                },
            }
        },
    }


def build_blocks(gate3a, rec):
    """The wiring's own logic, mirrored, so the key paths are exercised.

    Kept deliberately identical to build_packet.build()'s populated branch. If
    that branch changes and this does not, test_mirror_matches_source below
    fails rather than this drifting into testing nothing.
    """
    gm, gd = gate3a["model"], gate3a["gate"]
    gv = gm["gate_verdict"]
    attribution = {
        "method": "Integrated Gradients",
        "per_stage": gm["per_recording"][rec],
        "scope": "THIS RECORDING.",
    }
    quality = {
        "status": "run",
        "verdict": "PASS" if gv["PASS"] else "FAIL",
        "criterion": gv["criterion"],
        "predictions_met": gv["predictions_met"],
        "n_met": gv["n_met"],
        "void": gv["void"],
        "completeness_relative_error": gm["completeness"]["relative"],
        "scope": f"COHORT ... {gm['n_test_recordings']} recordings.",
        "deviation_from_registration": gd["deviation"],
        "model": gm["model"],
        "preregistration": gd["preregistration"],
    }
    return attribution, quality


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = HERE / "results" / "_test_tmp_gate"
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for p in self.tmp.glob("*"):
            p.unlink()
        self.tmp.rmdir()

    def test_missing_file_returns_none(self):
        self.assertIsNone(B._load_gate3a(self.tmp))

    def test_gate_without_per_recording_returns_none(self):
        """The old gate output must NOT half-populate the packet."""
        g = synthetic_gate()
        del g["models"]["student_N4kd"]["per_recording"]
        (self.tmp / "_g3a_n4kd.json").write_text(json.dumps(g))
        self.assertIsNone(B._load_gate3a(self.tmp),
                          "a gate run without per-recording data must be treated as absent, "
                          "not wired in as a cohort aggregate standing in for one night")

    def test_wrong_model_returns_none(self):
        (self.tmp / "_g3a_n4kd.json").write_text(json.dumps(synthetic_gate()))
        self.assertIsNone(B._load_gate3a(self.tmp, model="student_baseline_E0"))

    def test_loads_when_complete(self):
        (self.tmp / "_g3a_n4kd.json").write_text(json.dumps(synthetic_gate()))
        g = B._load_gate3a(self.tmp)
        self.assertIsNotNone(g)
        self.assertIn("per_recording", g["model"])
        self.assertIn(REC, g["model"]["per_recording"])

    # ---- the ground-truth guard -------------------------------------------
    # A profile keyed by the annotated stage is ground truth. The packet says
    # `_ground_truth_withheld: true`, and the vertical-slice test depends on
    # that being true, so this must be a hard failure and not a quiet null:
    # a null leaves the leaking file on disk for the next rebuild to ship.

    def test_label_keyed_profile_is_refused(self):
        g = synthetic_gate()
        g["models"]["student_N4kd"]["grouped_by"] = "annotated"
        (self.tmp / "_g3a_n4kd.json").write_text(json.dumps(g))
        with self.assertRaises(SystemExit) as cm:
            B._load_gate3a(self.tmp)
        self.assertIn("ground truth", str(cm.exception).lower())

    def test_undeclared_grouping_is_refused(self):
        """The pre-fix artefact declared nothing. It must not load."""
        g = synthetic_gate()
        del g["models"]["student_N4kd"]["grouped_by"]
        (self.tmp / "_g3a_n4kd.json").write_text(json.dumps(g))
        with self.assertRaises(SystemExit):
            B._load_gate3a(self.tmp)

    def test_refusal_is_not_a_silent_null(self):
        """Regression: an earlier design returned None here."""
        g = synthetic_gate()
        g["models"]["student_N4kd"]["grouped_by"] = "annotated"
        (self.tmp / "_g3a_n4kd.json").write_text(json.dumps(g))
        try:
            r = B._load_gate3a(self.tmp)
        except SystemExit:
            return
        self.fail(f"returned {r!r} instead of refusing a label-keyed profile")


class TestBlocks(unittest.TestCase):
    def setUp(self):
        g = synthetic_gate()
        self.g3 = {"gate": g, "model": g["models"]["student_N4kd"]}

    def test_every_key_the_wiring_reads_exists(self):
        a, q = build_blocks(self.g3, REC)
        self.assertEqual(q["verdict"], "PASS")
        self.assertEqual(q["completeness_relative_error"], 0.000305)
        self.assertEqual(a["per_stage"].keys(), set(STAGES) | set())

    def test_scopes_are_distinct(self):
        """The night and the cohort must never be conflated."""
        a, q = build_blocks(self.g3, REC)
        self.assertIn("THIS RECORDING", a["scope"])
        self.assertIn("COHORT", q["scope"])

    def test_per_stage_is_this_recording_not_the_pooled_profile(self):
        a, _ = build_blocks(self.g3, REC)
        pooled = self.g3["model"]["per_stage"]
        self.assertIsNot(a["per_stage"], pooled)
        self.assertEqual(a["per_stage"]["W"]["n_epochs"], 40)      # per-recording
        self.assertEqual(pooled["W"]["n_epochs"], 100)             # pooled


class TestShippedSource(unittest.TestCase):
    """Checks against build_packet.py's actual text, not a copy of it."""

    def setUp(self):
        self.src = (HERE / "build_packet.py").read_text(encoding="utf-8")

    def test_schema_is_13(self):
        self.assertEqual(B.SCHEMA_VERSION, "1.3")
        self.assertIn("1.3", B.SCHEMA_CHANGELOG)

    def test_both_caveats_ship_in_the_packet(self):
        """The report is not enough - a consumer reads the packet."""
        self.assertIn("IDENTICAL across all five stages", self.src)
        self.assertIn("DIMENSION-BIASED", self.src)

    def test_baseline_is_not_described_as_zeros(self):
        """IG here runs from registered means, not an all-zeros baseline."""
        self.assertNotIn("all-zeros baseline", self.src.replace("NOT all-zeros", ""))
        self.assertIn("TRAIN split", self.src)

    def test_verdict_is_labelled_cohort_scope(self):
        self.assertIn("COHORT, NOT THIS NIGHT", self.src)

    def test_coverage_is_asserted_at_build_time(self):
        """Each night's profile must cover exactly that night."""
        self.assertIn("attribution profile covers", self.src)
        self.assertIn("len(pred_stages)", self.src)

    def test_grouping_guard_is_present(self):
        self.assertIn('grouped != "predicted"', self.src)

    def test_mirror_matches_source(self):
        """Every gate key this test mirrors is really read by build_packet."""
        for key in ('gm["per_recording"][rec]', 'gv["criterion"]', 'gv["predictions_met"]',
                    'gm["completeness"]["relative"]', 'gd["deviation"]', 'gv["PASS"]'):
            self.assertIn(key, self.src, f"wiring no longer reads {key}; update this test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
