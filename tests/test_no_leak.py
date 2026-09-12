"""Ground truth must not escape - through a file, or through a message.

Two channels, both real.

1. FILES. The verifier reads the packet it is handed and nothing else. It must
   never open a label file, the test split, or any ground-truth artefact. Here
   `open` is instrumented and asserted to stay shut for the whole run, rather
   than checking a path allowlist - "opened nothing" is a stronger and simpler
   invariant than "opened only approved things".

2. MESSAGES. In a later phase, violation messages are fed back to the model for
   repair. That makes every message an output channel out of the verifier, and
   a message quoting anything outside the packet would breach
   `_ground_truth_withheld` through the back door. Every value in every
   violation's structured `detail` must be traceable to the packet, to the
   claim the model already wrote, or to a closed schema constant.

The packets themselves assert `_ground_truth_withheld: true`, and that assertion
was false once already in this project - a per-recording attribution profile
keyed by the annotated stage shipped in 29 packets before an arithmetic check
caught it. These tests exist because the invariant has been broken before.
"""
from __future__ import annotations

import builtins
import io
import json
import unittest

from _packets import all_packets, claim, packet_with_tier, valid_claim_set
from report import verify_report
from report import claim_schema as S
from report.render import render_report
from test_adversarial import good_value_claim

HIGH_NAME, HIGH = packet_with_tier("high")
LOW_NAME, LOW = packet_with_tier("low")

# Everything a message is allowed to say that is not packet content.
SCHEMA_LITERALS = (
    set(S.CLAIM_TYPES) | set(S.SUBJECTS) | set(S.TEXT_KEYS) | set(S.REASON_KEYS)
    | set(S.EVIDENCE_ID_VOCAB) | set(S.FORBIDDEN_FIELDS)
    | {"claim_id", "claim_type", "cites", "subject", "value", "unit",
       "text_key", "reason_key", "c<digits>", "str", "int", "float", "bool",
       "list", "dict", "NoneType", "JSONDecodeError", "TypeError",
       "safe_to_assert", "tier", "low", "high", "medium",
       "night_confidence.tier == 'low'",
       "the model.n1_reliability_warning evidence item's value == 'low'"}
)


class TestOpensNothing(unittest.TestCase):
    def test_verifier_and_renderer_open_no_files(self):
        # Load every packet BEFORE instrumenting: the test's own reads are not
        # the verifier's, and counting them would make the assertion vacuous
        # (it would have to allow packet paths, which is what we are testing).
        loaded = list(all_packets())
        cases = [(name, pk, valid_claim_set(pk), json.dumps(
            valid_claim_set(pk))) for name, pk in loaded]

        opened = []
        real_open, real_io = builtins.open, io.open

        def spy(file, *a, **k):
            opened.append(str(file))
            return real_open(file, *a, **k)

        def spy_io(file, *a, **k):
            opened.append(str(file))
            return real_io(file, *a, **k)

        builtins.open, io.open = spy, spy_io
        try:
            for name, pk, cs, raw in cases:
                r = verify_report(raw, pk)
                render_report(r, pk)                           # clean: it renders
                verify_report([good_value_claim(pk)], pk)      # a failing set too
        finally:
            builtins.open, io.open = real_open, real_io

        self.assertEqual(opened, [],
                         f"the verifier opened files: {opened[:5]}")


class TestMessagesArePacketDerived(unittest.TestCase):
    @staticmethod
    def _corpus(packet):
        """Claims that trip as many rules as possible."""
        return [
            claim(claim_id="c1", claim_type="value", cites=["attribution"],
                  value=1.0, unit="minutes"),
            claim(claim_id="c2", claim_type="prognosis", cites=["arch.waso"]),
            claim(claim_id="c3", claim_type="value", cites=["arch.waso"],
                  value=999.5, unit="hours"),
            claim(claim_id="c4", claim_type="hedged_value",
                  cites=["arch.total_sleep_time"], value=1.0, unit="minutes"),
            claim(claim_id="c5", claim_type="observation",
                  cites=["arch.rem_latency", "arch.rem_latency_sustained"],
                  text_key="tier_is_low"),
            claim(claim_id="c6", claim_type="population_association",
                  cites=["arch.waso"]),
            claim(claim_id="c7", claim_type="value", cites=["night.confidence"],
                  value=2.0, unit="tier"),
            claim(claim_id="c8", claim_type="value",
                  cites=["arch.waso"], value=1.0, unit="minutes",
                  safe_to_assert=True),
        ]

    def test_every_detail_value_is_traceable(self):
        for name, pk in all_packets():
            packet_text = json.dumps(pk)
            corpus = self._corpus(pk)
            claim_text = json.dumps(corpus)
            r = verify_report(corpus, pk)
            self.assertTrue(r.violations, name)
            for v in r.violations:
                for key, value in v.detail.items():
                    for token in (value if isinstance(value, list) else [value]):
                        self.assertTrue(
                            self._traceable(token, packet_text, claim_text),
                            f"{name}: {v.code} detail {key}={token!r} is not "
                            f"traceable to the packet, the claim, or a schema "
                            f"constant")

    @staticmethod
    def _traceable(token, packet_text, claim_text) -> bool:
        if token is None or isinstance(token, bool):
            return True
        if isinstance(token, (int, float)):
            # A number is traceable if it came from the packet or the claim.
            return (json.dumps(token) in packet_text
                    or json.dumps(token) in claim_text
                    or str(token) in packet_text or str(token) in claim_text)
        s = str(token)
        return (s in SCHEMA_LITERALS or s in packet_text or s in claim_text)

    def test_messages_render_without_crashing(self):
        r = verify_report(self._corpus(LOW), LOW)
        for v in r.violations:
            self.assertIsInstance(v.message, str)
            self.assertIn(str(v.code), v.message)

    def test_no_message_mentions_a_true_stage(self):
        """A true-stage count anywhere is a leak, not a mistake."""
        for name, pk in all_packets():
            r = verify_report(self._corpus(pk), pk)
            blob = " ".join(v.message for v in r.violations).lower()
            for word in ("ground_truth", "annotated", "expert", "label",
                         "true_stage", "hypnogram_true"):
                self.assertNotIn(word, blob, f"{name}: {word}")


class TestPacketStillWithholdsGroundTruth(unittest.TestCase):
    def test_flag_is_set_on_every_packet(self):
        for name, pk in all_packets():
            self.assertTrue(pk["_ground_truth_withheld"], name)

    def test_no_packet_carries_a_true_stage_sequence(self):
        for name, pk in all_packets():
            for banned in ("true_stages", "stage_sequence", "labels",
                           "annotated_stages", "y_true"):
                self.assertNotIn(banned, json.dumps(pk), f"{name}: {banned}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
