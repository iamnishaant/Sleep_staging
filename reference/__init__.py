"""The reference-model tier (Phase 2F): generation through a hosted API.

Kept outside report/ on purpose. report/ is the frozen deterministic tier and
gains no network dependency: nothing under report/ imports this package, and
tests/test_reference_schema.py fails if anything does. What passes between the
two is data - the reference model's output is scored by the same frozen
verifier and evaluator as every local candidate.

The reference model is named once, in reference/config.py. Everywhere else it
is "the reference model".
"""
