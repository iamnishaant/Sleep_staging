"""The local-candidate tier (Phase 2F/2G): small models under llama.cpp, on dev.

Kept outside report/, like reference/ and deploy/. Nothing under report/
imports it. The candidates' outputs are scored by the same frozen verifier
and evaluator as the reference model's.
"""
