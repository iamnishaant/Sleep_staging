"""The one place the reference model is named. Change it here and nowhere else."""

REFERENCE_MODEL = "gemini-3.8-flash"

API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Free-tier limits for this project, read from AI Studio on 12 September 2026
# (PHASE2_NOTES, "Reference model for 2F - setup").
RATE_LIMIT_RPM = 5
RATE_LIMIT_TPM = 250_000
RATE_LIMIT_RPD = 20
