# Test-set access log (ADDING EOG VERSION - ST NIGHTS)

Every read of held-out test labels from this folder is recorded here.

## TEST RE-SCORE — 2026-10-11 03:22

Purpose: report the frozen N4kd's test performance against corrected labels (preprocessing alignment bug). No model trained, tuned or selected.

- reproduced reported figures first: {'accuracy': 0.7773324160210583, 'kappa_argmax': 0.7000849649157589, 'kappa_decoded': 0.6992668465066421}
- decoded (as shipped), stored labels: acc 0.7769, kappa 0.6993
- decoded (as shipped), corrected labels: acc 0.7900, kappa 0.7169
- labels changed: 1663; never-scored epochs dropped: 66
