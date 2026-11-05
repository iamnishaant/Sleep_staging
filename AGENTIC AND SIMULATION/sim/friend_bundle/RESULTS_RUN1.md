# Edge simulation, run 1 (friend's Windows PC, Docker Desktop, 2 Nov 2026)

Raw results: `D:\sleep project data _simulation\dataset_archive\out` (copied by hand; the zip step failed).
Container: x86_64, Python 3.12.16, cgroup 4 CPUs (`400000 100000`) / 16 GiB, `--network none`, bundle read-only.

## Valid results

| Check | Result | Evidence |
|---|---|---|
| **P1 network off** | **Met (L1 container).** TCP 1.1.1.1:53 and DNS pypi.org both blocked | `check_x86_64.json` |
| **P4 no secrets** | **Met.** No `.env`, no Gemini key pattern in /repo, /sim, /opt, /root | `check_x86_64.json` |
| Release pins | All pinned hashes match (manifest `e41941ec…`; differs from `4973f498…` only because the llama executable is the container's own build, which is hashed but not pinned) | `check_x86_64.json` |
| **Staging parity, Linux x86 vs this PC (Windows)** | **31/31 nights, 61/61 arrays (pass 1 + pass 2) with identical hypnograms; max \|Δprob\| 1.8e-5 (limit 1e-4)** | recomputed from `rt_pathb_oracle_x86_64/runs/*`, script below |
| Same, llama run | 3 nights, 5/5 arrays identical, max 8.3e-7; SC4111 → REVIEW_ONLY (Q03), same as this PC | `rt_pathb_llama_x86_64` |
| Staging memory | peak 0.8 GB per night (whole process tree) | `pathb_oracle_x86_64.json` |

E1 itself is **ARM64** parity; the ARM step was not run. The x86 Linux result shows staging does not drift
across OS / compiler / torch build, which is the precondition for E1, not E1.

The summary's "30 failed / E1_pass false" is an artefact: nights crashed after staging (bug 2) so `measure.py`
found no run id and compared nothing; the run folders hold the complete staging outputs.

## Invalid (bundle bugs, fixed for run 2)

| Bug | Effect | Fix |
|---|---|---|
| 1. llama.cpp built as shared libs; only `build/bin` copied | `libllama-completion-impl.so` missing → every report F02 → 31/31 Path A FALLBACK, 0 text matches, timings meaningless (12 s) | `Dockerfile`: `-DBUILD_SHARED_LIBS=OFF`, copy any `.so`, `LD_LIBRARY_PATH`, `--version` at build time |
| 2. `deploy/measure.py` not bundled (`student/build_set.py` imports it) | oracle nights crashed at the report step (`ModuleNotFoundError: deploy`) | `make_bundle.py` copies `deploy/*.py`; full import sweep run on the bundle |
| 3. `Compress-Archive` on the open transcript | no zip | `run_all.ps1` stops the transcript while zipping |

Not yet measured: report-model speed and memory under 4 cores / 16 GB, report text identity, end-to-end outcome
match, ARM64 parity (E1). Needs run 2.

## Parity recomputation

```python
for run in glob(f"{OUT}/rt_pathb_oracle_x86_64/runs/*"):
    rec = json.load(open(f"{run}/input/meta.json"))["recording_id"]
    for k in ("pass1", "pass2"):
        a, b = np.load(glob(f"{run}/*_stage_{k}/{k}.npy")[0]), np.load(f"{REF}/{k}_reference/{rec}.npy")
        # max |a-b|, (a.argmax(-1) == b.argmax(-1)).all()
```
