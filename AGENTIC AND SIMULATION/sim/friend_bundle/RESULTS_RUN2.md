# Edge simulation, run 2 (friend's Windows PC, Docker Desktop, 2-3 Nov 2026)

Raw results: `AGENTIC AND SIMULATION/results_20261103_0633/` (the zip the script produced; logs and JSON, no run folders).
Run 1's three bundle bugs were fixed first (`RESULTS_RUN1.md`). Container: Linux x86-64, Python 3.12.16, llama.cpp b10927
built from source in the image, cgroup 4 CPUs / 16 GiB, `--network none`, project and data read-only. The numbers below
were recomputed from the per-night rows of each JSON, not only read from `SUMMARY.md`.

## Results

| Step | Result |
|---|---|
| **Environment (P1, P4)** | Network blocked (TCP 1.1.1.1:53 and DNS pypi.org); no `.env` and no key pattern in the mounted project or image; release pins OK; `llama-completion --version` OK (also run at build time) |
| **Staging, x86-64 container vs this PC** (31 dev nights, oracle backend) | 31/31 nights, 61/61 arrays (pass 1 + pass 2) with identical hypnograms; max \|Δp\| 1.8e-5; median 18 s, peak 0.82 GB; 30 verified + 1 review (SC4111E0, Q03), as on this PC |
| **Report model, Path A** (31 dev packets, llama backend, 4 CPUs) | **31/31 VERIFIED_REPORT; report text byte-identical to this PC's on 31/31**; 0 failed runs; median 151 s (p95 160 s, range 146-167 s); peak 5.16 GB |
| **End to end, Path B** (3 raw nights: SC4081, SC4082, SC4111) | Outcome equals this PC's on 3/3 (VERIFIED, VERIFIED, REVIEW_ONLY Q03); staging arrays identical (max 8.3e-7); median 158 s per night (range 28-217 s; 28 s is the review-only night, no report); peak 5.64 GB |
| **ARM64 under QEMU** (8 nights, oracle backend) | **8/8 nights, 15/15 arrays identical hypnograms; max \|Δp\| 7.7e-7** (bar 1e-4); 7 verified + 1 review (SC4111E0); median 133 s per night under emulation (7x the x86 time); peak 1.02 GB |

## Reading these results

- **E1 (ARM64 staging parity) is met under emulation, on a subset.** The 8 nights are the first 8 in alphabetical order,
  so all are Sleep Cassette (SC) recordings; no Sleep Telemetry (ST) night was run on ARM. QEMU runs the real aarch64
  PyTorch binaries but emulates the CPU, so floating-point behaviour on real Cortex-A76 silicon (fused multiply-add)
  could still differ slightly; the 1e-4 bar leaves two orders of magnitude of room. E1 is therefore "met in emulation",
  and real hardware (L5) remains the final check.
- **Report-model parity is exact in the container:** a Linux build of llama.cpp (no AVX tuning flags, `GGML_NATIVE=OFF`)
  reproduced this PC's Windows-built output on all 31 packets. The one night where this PC's own real-model run once
  differed from the 4 Oct cache (SC4582G0, PENDING D1) is not a container effect.
- **Memory:** the whole process tree peaked at 5.16 GB for report generation (analytical estimate 6-8 GB for the Pi, of
  which the model is 4.7 GB), so the 10 GB bar (E4) is met in the container with 4.8 GB to spare; this is x86, not a Pi.
- **Speed:** 151 s per report on the friend's x86 CPU with 4 cores, against 78 s on this PC. The Pi 5's Cortex-A76 cores
  are slower than a desktop x86 core and its memory is narrower; this is **not** a Pi timing and E3 stays open. The
  analytical estimate (266-380 s generation, 6-9 min per night) is unchanged.
- **Container manifest `5489b3f0...`** differs from this PC's `4973f498...` only because the llama.cpp executable is the
  image's own Linux build (the executable is hashed into the manifest; it is not one of the pinned files). Run 1's manifest
  `e41941ec...` was the earlier shared-library build.

## Not shown by run 2

| Item | Status |
|---|---|
| Per-night report text compared on ARM | Not run (ARM image has no llama.cpp; compiling it under QEMU takes hours) |
| ST-cohort nights on ARM | Not run |
| `check` inside the ARM container | Not run: network-off and secrets were checked on x86 only (the ARM step used the same `--network none` limits) |
| Friend's CPU model | Not recorded in the results; ask for it before quoting 151 s against any device |
| E2 (Pi report parity), E3 (time), E5 (thermal), 7-night soak | Need a real Pi 5 |
