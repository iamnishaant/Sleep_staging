# Edge packaging: build and run (roadmap R9, D5)

**25 October 2026.** `Dockerfile` and `requirements-edge.txt` are written. **They have not been built or run**:
Docker is not installed on the development PC. What was checked without it: every pinned package has a
Linux ARM64 wheel for CPython 3.12.

## Build (on a machine with Docker + buildx)

```
cd "AGENTIC AND SIMULATION/sim"
docker buildx build --platform linux/amd64 -t sleep-agent:x86 --load .
docker buildx build --platform linux/arm64 -t sleep-agent:arm --load .
```
The arm64 build under QEMU compiles llama.cpp, which takes a long time. Expect tens of minutes.

## L1: x86 with Pi-like limits, no network

```
docker run --rm --network none --cpus=4 --memory=16g -v "%CD%":/repo:ro -v %TEMP%\agent_rt:/rt ^
    sleep-agent:x86 run-packet /repo/<packet.json> --release v2 --backend llama --runtime /rt
```
(The repo is mounted read-only, so the runtime directory is a separate writable mount.)

## L2: ARM64 emulated: staging parity, then 3 report packets (slow, run overnight)

Same command with `sleep-agent:arm`. Staging probabilities must match the x86 run (max |Δ| ≤ 1e-4,
identical hypnograms); 3 of 3 reports must pass the verifier.

## What to check on the first build

1. `llama-completion --version` inside the image reports build b10927.
2. The model path in the release is relative to the repo, so it resolves under `/repo`; the executable
   path comes from `AGENTIC_LLAMA_BIN`.
3. `--network none` runs succeed (nothing downloads at run time). The in-process network test already
   passes (`agentic/tests/test_privacy.py`); this covers the llama-completion child process too.
4. The report text from the ARM binary will differ from x86 in places. ACCEPTANCE §7's rule applies: parity
   is against a same-environment reference, and every difference must be checked by the verifier.
5. Peak memory (`docker stats`) ≤ 10 GB.

## Known limits

- The image has not been run on a real Pi 5; speed and memory there are still estimates.
- Python 3.12 in the image, 3.14 on the development PC. The adapters and tests ran only on 3.14. Run the
  test suite inside the image once.
