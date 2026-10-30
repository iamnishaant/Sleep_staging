# One file for linux/amd64 (simulation L1) and linux/arm64 (L2 under QEMU, and the real Pi 5).
# The repository, the models and the data are MOUNTED, never baked in (04_ROADMAP.md, Phase 5), so the
# image holds only the interpreter, the pinned Python packages and llama.cpp b10927.
#
#   build : cd "AGENTIC AND SIMULATION/sim" && docker buildx build --platform linux/amd64 -t sleep-agent:x86 --load .
#           (the sim folder is the build context: only requirements-edge.txt is copied in)
#   run   : see BUILD.md
#
# UNTESTED as of 25 Oct 2026: Docker is not installed on the development PC. Build it once on a machine
# that has Docker before relying on it; BUILD.md lists what to check.
FROM python:3.12-slim-bookworm AS llama
ARG LLAMA_TAG=b10927
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${LLAMA_TAG} https://github.com/ggml-org/llama.cpp /src
# GGML_NATIVE=OFF: do not tune to the build machine's CPU (under QEMU that would be the emulated CPU).
# Pi 5 = Cortex-A76 = armv8.2-a with dot-product and fp16.
RUN ARCH_FLAGS=""; if [ "$(uname -m)" = "aarch64" ]; then ARCH_FLAGS="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"; fi \
    && cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF $ARCH_FLAGS \
    && cmake --build /src/build --target llama-completion -j"$(nproc)"

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-edge.txt /tmp/requirements-edge.txt
RUN pip install --no-cache-dir -r /tmp/requirements-edge.txt
COPY --from=llama /src/build/bin/ /opt/llama/
# the release file's executable path is a Windows path; this overrides it (agentic/core/release.py)
ENV AGENTIC_LLAMA_BIN=/opt/llama/llama-completion \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /repo
# the agent is run as a module from the mounted repo: python -m agentic run-edf ...
ENTRYPOINT ["python", "-m", "agentic"]
