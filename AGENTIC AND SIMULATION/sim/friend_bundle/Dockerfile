# Edge simulation image (based on ../Dockerfile, which stays unchanged).
# The project, models and data are MOUNTED at run time, never baked in.
#
#   target "agent"   : Python + pinned packages + llama.cpp b10927 (L1: full system, x86)
#   target "staging" : Python + pinned packages, no llama.cpp (L2: ARM64 staging parity under emulation,
#                      without compiling llama.cpp under QEMU, which takes hours)
FROM python:3.12-slim-bookworm AS llama
ARG LLAMA_TAG=b10927
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${LLAMA_TAG} https://github.com/ggml-org/llama.cpp /src
# GGML_NATIVE=OFF: don't tune to the build machine's CPU. On arm64: Cortex-A76 (Pi 5) feature set.
RUN ARCH_FLAGS=""; if [ "$(uname -m)" = "aarch64" ]; then ARCH_FLAGS="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"; fi \
    && cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
             -DBUILD_SHARED_LIBS=OFF $ARCH_FLAGS \
    && cmake --build /src/build --target llama-completion -j"$(nproc)"
# Run 1 failed with "libllama-completion-impl.so: cannot open shared object file": build static, and still
# collect any shared library the build produced next to the executable.
RUN mkdir -p /out && cp /src/build/bin/* /out/ && find /src/build -name '*.so*' -exec cp -P {} /out/ \;

FROM python:3.12-slim-bookworm AS staging
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
# CPU-only torch (the default x86 wheel pulls several GB of CUDA libraries the Pi will never have)
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-sim.txt /tmp/requirements-sim.txt
RUN pip install --no-cache-dir -r /tmp/requirements-sim.txt
# Staging-only image: a stub in place of llama-completion, so the release manifest can hash a file.
# The oracle backend used for staging runs never calls it.
RUN mkdir -p /opt/llama && printf '#!/bin/sh\necho "staging-only image: no llama-completion" >&2\nexit 3\n' \
        > /opt/llama/llama-completion && chmod +x /opt/llama/llama-completion
ENV AGENTIC_LLAMA_BIN=/opt/llama/llama-completion PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /repo
ENTRYPOINT ["python"]

FROM staging AS agent
COPY --from=llama /out/ /opt/llama/
ENV LD_LIBRARY_PATH=/opt/llama
# fail the BUILD, not the 31 nights, if the executable cannot start
RUN /opt/llama/llama-completion --version
