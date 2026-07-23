# syntax=docker/dockerfile:1

# Multi-stage build for Pixano Inference.
#
# The default image is a GPU-capable server bundling the PyTorch runtime plus the built-in
# torch backends (transformers) and the SAM2 plugin. GPU access at runtime is provided by
# the host driver via the NVIDIA Container Toolkit, so no CUDA base image is needed — the
# CUDA-enabled torch wheels are self-contained.
#
# Build args let you produce variants:
#   TORCH_INDEX_URL   torch wheel index; set empty to skip torch (CPU/framework-free image),
#                     or https://download.pytorch.org/whl/cpu for a CPU torch build.
#   PIXANO_EXTRAS     core extras to install (e.g. "transformers"); empty for none.
#   INSTALL_SAM       "true" to bundle the SAM2 plugin (+ the git-only sam-2 library).
#   INSTALL_EXAMPLE   "true" to bundle the framework-free numpy example plugin.
#
# Examples:
#   docker build -t pixano-inference .                                   # GPU (default)
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t pixano-inference:cpu .

ARG PYTHON_VERSION=3.11

# --- Builder ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG PIXANO_EXTRAS=transformers
ARG INSTALL_SAM=true
ARG INSTALL_EXAMPLE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv build tool. Pin to a released tag/digest for fully reproducible production builds.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY packages ./packages
COPY examples ./examples

# Pin the sam-2 source to a known-good commit for reproducibility.
ARG SAM2_REF=2b90b9f5ceec907a1c18123530e92e794ad901a4

RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    # Install torch AND torchvision from the same index so the CPU/cu124 selection is honoured
    # for both and their CUDA builds stay ABI-coherent (torchvision is required by
    # transformers and sam-2).
    if [ -n "${TORCH_INDEX_URL}" ]; then uv pip install torch torchvision --index-url "${TORCH_INDEX_URL}"; fi; \
    # The lightweight client is a hard core dependency; install it from the bundled source first
    # (it is not on PyPI at build time) so the core install below resolves it locally.
    uv pip install ./packages/pixano-inference-client; \
    if [ -n "${PIXANO_EXTRAS}" ]; then uv pip install ".[${PIXANO_EXTRAS}]"; else uv pip install .; fi; \
    if [ "${INSTALL_SAM}" = "true" ]; then \
        uv pip install ./packages/pixano-inference-sam "sam-2 @ git+https://github.com/facebookresearch/sam2.git@${SAM2_REF}"; \
    fi; \
    if [ "${INSTALL_EXAMPLE}" = "true" ]; then uv pip install ./examples/numpy_detector; fi

# --- Runtime ----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# curl for the healthcheck; libgl/libglib for image/vision libraries; g++ so torch.compile
# (TorchInductor) can build its host glue at inference time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 gcc g++ \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 pixano \
    && mkdir -p /data/hf \
    && chown -R pixano:pixano /data

COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1

VOLUME /data
EXPOSE 7463
USER pixano

# start-period covers a cold first boot (HF weight download + model load can take minutes;
# the app allows up to 600s per model deploy).
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:7463/health || exit 1

ENTRYPOINT ["pixano-inference"]
CMD ["--host", "0.0.0.0", "--port", "7463"]
