<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Docker deployment

The repository ships a production-oriented `Dockerfile` and `docker-compose.yml`. The default
image is a GPU-capable server bundling PyTorch and the SAM2, Grounding DINO and Transformers
VLM model packages.

## Prerequisites

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  for GPU access (the image uses CUDA-enabled torch wheels; no CUDA base image is needed).
- For CPU-only use, build the CPU variant (below) — no toolkit required.

## Quick start

```bash
# 1. (Recommended) set an API key so inference/admin routes require auth.
export PIXANO_INFERENCE_API_KEYS="a-long-random-key"

# 2. Edit docker/models.py to declare the models to deploy (defaults to SAM2 image).

# 3. Build and run.
docker compose up --build
```

The compose file:

- binds the published port to **127.0.0.1 by default** (secure default) — to expose the
  server off-host, change the mapping to `7463:7463` **and set `PIXANO_INFERENCE_API_KEYS`**
  (without a key every route is open). Other services on the same compose network reach it as
  `http://pixano-inference:7463` regardless;
- persists the HuggingFace weight cache in the `pixano-weights` volume (`/data/hf`), so
  restarts don't re-download weights;
- mounts `docker/models.py` read-only at `/config/models.py`;
- reserves all GPUs, sets `shm_size: 8gb` (Ray's object store lives in `/dev/shm`), `init: true`
  (reap Ray's child processes), `restart: unless-stopped`, and a `stop_grace_period` matched to
  the bounded graceful drain;
- wires a healthcheck to `/health` with a 600s start period (a cold first boot downloads model
  weights). `/health` is liveness-only; use `/v1/ready` for readiness (503 until every model is
  `RUNNING`).

```bash
curl -fsS http://localhost:7463/v1/ready
```

## Run without compose

GPU image (needs the NVIDIA Container Toolkit; `--shm-size` for Ray, a config, and a weights
volume):

```bash
docker run --rm --gpus all --shm-size=8g -p 127.0.0.1:7463:7463 \
  -e PIXANO_INFERENCE_API_KEYS="a-long-random-key" \
  -v pixano-weights:/data \
  -v "$PWD/docker/models.py:/config/models.py:ro" \
  pixano-inference --host 0.0.0.0 --port 7463 --config /config/models.py
```

CPU-only (build the CPU image above, use a `num_gpus=0` config, no `--gpus`):

```bash
docker run --rm --shm-size=2g -p 127.0.0.1:7463:7463 \
  -v "$PWD/docker/models.numpy.py:/config/models.py:ro" \
  pixano-inference:cpu --host 0.0.0.0 --port 7463 --config /config/models.py
```

The default `docker/models.py` requests a GPU (`num_gpus=1`) and the compose file reserves an
NVIDIA device, so the compose stack requires a GPU host. For CPU, use a config with
`num_gpus=0` (like `docker/models.numpy.py`) and `docker run` without `--gpus`.

## Your own model in the image

A model package does not have to be published: bake it in from wherever it lives.

```bash
# From a git repository (public, or private with credentials in the URL / a private index):
docker build --build-arg EXTRA_PACKAGES="my-model @ git+https://github.com/acme/my-model.git" -t pixano-inference:acme .
```

`EXTRA_PACKAGES` are installed together with the core built from this repository, so a
package that depends on `pixano-inference` resolves it from the build, not from PyPI. Add
`./packages/pixano-inference-torch` to the list if the model uses the torch helpers.

For a private repository over SSH, or a model that lives in a local directory, derive from the
image instead (it ships `git` and `pip`):

```dockerfile
FROM pixano-inference:latest
USER root
# private repository, with the host's SSH agent forwarded by `docker build --ssh default`
RUN --mount=type=ssh pip install "my-model @ git+ssh://git@github.com/acme/my-model.git"
# or a local directory copied into the build context
# COPY my-model /tmp/my-model && pip install /tmp/my-model
USER pixano
```

## Image variants

Build args let you tailor the image:

| Build arg         | Default                                                                                  | Purpose                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu124`                                                 | torch wheels; empty skips the index                                                        |
| `MODEL_PACKAGES`  | `pixano-inference-sam pixano-inference-grounding-dino pixano-inference-transformers-vlm` | model packages to bundle (directories under `packages/`); empty for a framework-free image |
| `EXTRA_PACKAGES`  | _(empty)_                                                                                | extra requirement specs to install, e.g. a private model from a git URL or a private index |
| `INSTALL_EXAMPLE` | `false`                                                                                  | bundle the numpy example plugin                                                            |

```bash
# CPU-only image (torch CPU wheels, no GPU toolkit needed):
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t pixano-inference:cpu .

# Minimal, framework-free image with the numpy example plugin:
docker build \
  --build-arg TORCH_INDEX_URL= \
  --build-arg MODEL_PACKAGES= \
  --build-arg INSTALL_EXAMPLE=true \
  -t pixano-inference:numpy .
docker run --rm -p 7463:7463 --shm-size=2g \
  -v "$PWD/docker/models.numpy.py:/config/models.py:ro" \
  pixano-inference:numpy --host 0.0.0.0 --port 7463 --config /config/models.py
```

## Custom models

Custom models are installed as plugin packages (see the custom-models guide). To bundle your
own model in the image, add an install line for it to the Dockerfile builder stage, e.g.:

```dockerfile
RUN uv pip install my-model         # from PyPI / a private index / a git URL
```

Then reference it by name in your `docker/models.py`.

## Notes

- The container runs as a non-root user (`pixano`, uid 1000). The `pixano-weights` named
  volume inherits that ownership automatically. If you instead bind-mount a **host directory**
  at `/data`, make it writable by uid 1000 (e.g. `chown -R 1000:1000 ./cache`), or the HF cache
  writes fail.
- The `sam-2` library is built from a pinned git commit during the image build (it is not on
  PyPI). The builder has no CUDA toolkit, so `sam-2`'s optional native `_C` kernels are **not**
  compiled — image segmentation works; SAM2's connected-components post-processing uses a slower
  pure-PyTorch fallback. Add a CUDA toolchain to the builder if you need the kernels.
- SIGTERM triggers a bounded graceful drain (`serve.shutdown` + `ray.shutdown`) that completes
  within `stop_grace_period`.
- For fully reproducible builds, pin the `uv` image tag and use the committed lockfile; the
  Dockerfile pins the `sam-2` commit via the `SAM2_REF` build arg.
