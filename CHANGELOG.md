<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The core is now **framework-free**: it ships no model implementation and depends on no ML
framework, not even through extras. Every model is a self-contained package with its own
`pyproject.toml`, `uv.lock` and tests.

### ⚠️ Breaking changes

- **Model extras removed.** `pixano-inference[sam|clip|transformers|vllm|torch|jax|tensorflow|mlx|ultralytics]`
  no longer exist. Install the model packages instead: `pixano-inference-sam`,
  `pixano-inference-clip`, `pixano-inference-grounding-dino`,
  `pixano-inference-transformers-vlm`, `pixano-inference-vllm`.
- **Grounding DINO, the Transformers VLM and the vLLM VLM moved out of the core** into
  `pixano-inference-grounding-dino`, `pixano-inference-transformers-vlm` and
  `pixano-inference-vllm`. Their params are imported from those packages
  (`from pixano_inference_grounding_dino import GroundingDINOParams`), no longer from
  `pixano_inference.configs`. Model class names are unchanged, so configs that reference models
  by name keep working once the package is installed.
- **`pixano_inference.frameworks` and `pixano_inference.impls` removed.** The PyTorch helpers
  (`resolve_device`, `resolve_torch_dtype`, `should_compile`, tensor conversions) now live in
  the `pixano-inference-torch` package; the unused JAX/TensorFlow/MLX adapters are gone.
- **`pixano_inference.utils`** no longer exposes the per-framework `is_*_installed` /
  `assert_*_installed` helpers (the generic `is_package_installed` /
  `assert_package_installed` remain), and **`pixano_inference.ray.detect_optional_packages`**
  and `build_runtime_env(auto_detect=...)` are removed.
- **Docker build args.** `PIXANO_EXTRAS` and `INSTALL_SAM` are replaced by `MODEL_PACKAGES`
  (space-separated package directories under `packages/`).

### Fixed

- The client, model and example packages could not be built or installed from their sdist:
  their `pyproject.toml` pointed at a license file outside the package. Each package now
  carries its own `LICENSE`.
- The core sdist bundled the whole repository (model packages, docs, examples); it now
  contains the core sources only.
- The model packages declare every dependency they import (`pydantic`, `numpy`, `torch`)
  instead of relying on what the core or their framework pulls in.
- An unknown `model_class` error now lists the registered model classes and the model
  plugins that were discovered or failed to load (with the reason), and discovery logs that
  summary at startup.
- `GroundingDINOModel` called `post_process_grounded_object_detection` with `box_threshold`,
  which transformers no longer accepts (every detection raised `TypeError`).
- `TransformersVLMModel` passed the prompt where processors expect images, and its generic
  fallback used `AutoModelForVision2Seq`, removed in transformers 5
  (now `AutoModelForImageTextToText`).
- `VLLMVLMModel` passed a `device` argument that `vllm.LLM` rejects.

## [0.6.0] - 2026-07-25

Major production-hardening release. The serving stack is rebuilt on **real Ray Serve** behind a
versioned, camelCase **`/v1`** HTTP API, with a security baseline, a framework-agnostic core,
installable plugin packages, a standalone client distribution, single-node Docker deployment,
and observability.

> **Upgrading:** the HTTP API and the Python client are intentionally breaking. Server operators
> install `pixano-inference` (optionally with `[sam]`, `[clip]`, `[transformers]`); apps that
> only _call_ a server should install the new lightweight **`pixano-inference-client`** instead.

### ⚠️ Breaking changes

- **Versioned `/v1` API.** All inference/admin/service routes live under `/v1`, are typed
  (`response_model`), and are **camelCase on the wire** (snake_case stays in Python). Errors use
  a single envelope `{"error": {code, message, requestId}}`; 500s never leak exception text. The
  old unversioned, snake_case routes are removed.
- **Binary `NDArray` wire format.** Arrays serialize as `{shape, dtype, data}` (base64 raw
  bytes) instead of JSON float lists — orders of magnitude smaller for embeddings/masks.
- **Client rewritten and split out.** `PixanoInferenceClient` / `SyncPixanoInferenceClient`
  (httpx-only: pooled transport, retries/backoff, API key, typed methods, jobs) now ship as a
  **separate `pixano-inference-client` distribution** (deps: httpx/pydantic/numpy only — no Ray,
  FastAPI, or torch). `from pixano_inference.client import PixanoInferenceClient` still works
  when the full package is installed.
- **Custom models are installable plugin packages** discovered via the `pixano_inference.models`
  entry point. The broken `--module-path` flag is removed.
- **SAM2 extracted to a plugin.** `pip install pixano-inference[sam]` (was `[sam2]`/`[sam3]`);
  SAM2 is no longer bundled in core.
- **Framework-agnostic core.** The core depends on numpy only; PyTorch/JAX/TensorFlow/MLX are
  optional peer runtimes.
- **Secure defaults.** The server binds `127.0.0.1` by default; exposing it externally requires
  configuring API keys.

### Added

- **Real Ray Serve backend** — working autoscaling, replicas, `@serve.batch`, health checks, and
  crash-restarts; one Serve app per model; graceful lifecycle (SIGTERM drain); pre-flight
  resource checks; runtime deploy/undeploy admin routes; an async `JobManager` (TTL + bounded)
  for long-running tracking jobs; per-capability inference timeouts (504 on hang).
- **Embedding capability** — one `embedding` capability that maps image **or** text into a shared
  CLIP space, plus the bundled **`pixano-inference-clip`** plugin (MobileCLIP2 via open_clip,
  CPU-friendly).
- **NER capability.**
- **Security baseline** — optional API-key auth (`X-API-Key` / Bearer, constant-time), optional
  CORS, request body-size limits, and an SSRF-guarded media resolver (`MediaPolicy`: http/https
  only, private/loopback/link-local blocks with allowlist, timeouts, streamed size caps,
  redirect re-validation, media-root path containment).
- **Docker deployment** — multi-stage `Dockerfile` (CUDA torch wheels, non-root, `HF_HOME`
  volume, healthcheck) and `docker-compose.yml` (GPU reservation, weights volume); CPU /
  framework-free / GPU build variants; deployment docs.
- **Observability** — `X-Request-ID` propagation middleware, Prometheus HTTP metrics at
  `/v1/metrics`, and request-id-aware logging (`configure_logging`, plain or JSON).
- **Committed OpenAPI schema** (`docs/openapi.json`) with a generator, kept in sync by CI — a
  stable source for generated frontend types.
- **`scripts/load_test.py`** — an async load generator reporting throughput and latency
  percentiles.
- **New distributions:** `pixano-inference-client`, `pixano-inference-sam`,
  `pixano-inference-clip`.

### Changed

- `torch.compile` on the transformers backends is now gated (explicit flag, else auto = GPU
  only) — it was compiling unconditionally, a slow no-win or outright failure on CPU.
- Extras restructured into framework-runtime extras (`torch`, `jax`, `tensorflow`, `mlx`) vs
  model-backend extras (`sam`, `transformers`, `vllm`, `ultralytics`).
- `/ready` returns 503 unless every configured model is `RUNNING`; `/health` is a cheap liveness
  probe; `/info` reports live cluster/model status.

### Fixed

- Plugin param defaults now resolve when a model is referenced from a config file (previously the
  replica failed at load with `KeyError` on a defaulted param such as the checkpoint path).
- The client's numpy range was widened to `>= 1.26, < 3` so numpy-2.x consumers are not forced
  into a downgrade (the server keeps `< 2`).
- The `/v1` error envelope JSON-encodes validation error context (a raised `ValueError` no longer
  500s the handler).

### Packaging / CI

- Published wheels are PyPI-clean (no direct/VCS references); the git-only `sam-2` dependency
  lives in the dev group and Docker image only.
- `ray[serve]` pinned to `>= 2.53, < 3`; license field/classifier and GitHub project URLs added.
- CI overhauled: correct coverage package, a CPU-torch job so model paths are exercised, a
  Ray Serve integration job, a framework-free-core guardrail job, a standalone-client job, and a
  Docker build-and-smoke job. Publishing the standalone client is wired into the release workflow.

[0.6.0]: https://github.com/pixano/pixano-inference/releases/tag/v0.6.0
