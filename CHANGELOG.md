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

## [0.6.0] - 2026-09-23

Major release. The serving stack is rebuilt on **real Ray Serve** behind a versioned, camelCase
**`/v1`** HTTP API, with a security baseline, a standalone client distribution, single-node
Docker deployment and observability. The core is now the contract only: it ships no model and
depends on no ML framework, not even through an extra. Every model is a self-contained package
with its own `pyproject.toml`, `uv.lock` and tests, plugged in at runtime.

> **Upgrading:** the HTTP API and the Python client are intentionally breaking. Server operators
> install `pixano-inference` plus one extra per model (`[sam]`, `[clip]`, `[grounding-dino]`,
> `[transformers-vlm]`, `[vllm]`); apps that only _call_ a server install the lightweight
> **`pixano-inference-client`** instead.

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
- **Every model is a separate package**, discovered through the `pixano_inference.models` entry
  point: `pixano-inference-sam` (SAM2 image and video), `pixano-inference-clip`,
  `pixano-inference-grounding-dino`, `pixano-inference-transformers-vlm` and
  `pixano-inference-vllm`. The core bundles none of them. Extras map to those packages:
  `pip install pixano-inference[sam]` (was `[sam2]`/`[sam3]`), `[clip]`, `[grounding-dino]` and
  `[transformers-vlm]` (replace `[transformers]`), `[vllm]`; `[torch]` installs the
  `pixano-inference-torch` helpers. The `jax`, `tensorflow`, `mlx` and `ultralytics` extras are
  gone. Model-specific params are imported from their package
  (`from pixano_inference_grounding_dino import GroundingDINOParams`), no longer from
  `pixano_inference.configs`; model class names are unchanged, so configs that reference a
  model by name keep working once its package is installed.
- **Framework-free core.** The core depends on numpy only. The PyTorch helpers
  (`resolve_device`, `resolve_torch_dtype`, `should_compile`, tensor conversions) live in
  `pixano-inference-torch`, and `pixano_inference.utils` no longer exposes the per-framework
  `is_*_installed` / `assert_*_installed` helpers (the generic `is_package_installed` /
  `assert_package_installed` remain).
- **Custom models are installable plugin packages** that follow the
  [custom model specification](docs/ray_serve/custom_model_spec.md). The broken
  `--module-path` flag is removed.
- **Secure defaults.** The server binds `127.0.0.1` by default; exposing it externally requires
  configuring API keys. Request images must be in `media_allowed_image_formats` (default JPEG,
  JPEG2000, PNG, BMP, GIF, TIFF, WEBP; empty accepts anything Pillow supports) and under
  `media_max_image_pixels` (default 100M).

### Added

- **Real Ray Serve backend** — working autoscaling, replicas, `@serve.batch`, health checks, and
  crash-restarts; one Serve app per model; graceful lifecycle (SIGTERM drain); pre-flight
  resource checks; runtime deploy/undeploy admin routes (`POST /v1/models`, so `--config` is
  optional); an async `JobManager` (TTL + bounded) for long-running tracking jobs;
  per-capability inference timeouts (504 on hang).
- **Bounded, interruptible startup.** Ctrl-C works during startup, and every wait on Ray is
  bounded (`serve_start_timeout_s`, `deploy_timeout_s`) with a diagnostic pointing at the raylet
  log instead of hanging forever when a node dies.
- **`--num-gpus` CLI flag**, to pin the GPU count when Ray's accelerator autodetection is wrong.
- **Embedding capability** — one `embedding` capability that maps image **or** text into a shared
  CLIP space, plus the `pixano-inference-clip` package (MobileCLIP2 via open_clip, CPU-friendly).
- **NER capability.**
- **Security baseline** — optional API-key auth (`X-API-Key` / Bearer, constant-time), optional
  CORS, request body-size limits, and an SSRF-guarded media resolver (`MediaPolicy`: http/https
  only, private/loopback/link-local blocks with allowlist, timeouts, streamed size caps,
  redirect re-validation, media-root path containment, image format allowlist and pixel cap).
- **Install from source, publishing optional.** A model package installs from a local
  directory, a (private) git repository, a `#subdirectory=` of a monorepo, or a directory of
  wheels; `uv` follows the package's `[tool.uv.sources]` to the core it depends on.
- **Docker deployment** — multi-stage `Dockerfile` (CUDA torch wheels, non-root, `HF_HOME`
  volume, healthcheck) and `docker-compose.yml`; build args `TORCH_INDEX_URL`, `MODEL_PACKAGES`
  (which model packages to bundle), `EXTRA_PACKAGES` (a private model from a git URL or an index)
  and `INSTALL_EXAMPLE`; the runtime image ships `git` so a derived image can `pip install` a
  private model.
- **Plugin diagnostics.** Discovery logs which model plugins loaded, and an unknown
  `model_class` error lists the registered classes and the plugins that failed to load, with
  the reason.
- **Observability** — `X-Request-ID` propagation middleware, Prometheus HTTP metrics at
  `/v1/metrics`, and request-id-aware logging (`configure_logging`, plain or JSON).
- **Committed OpenAPI schema** (`docs/openapi.json`) with a generator, kept in sync by CI — a
  stable source for generated frontend types.
- **`scripts/load_test.py`** — an async load generator reporting throughput and latency
  percentiles.
- **Custom model specification** (`docs/ray_serve/custom_model_spec.md`), with
  `examples/numpy_detector` as the reference implementation, alongside the guide.
- **New distributions:** `pixano-inference-client`, `pixano-inference-torch`,
  `pixano-inference-sam`, `pixano-inference-clip`, `pixano-inference-grounding-dino`,
  `pixano-inference-transformers-vlm`, `pixano-inference-vllm`.

### Changed

- `torch.compile` on the transformers-based models is gated (explicit flag, else auto = GPU
  only) — it was compiling unconditionally, a slow no-win or outright failure on CPU.
- `/ready` returns 503 unless every configured model is `RUNNING`; `/health` is a cheap liveness
  probe; `/info` reports live cluster/model status.
- Dependencies raised past published advisories (154 of the 158 Dependabot alerts on the
  previous lock): `ray[serve] >= 2.56`, `transformers >= 5.10`, `vllm >= 0.28`, `Pillow >= 12.3`,
  `python-multipart >= 0.0.30`, `pydantic-settings >= 2.14.2`, `requests >= 2.33`,
  `python-dotenv >= 1.2.2`. The server's `numpy < 2` cap is lifted: server and client accept
  `numpy >= 1.26, < 3`.

### Fixed

- Plugin param defaults now resolve when a model is referenced from a config file (previously the
  replica failed at load with `KeyError` on a defaulted param such as the checkpoint path).
- The `/v1` error envelope JSON-encodes validation error context (a raised `ValueError` no longer
  500s the handler).
- `GroundingDINOModel` called `post_process_grounded_object_detection` with `box_threshold`,
  which transformers no longer accepts (every detection raised `TypeError`).
- `TransformersVLMModel` passed the prompt where processors expect images, and its generic
  fallback used `AutoModelForVision2Seq`, removed in transformers 5
  (now `AutoModelForImageTextToText`).
- `VLLMVLMModel` passed a `device` argument that `vllm.LLM` rejects.

### Packaging / CI

- Published wheels are PyPI-clean (no direct/VCS references); the git-only `sam-2` dependency
  lives in the SAM package's dev group and the Docker image only. The core sdist contains the
  core sources only. Model packages pin the core to `>= 0.6.0, < 0.7.0`, and the core pins the
  client to `>= 0.1.0, < 0.2.0`: a package never resolves a core of another contract.
- CI tests the core with no ML framework installed, every model package from its own lock file
  and environment, a Ray Serve integration job, the three install-from-source paths, the
  standalone client, the OpenAPI schema and a Docker build-and-smoke. The release workflow
  publishes all nine distributions and refuses a tag that does not match `__version__`
  (see `RELEASING.md`).

[0.6.0]: https://github.com/pixano/pixano-inference/releases/tag/v0.6.0
