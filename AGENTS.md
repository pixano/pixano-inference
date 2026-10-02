# Repository Guidelines

## Project Overview

Pixano-Inference serves AI models for Pixano through Ray Serve, a REST API, and a Python client. The core has no ML framework dependency; model implementations are installed as separate plugin packages.

## Project Structure & Module Organization

The repository contains the core Python distribution, independent packages under `packages/`, and example plugins under `examples/`.

- `src/pixano_inference/api/v1/`: FastAPI routes, request handling, and API schemas.
- `src/pixano_inference/ray/`: Ray Serve application, deployments, configuration, and server lifecycle.
- `src/pixano_inference/models/`: capability base classes and model registry; concrete models live in plugin packages.
- `src/pixano_inference/configs/` and `src/pixano_inference/schemas/`: model configuration and shared data types.
- `src/pixano_inference/plugins.py`: discovery through the `pixano_inference.models` entry-point group.
- `src/pixano_inference/main.py`: Typer CLI entry point.
- `src/pixano_inference/utils/`: media loading, media security, and other shared helpers.
- `packages/pixano-inference-client/`: standalone HTTP client and wire schemas, without server dependencies.
- `packages/pixano-inference-torch/`: optional PyTorch helpers.
- Other `packages/pixano-inference-*/` directories: SAM2, CLIP, Grounding DINO, Transformers VLM, and vLLM implementations.
- `tests/`: core unit tests and Ray Serve integration tests; package tests live alongside their own packages.
- `examples/numpy_detector/` and `examples/yolo/`: reference custom model packages.
- `docs/` and `mkdocs.yml`: MkDocs documentation; `scripts/gen_openapi.py` generates `docs/openapi.json`.
- `Dockerfile`, `docker/`, and `docker-compose.yml`: container setup and example deployment configurations.

Before changing plugin discovery or implementing a model, read the [custom model specification](docs/ray_serve/custom_model_spec.md) and [custom models guide](docs/ray_serve/custom_models.md).

## Tech Stack & Package Boundaries

Python 3.10–3.13, FastAPI, Pydantic v2, Ray Serve, and Typer form the server stack. Use `uv` for dependency management and Hatchling for builds. Documentation uses MkDocs Material.

- Keep the core free of ML frameworks such as PyTorch, TensorFlow, JAX, MLX, Transformers, and vLLM.
- Keep the standalone client usable without the core, Ray, FastAPI, or an ML framework installed.
- Declare model dependencies in the model package's `pyproject.toml`. Model packages must not import other model packages.
- Discover models through entry points and registration decorators. Import frameworks lazily inside model lifecycle methods, as specified in the model contract.
- Use the shared media-loading helpers so model inputs follow the server's URL, local-path, and size policies.

## Build, Test, and Development Commands

Run these commands from the repository root unless stated otherwise. Install the core development environment and start a CPU server with the NumPy example model:

```sh
uv sync
uv run pixano-inference --config docker/models.numpy.py --num-gpus 0
```

The default server address is `http://127.0.0.1:7463`; readiness is exposed at `/v1/ready`. The default development group installs the NumPy example plugin without an ML framework.

Run core unit tests and, when relevant, the separate integration suite:

```sh
uv run pytest --cov=pixano_inference -m "not integration" tests/
uv run pytest -m integration tests/integration/
```

Develop model packages in their own environments. For example:

```sh
uv run --project packages/pixano-inference-sam pytest packages/pixano-inference-sam/tests/
```

Read the affected package's README and `pyproject.toml` for framework, platform, and model-weight requirements. Package CI uses CPU dependencies where possible and mocks vLLM; follow `.github/workflows/test_back.yml` when reproducing those environments.

Build the core distribution and documentation:

```sh
uv build
uv run mkdocs build
```

After changing API routes or wire schemas, regenerate and check the OpenAPI document using the locked root environment:

```sh
uv run python scripts/gen_openapi.py
uv run python scripts/gen_openapi.py --check
```

Follow [RELEASING.md](RELEASING.md) for package versions, dependency compatibility pins, and release steps.

## Coding Style & Naming Conventions

Ruff enforces Python formatting and lint rules: a 119-character line limit, double quotes, and Google-style docstrings. Use `snake_case` for functions and modules and `PascalCase` for classes. Follow the existing typed interfaces and Pydantic schema patterns.

Preserve the CEA-LIST copyright and CeCILL-C license headers in Python, YAML, and Markdown files. `check_license_header.py` defines the required headers and exceptions. Top-level Markdown and YAML and documentation Markdown are formatted with Prettier 2.8.8 in CI.

## Testing Guidelines

Add or update tests for behavior changes in the matching `tests/` directory. Use `test_*.py` names. Keep framework-dependent tests in the affected package and use lightweight fakes for core unit tests.

Tests marked `integration` start a real local Ray Serve runtime. The core integration suite uses CPU-only NumPy models; model-package integration tests may need additional dependencies or downloaded weights. Check the affected tests before running them.

Preserve the framework-free core and lightweight client checks in `tests/test_core_framework_free.py` and `packages/pixano-inference-client/tests/test_import_light.py`. Validate client isolation in an environment containing only the client and its test dependencies, following the standalone-client CI job.

Run checks relevant to the change and report any checks that could not run. Documentation-only changes need formatting and license-header checks, rather than model inference tests.

## Git Workflow

### Committing Changes

- Keep commits focused and use short, imperative subjects, with prefixes such as `fix:`, `feat:`, `docs:`, `refactor:`, `ci:`, or `chore:` where appropriate.
- All commits must include a DCO sign-off via `git commit -s`.
- Do not add `Co-Authored-By` trailers or other co-author lines.
- Run the relevant pre-commit hooks before committing and inspect any changes they make.

```sh
git commit -s -m "docs: add repository agent guidance"
```

### Creating Pull Requests

Follow the [pull request template](.github/PULL_REQUEST_TEMPLATE.md), link the related issue when there is one, and describe the change and validation performed. For multiline GitHub CLI descriptions, write the body to a file and use `gh pr create --body-file path/to/body.md` to preserve Markdown and avoid shell interpolation.

### Checking CI Status

```sh
gh run list --branch "$(git branch --show-current)"
gh run view <run-id>
gh pr checks
```

## Pre-commit Hooks

Pre-commit runs in an isolated tool environment; it is not a project dependency. Install it and the repository hooks with:

```sh
uv tool install pre-commit
uv tool run pre-commit install --install-hooks
```

Run all hooks or limit them to changed files:

```sh
uv tool run pre-commit run --all-files
uv tool run pre-commit run --files path/to/file.py
```

The configured hooks include Ruff linting and formatting, mypy, file checks, and the license-header check. Some hooks rewrite files, so inspect `git status --short` afterward. The license-header hook scans the repository even when specific files are selected.
