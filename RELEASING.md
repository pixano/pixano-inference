<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Releasing

A release publishes nine distributions to PyPI and the versioned documentation, all from the
`Publish` workflow, which runs when a GitHub release is published.

1. Set the core version in `src/pixano_inference/__version__.py`, and bump the version of any
   package under `packages/` whose contents changed (`pixano-inference-client`,
   `pixano-inference-torch`, the model packages). An unchanged package keeps its version and
   its upload is skipped. When the core's minor version changes, update the
   `pixano-inference >= X, < Y` pin in every package under `packages/` and `examples/`.
2. Turn the `[Unreleased]` section of `CHANGELOG.md` into `[<version>] - <date>`.
3. Merge, then publish a GitHub release whose tag is `v<version>`. The workflow refuses a tag
   that does not match `__version__`.

The workflow publishes the client first, then the torch helpers and the model packages, then
the core last, so a run that fails midway never leaves a core on PyPI whose extras cannot
resolve. Re-run the failed jobs from the Actions tab; uploads that already exist are skipped.

Projects can be brought online in stages. List the packages whose PyPI project has no trusted
publisher yet in the repository variable `PYPI_SKIP_PACKAGES` (Settings → Secrets and
variables → Actions → Variables; space-separated distribution names): the release still builds
them and skips their upload. Once a project has its publisher, remove it from the variable and
re-run that package's job on the release run; delete the variable when all are published. Until
a model package is on PyPI, `pip install pixano-inference[<its extra>]` fails on that name.

## PyPI prerequisites

Each project needs a [trusted publisher](https://docs.pypi.org/trusted-publishers/) for the
repository `pixano/pixano-inference` and the workflow `publish.yml` (no environment). For a
project that does not exist on PyPI yet, add it as a _pending_ publisher; the first upload
creates it. PyPI allows one pending publisher per configuration and at most three at a time,
so either bring the projects online one release job at a time (`PYPI_SKIP_PACKAGES`), or create
them all at once with a one-off manual upload using an account-scoped API token and then add
regular publishers to the created projects. The projects: `pixano-inference`, `pixano-inference-client`, `pixano-inference-torch`,
`pixano-inference-sam`, `pixano-inference-clip`, `pixano-inference-grounding-dino`,
`pixano-inference-transformers-vlm`, `pixano-inference-vllm`.
