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
   its upload is skipped.
2. Turn the `[Unreleased]` section of `CHANGELOG.md` into `[<version>] - <date>`.
3. Merge, then publish a GitHub release whose tag is `v<version>`. The workflow refuses a tag
   that does not match `__version__`.

The workflow publishes the client first, then the torch helpers and the model packages, then
the core last, so a run that fails midway never leaves a core on PyPI whose extras cannot
resolve. Re-run the failed jobs from the Actions tab; uploads that already exist are skipped.

## PyPI prerequisites

Each project needs a [trusted publisher](https://docs.pypi.org/trusted-publishers/) for the
repository `pixano/pixano-inference` and the workflow `publish.yml` (no environment). For a
project that does not exist on PyPI yet, add it as a _pending_ publisher; the first upload
creates it. The projects: `pixano-inference`, `pixano-inference-client`, `pixano-inference-torch`,
`pixano-inference-sam`, `pixano-inference-clip`, `pixano-inference-grounding-dino`,
`pixano-inference-transformers-vlm`, `pixano-inference-vllm`.
