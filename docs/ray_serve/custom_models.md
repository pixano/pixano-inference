<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Custom Models Guide

Extend Pixano Inference with your own models **without modifying the `pixano_inference`
package**. A custom model is an ordinary, installable Python package that advertises itself
through an entry point; once installed in the server's environment, it is discovered
automatically — and because it is installed (not a loose directory), it is importable in
every Ray Serve worker with no extra plumbing.

This is the foundation for a shared **model store**: teams keep each model in its own
repository, version and publish it like any package, and install the ones they need. The
first-party models (SAM2, CLIP, Grounding DINO, VLMs) are distributed exactly this way: each is
a self-contained package under
[`packages/`](https://github.com/pixano/pixano-inference/tree/main/packages).

A complete, runnable example lives in
[`examples/numpy_detector`](https://github.com/pixano/pixano-inference/tree/main/examples/numpy_detector)
— a framework-free (numpy-only) detector.

## 1. Choose a base class

| Base class          | Capability     | Input type          | Output type          |
| ------------------- | -------------- | ------------------- | -------------------- |
| `SegmentationModel` | `segmentation` | `SegmentationInput` | `SegmentationOutput` |
| `DetectionModel`    | `detection`    | `DetectionInput`    | `DetectionOutput`    |
| `TrackingModel`     | `tracking`     | `TrackingInput`     | `TrackingOutput`     |
| `VLMModel`          | `vlm`          | `VLMInput`          | `VLMOutput`          |
| `NERModel`          | `ner`          | `NERInput`          | `NEROutput`          |

The capability is inferred from the base class, so your model is served on the matching
`/v1/inference/<capability>` route automatically.

## 2. Write the model

Any framework works (PyTorch, JAX, TensorFlow, MLX, or plain numpy) — the core contract is
framework-agnostic and the core depends on no ML framework, so your package declares the one it
uses. PyTorch models can reuse the device/dtype/tensor helpers of `pixano-inference-torch`.
Subclass a base class, implement `load_model()` and `predict()`, and register it with
`@register_model`.

```python
# src/my_pkg/model.py
from pixano_inference.models import DetectionInput, DetectionModel, DetectionOutput, register_model


@register_model("MyDetector")
class MyDetector(DetectionModel):
    def load_model(self) -> None:
        self.model = load_weights(self.config.model_params["path"])

    def predict(self, input: DetectionInput) -> DetectionOutput:
        boxes, scores, classes = self.model(input.image)
        return DetectionOutput(boxes=boxes, scores=scores, classes=classes)
```

`load_model()` runs once per replica; `unload()` (optional) runs when the replica is torn
down. Weights are resolved from `self.config.model_params` (e.g. a HuggingFace id or a path).

Two rules keep a model package self-contained and cheap to discover:

- **Declare every dependency you import** (your framework included) in your own
  `pyproject.toml`. The core brings none of the ML stack, and nothing should rely on what
  another package happens to install.
- **Import the framework lazily**, inside `load_model()` / `predict()` rather than at module
  level. The server imports every installed model package at startup to discover its models;
  a lazy import keeps that cheap and means a broken framework install surfaces when the model
  is deployed, with a clear error, instead of hiding the whole package from discovery.

## 3. Declare the entry point

In your package's `pyproject.toml`, advertise the model under the `pixano_inference.models`
group. The value is the module to import (importing it runs `@register_model`):

```toml
[project]
name = "my-pkg"
dependencies = ["pixano-inference", "torch"]  # the core plus whatever the model imports

[project.entry-points."pixano_inference.models"]
my_detector = "my_pkg.model"        # or "my_pkg.model:MyDetector" to point at the class
```

## 4. Install and deploy

Publishing is optional: a model package can stay private and install straight from its
source. Any of these puts it in the server's environment, where it is discovered:

```bash
uv pip install ./my-pkg                                        # a local directory
uv pip install -e ./my-pkg                                     # editable, for live iteration
uv pip install "my-pkg @ git+ssh://git@github.com/acme/my-pkg.git"          # a private repository
uv pip install "my-pkg @ git+https://github.com/acme/models.git#subdirectory=my-pkg"  # a subdirectory
uv build && uv pip install --find-links dist my-pkg            # wheels copied to the server
# pip install my-pkg                                           # from PyPI or a private index
```

Since your package depends on `pixano-inference`, its own environment already contains the
server: `uv sync && uv run pixano-inference --config models.py` runs it with exactly your
package's locked dependencies. The first-party packages under `packages/` work this way.

### Depending on the core

`dependencies = ["pixano-inference"]` resolves the core from PyPI. To build against a
specific commit, a fork, or a core version that is not published, point `uv` at git:

```toml
[tool.uv.sources]
pixano-inference = { git = "https://github.com/pixano/pixano-inference", tag = "v0.7.0" }  # or branch = / rev =
```

`uv` follows that source wherever the package is installed from (a path, a git URL, an sdist).
Plain `pip` ignores `[tool.uv.sources]`; a private package can instead declare the core as a
direct reference (`"pixano-inference @ git+https://github.com/pixano/pixano-inference@v0.7.0"`), which PyPI would
reject but a private index or git install accepts. A PyTorch model that uses
`pixano-inference-torch` declares it the same way.

Reference the model **by name** in a config file — no import needed, because the entry point
already registered it:

```python
# models.py
from pixano_inference.configs import DeploymentConfig, ModelConfig

models = [
    ModelConfig(
        name="my-detector",
        model_class="MyDetector",
        model_params={"path": "/models/my_detector.pt"},
        deployment=DeploymentConfig(num_gpus=1, num_cpus=2, min_replicas=1, max_replicas=4),
    ),
]
```

```bash
pixano-inference --host 0.0.0.0 --port 7463 --config models.py
```

Or deploy it at runtime through the admin API (`POST /v1/models`).

## Deployment tuning

`DeploymentConfig` controls how Ray Serve runs your model:

| Field                   | Meaning                                                        |
| ----------------------- | -------------------------------------------------------------- |
| `num_gpus` / `num_cpus` | Resources per replica.                                         |
| `min_replicas`          | 0 enables scale-to-zero; ≥1 keeps replicas warm.               |
| `max_replicas`          | Upper bound for autoscaling.                                   |
| `max_ongoing_requests`  | Concurrent requests per replica before queueing / scaling up.  |
| `max_batch_size`        | >1 enables batching (implement `predict_batch` to exploit it). |
| `timeout_s`             | Per-request inference timeout (else a per-capability default). |

## Notes

- **Package, don't ship directories.** Defining a model in `__main__` or a loose script and
  relying on it being sent to workers is fragile; an installed package is imported normally
  everywhere. The server warns if a configured class is defined in `__main__`.
- **Framework-agnostic.** The `examples/numpy_detector` package needs no ML framework at all,
  which shows the contract does not privilege PyTorch.
