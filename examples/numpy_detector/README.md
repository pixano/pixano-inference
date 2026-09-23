<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Example custom model: `pixano-numpy-detector`

A complete, **framework-free** (numpy-only) custom detection model, packaged as an
independent, installable Python distribution. This is the recommended way to extend Pixano
Inference: your model lives in its own repository, is versioned and shared like any package,
and the server discovers it automatically — no source directory to ship to workers, and it
is importable in every Ray Serve worker because it is installed in the environment.

## How discovery works

The package advertises the model through the `pixano_inference.models` entry point in its
[`pyproject.toml`](./pyproject.toml):

```toml
[project.entry-points."pixano_inference.models"]
numpy_detector = "pixano_numpy_detector.model"   # importing the module runs @register_model
```

At startup Pixano Inference loads every entry point in that group, so a `pip install` is all
it takes to make a model available by name.

## Try it

The package is self-contained (its own `pyproject.toml` and `uv.lock`), and its environment
holds the Pixano Inference core plus the model, installed editable for live iteration:

```bash
uv sync --project examples/numpy_detector
uv run --project examples/numpy_detector pytest examples/numpy_detector/tests
# or, from your own model repo:  pip install pixano-numpy-detector
```

Write a config that references the model **by name** (no import needed):

```python
# models.py
from pixano_inference.configs import DeploymentConfig, ModelConfig

models = [
    ModelConfig(
        name="numpy-detector",
        model_class="NumpyDetector",
        model_params={"threshold": 20},
        deployment=DeploymentConfig(num_gpus=0, num_cpus=1),
    ),
]
```

Start the server and call it:

```bash
uv run --project examples/numpy_detector pixano-inference --host 0.0.0.0 --port 7463 --config models.py
```

```python
from pixano_inference.client import SyncPixanoInferenceClient
from pixano_inference.schemas import DetectionRequest

client = SyncPixanoInferenceClient("http://localhost:7463")
result = client.detection(DetectionRequest(model="numpy-detector", image="data:image/png;base64,..."))
print(result.data.boxes, result.data.classes)
```

## Making your own

1. Copy this package layout.
2. Subclass one of the capability base classes (`DetectionModel`, `SegmentationModel`,
   `TrackingModel`, `VLMModel`, `NERModel`), implement `load_model()` and `predict()`, and
   decorate it with `@register_model("YourModel")`.
3. Declare the `pixano_inference.models` entry point pointing at your module.
4. Publish it (PyPI, a private index, or a git URL) and `pip install` it wherever the server
   runs. It will be picked up automatically.
