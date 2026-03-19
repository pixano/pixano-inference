<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Custom Models Guide

This guide shows how to deploy your own models on Pixano-Inference without
modifying the `pixano_inference` package itself.

## Overview

Custom deployment requires three pieces:

1. A model class that subclasses one of the base classes from `pixano_inference.models`
2. A `@register_model(...)` decorator so the class can be resolved by name
3. A Python config file that declares a `ModelConfig` with `model_module`

At startup, the server imports the module named by `model_module`, resolves the
registered class, infers its capability from the base model class, and deploys
it as a Ray Serve actor.

## Choose a base class

| Base class          | Capability     | Input type          | Output type          |
| ------------------- | -------------- | ------------------- | -------------------- |
| `SegmentationModel` | `segmentation` | `SegmentationInput` | `SegmentationOutput` |
| `DetectionModel`    | `detection`    | `DetectionInput`    | `DetectionOutput`    |
| `TrackingModel`     | `tracking`     | `TrackingInput`     | `TrackingOutput`     |
| `VLMModel`          | `vlm`          | `VLMInput`          | `VLMOutput`          |

These are the current HTTP-exposed model families in `pixano_inference.models`.

## Project layout

```text
my_project/
├── models.py
└── my_models/
    ├── __init__.py
    └── segmenter.py
```

## Example custom model

Create `my_models/segmenter.py`:

```python
from __future__ import annotations

import numpy as np

from pixano_inference.models import SegmentationInput, SegmentationModel, SegmentationOutput, register_model
from pixano_inference.ray.config import ModelDeploymentConfig
from pixano_inference.schemas import CompressedRLE, NDArrayFloat


@register_model("MySegmenter")
class MySegmenter(SegmentationModel):
    def __init__(self, config: ModelDeploymentConfig) -> None:
        super().__init__(config)
        self._threshold = float(config.model_params.get("threshold", 0.5))

    def load_model(self) -> None:
        self._path = self.config.model_params["path"]

    def predict(self, input: SegmentationInput) -> SegmentationOutput:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1
        scores = NDArrayFloat.from_numpy(np.array([[self._threshold]], dtype=np.float32))
        return SegmentationOutput(
            masks=[[CompressedRLE.from_mask(mask)]],
            scores=scores,
        )
```

## Python config

Create `models.py` next to your package:

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig


models = [
    ModelConfig(
        name="my-segmenter",
        model_class="MySegmenter",
        model_module="my_models.segmenter",
        model_params={"path": "my-org/my-segmentation-model", "threshold": 0.6},
        deployment=DeploymentConfig(num_gpus=1, min_replicas=0, max_replicas=2, max_batch_size=4),
    )
]
```

## Start the server

If the module is not installed as a package, add the project root to the module
search path:

```bash
cd my_project
pixano-inference --module-path . --config models.py
```

## Verify the deployment

```bash
curl http://localhost:7463/app/models/
```

```bash
curl -X POST http://localhost:7463/inference/segmentation/ \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-segmenter",
    "image": "data:image/png;base64,..."
  }'
```

## Design rules

- Keep heavy imports inside `load_model()` or `predict()` when practical.
- Read deployment-specific options from `self.config.model_params`.
- Return the typed output models from `pixano_inference.models`.
- Use helper types from `pixano_inference.schemas` when the payload contains
  masks or array-like values.
- Pick the correct base model class first; it determines the endpoint family and request contract.
- Respect `self.config.resources.num_gpus` when selecting CPU vs GPU execution.

## Troubleshooting

- `Model class 'MySegmenter' not found`
  Ensure `model_module="my_models.segmenter"` is set and the module is importable.
- Import errors inside the deployment
  Install the third-party dependencies required by your model in the same
  environment that starts Pixano-Inference.
- CPU-only execution
  Set `num_gpus=0` in `DeploymentConfig` and make `load_model()` fall back to CPU.
