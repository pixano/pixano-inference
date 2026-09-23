<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# pixano-inference-grounding-dino

[Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino) zero-shot
(open-vocabulary) object detection for Pixano Inference, distributed as a standalone plugin
package built on Hugging Face Transformers. It serves the `detection` capability.

## Install

```bash
pip install pixano-inference-grounding-dino
```

Installing the package is enough: the server discovers `GroundingDINOModel` through its
`pixano_inference.models` entry point.

## Use

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig
from pixano_inference_grounding_dino import GroundingDINOParams

models = [
    ModelConfig(
        name="grounding-dino",
        model_class="GroundingDINOModel",
        model_params=GroundingDINOParams(path="IDEA-Research/grounding-dino-tiny"),
        deployment=DeploymentConfig(num_gpus=1),
    ),
]
```

Requests must provide `classes` (the text prompt); `box_threshold` and `text_threshold`
filter the detections.

## Development

The package is self-contained, with its own lock file:

```bash
cd packages/pixano-inference-grounding-dino
uv sync
uv run pytest
```
