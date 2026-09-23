<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# pixano-inference-transformers-vlm

Vision-language models (visual question answering, captioning) for Pixano Inference, served
through Hugging Face Transformers and distributed as a standalone plugin package. It serves
the `vlm` capability.

LLaVA, LLaVA-NeXT and LLaVA-NeXT-Video checkpoints are loaded with their dedicated classes
(selected from `model_type`, or from the checkpoint path); anything else loads through
`AutoModelForImageTextToText`.

## Install

```bash
pip install pixano-inference-transformers-vlm
```

## Use

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig
from pixano_inference_transformers_vlm import TransformersVLMParams

models = [
    ModelConfig(
        name="llava",
        model_class="TransformersVLMModel",
        model_params=TransformersVLMParams(path="llava-hf/llava-1.5-7b-hf", config={"dtype": "bfloat16"}),
        deployment=DeploymentConfig(num_gpus=1),
    ),
]
```

## Development

The package is self-contained, with its own lock file:

```bash
cd packages/pixano-inference-transformers-vlm
uv sync
uv run pytest
```
