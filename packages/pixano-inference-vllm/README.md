<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# pixano-inference-vllm

Vision-language models for Pixano Inference, served by [vLLM](https://docs.vllm.ai) and
distributed as a standalone plugin package. It serves the `vlm` capability.

vLLM runs on Linux with a GPU; on other platforms the package installs without it (so its
tests run everywhere), but the model cannot be loaded.

## Install

```bash
pip install pixano-inference-vllm
```

Not on PyPI yet: install from the repository (`uv` also installs the core from it):

```bash
uv pip install "pixano-inference-vllm @ git+https://github.com/pixano/pixano-inference#subdirectory=packages/pixano-inference-vllm"
# or, from a clone:  uv pip install ./packages/pixano-inference-vllm
```

## Use

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig
from pixano_inference_vllm import VLLMVLMParams

models = [
    ModelConfig(
        name="qwen-vl",
        model_class="VLLMVLMModel",
        model_params=VLLMVLMParams(path="Qwen/Qwen2-VL-7B-Instruct", config={"max_model_len": 8192}),
        deployment=DeploymentConfig(num_gpus=1),
    ),
]
```

`config` is forwarded to `vllm.LLM`. Prompts must use the chat format, with images embedded
as `image_url` content parts.

## Development

The package is self-contained, with its own lock file:

```bash
cd packages/pixano-inference-vllm
uv sync
uv run pytest
```
