<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

<div align="center">

<img src="https://raw.githubusercontent.com/pixano/pixano/main/docs/assets/pixano_wide.png" alt="Pixano" height="100"/>

<br/>
<br/>

**Pixano-Inference is an open-source inference library for Pixano.**

**_Under active development, subject to API change_**

[![GitHub version](https://img.shields.io/github/v/release/pixano/pixano-inference?label=release&logo=github)](https://github.com/pixano/pixano-inference/releases)
[![PyPI version](https://img.shields.io/pypi/v/pixano-inference?color=blue&label=release&logo=pypi&logoColor=white)](https://pypi.org/project/pixano-inference/)
[![Tests](https://img.shields.io/github/actions/workflow/status/pixano/pixano-inference/test_back.yml?branch=develop)](https://github.com/pixano/pixano-inference/actions/workflows/test_back.yml)
[![Documentation](https://img.shields.io/website?url=https%3A%2F%2Fpixano.github.io%2F&up_message=online&down_message=offline&label=docs)](https://pixano.github.io)
[![Python version](https://img.shields.io/pypi/pyversions/pixano-inference?color=important&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CeCILL--C-blue.svg)](LICENSE)

</div>

<hr />

# Pixano-Inference

A [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) inference server built for the
[Pixano](https://pixano.github.io/pixano/latest/) annotation tool: typed model configs, a REST
API and a Python client. The core ships no model and depends on no ML framework. Each model is
a separate package that brings its own framework, and the server discovers every installed one.

## Install

```bash
pip install pixano-inference                    # the server; ships no model
pip install pixano-inference[grounding-dino]    # add a model
```

Each extra installs one model package, a separate distribution with its own framework:

| Extra              | Models                                                             |
| ------------------ | ------------------------------------------------------------------ |
| `sam`              | SAM2 image segmentation and video tracking (plus `sam-2` from git) |
| `clip`             | CLIP-style image/text embeddings (MobileCLIP2)                     |
| `grounding-dino`   | Grounding DINO zero-shot detection                                 |
| `transformers-vlm` | Vision-language models through Hugging Face                        |
| `vllm`             | Vision-language models served by vLLM (Linux, GPU)                 |
| `torch`            | PyTorch helpers for your own model                                 |

## First model

`models.py`:

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig
from pixano_inference_grounding_dino import GroundingDINOParams

models = [
    ModelConfig(
        name="grounding-dino",
        model_class="GroundingDINOModel",
        model_params=GroundingDINOParams(path="IDEA-Research/grounding-dino-tiny"),
        deployment=DeploymentConfig(num_gpus=1),  # 0 on a CPU-only host
    )
]
```

```bash
pixano-inference --host 0.0.0.0 --port 7463 --config models.py
curl http://localhost:7463/v1/ready   # {"ready":true,"models":{"grounding-dino":"RUNNING"},...}
```

The first start downloads the weights. `--config` is optional: models can also be deployed
later with `POST /v1/models`.

```python
from pixano_inference_client import DetectionRequest, SyncPixanoInferenceClient

client = SyncPixanoInferenceClient("http://localhost:7463")
result = client.detection(
    DetectionRequest(
        model="grounding-dino",
        image="https://raw.githubusercontent.com/pixano/pixano-inference/main/docs/assets/examples/sam2/bedroom/00000.jpg",
        classes=["bed", "lamp"],
        box_threshold=0.3,
        text_threshold=0.25,
    )
)
print(result.data.classes, result.data.boxes, result.data.scores)
```

Applications that only call a server install `pixano-inference-client` alone (httpx, pydantic,
numpy). Docker images, autoscaling and the API: see the
[documentation](https://pixano.github.io/pixano-inference/latest/).

## Development

```bash
uv sync && uv run pytest -m "not integration"                                        # the framework-free core
uv run --project packages/pixano-inference-sam pytest packages/pixano-inference-sam/tests  # one model, in its own environment
uv run --project packages/pixano-inference-sam pixano-inference --config models.py
```

Every package under `packages/` and `examples/` has its own `pyproject.toml`, `uv.lock` and
tests. Your own model is a package like these and can stay private: see the
[custom models guide](docs/ray_serve/custom_models.md).

## License

Pixano-Inference is released under the terms of the [CeCILL-C license](LICENSE).
