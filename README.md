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

**Pixano-Inference is an inference library for Pixano.**

**_Under active development, subject to API change_**

[![GitHub version](https://img.shields.io/github/v/release/pixano/pixano-inference?label=release&logo=github)](https://github.com/pixano/pixano-inference/releases)
[![PyPI version](https://img.shields.io/pypi/v/pixano-inference?color=blue&label=release&logo=pypi&logoColor=white)](https://pypi.org/project/pixano-inference/)
[![Tests](https://img.shields.io/github/actions/workflow/status/pixano/pixano-inference/test_back.yml?branch=main)](https://github.com/pixano/pixano-inference/actions/workflows/test_back.yml)
[![Documentation](https://img.shields.io/website?url=https%3A%2F%2Fpixano.github.io%2Fpixano-inference%2F&up_message=online&down_message=offline&label=docs)](https://pixano.github.io/pixano-inference/)
[![Python version](https://img.shields.io/pypi/pyversions/pixano-inference?color=important&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CeCILL--C-blue.svg)](LICENSE)

</div>

<hr />

# Pixano-Inference

A [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) inference server for the
[Pixano](https://pixano.github.io/pixano/latest/) annotation tool, with a REST API and a Python client.

Models are independent Python packages with their own code, dependencies, and environments.
The server discovers installed models automatically; the core requires no ML framework.

## Quickstart

Requires Python 3.10–3.13. Install Pixano-Inference with the Grounding DINO model:

```bash
pip install "pixano-inference[grounding-dino]"
```

Create `models.py`:

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig
from pixano_inference_grounding_dino import GroundingDINOParams

models = [
    ModelConfig(
        name="grounding-dino",
        model_class="GroundingDINOModel",
        model_params=GroundingDINOParams(path="IDEA-Research/grounding-dino-tiny"),
        deployment=DeploymentConfig(num_gpus=0),  # Set to 1 to use a GPU
    ),
]
```

Start the server. The first run downloads the model weights.

```bash
pixano-inference --config models.py
```

Check [readiness](http://localhost:7463/v1/ready), then save this request as `predict.py`:

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

Run it in another terminal, using the same Python environment:

```bash
python predict.py
```

Applications calling an existing server only need [pixano-inference-client](packages/pixano-inference-client).
See the [documentation](https://pixano.github.io/pixano-inference/latest/) for the API,
Docker deployment, and autoscaling.

## Model packages

Choose a model with an extra, for example `pip install "pixano-inference[sam]"`:

| Extra              | Package                                                        | Supports                                     |
| ------------------ | -------------------------------------------------------------- | -------------------------------------------- |
| `sam`              | [SAM](packages/pixano-inference-sam)                           | SAM2 image segmentation and video tracking   |
| `clip`             | [CLIP](packages/pixano-inference-clip)                         | Image/text embeddings, including MobileCLIP2 |
| `grounding-dino`   | [Grounding DINO](packages/pixano-inference-grounding-dino)     | Object detection from text prompts           |
| `transformers-vlm` | [Transformers VLM](packages/pixano-inference-transformers-vlm) | Hugging Face vision-language models          |
| `vllm`             | [vLLM](packages/pixano-inference-vllm)                         | Vision-language models on Linux GPUs         |

Each extra installs an independent model package. SAM also needs the upstream `sam-2`
library; follow its package's installation instructions.

## Development

For source development, clone the repository and use [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/pixano/pixano-inference.git
cd pixano-inference
uv sync
uv run pytest -m "not integration" tests/  # Core unit tests
uv run --project packages/pixano-inference-sam pytest packages/pixano-inference-sam/tests
```

Develop and test each model in its own package, with its own `pyproject.toml`, `uv.lock`,
and tests. To build a custom model, start with the [numpy detector](examples/numpy_detector)
and follow the [guide](docs/ray_serve/custom_models.md) and
[package specification](docs/ray_serve/custom_model_spec.md).

## License

Pixano-Inference is released under the terms of the [CeCILL-C license](LICENSE).
