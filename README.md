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

An inference server for the [Pixano](https://pixano.github.io/pixano/latest/) annotation tool.
It runs models on [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) and exposes them
through a REST API and a Python client.

Each model lives in its own Python package. Install the ones you need and the server finds them.

## Quickstart

This runs SAM2 and segments an object from a single click. You need Python 3.10 to 3.13.

```bash
pip install "pixano-inference[sam]"
pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
```

Write the server configuration in `models.py`:

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig
from pixano_inference_sam import Sam2ImageParams

models = [
    ModelConfig(
        name="sam2-image",
        model_class="Sam2ImageModel",
        model_params=Sam2ImageParams(path="facebook/sam2-hiera-base-plus"),
        deployment=DeploymentConfig(num_gpus=1),  # 0 to run on CPU
    ),
]
```

Start the server. The first start downloads the weights.

```bash
pixano-inference --config models.py
```

When http://localhost:7463/v1/ready answers `"ready": true`, run this from another terminal:

```python
from pixano_inference.client import SyncPixanoInferenceClient
from pixano_inference.schemas import SegmentationRequest

client = SyncPixanoInferenceClient("http://localhost:7463")
result = client.segmentation(
    SegmentationRequest(
        model="sam2-image",
        image="https://raw.githubusercontent.com/pixano/pixano-inference/main/docs/assets/examples/sam2/truck.jpg",
        points=[[[500, 375]]],  # one click, in pixels
        labels=[[1]],  # 1: the click is on the object
    )
)

scores = result.data.scores.to_numpy().ravel()
best = scores.argmax()
mask = result.data.masks[0][best].to_mask()
print(f"score {scores[best]:.2f}, mask of {mask.sum()} pixels")
```

## Models

| Install                                | Models                                     |
| -------------------------------------- | ------------------------------------------ |
| `pip install "pixano-inference[sam]"`  | SAM2 image segmentation and video tracking |
| `pip install "pixano-inference[clip]"` | Image and text embeddings (MobileCLIP2)    |

Each package has its own README in [`packages/`](packages).

An application that only sends requests to a server needs `pip install pixano-inference`, which
installs the client without the server.

## Your own model

A model is a small Python package. The [YOLO example](examples/yolo) shows a detector and a tracker,
and the [numpy detector](examples/numpy_detector) a model with no ML framework at all. The
[custom models guide](docs/ray_serve/custom_models.md) explains the rest.

## Development

```bash
git clone https://github.com/pixano/pixano-inference.git
cd pixano-inference
uv sync
uv run pytest -m "not integration" tests/
```

Each package under `packages/` has its own environment and tests, for example:

```bash
uv run --project packages/pixano-inference-sam pytest packages/pixano-inference-sam/tests
```

The [documentation](https://pixano.github.io/pixano-inference/latest/) covers the API, Docker
deployment and autoscaling.

## License

Pixano-Inference is released under the terms of the [CeCILL-C license](LICENSE).
