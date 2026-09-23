<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# pixano-inference-sam

The SAM2 models (image segmentation and video tracking) for Pixano Inference, distributed as
a standalone **plugin package** — a first-party example of the "model store" pattern. The
server discovers it automatically through the `pixano_inference.models` entry point.

## Install

```bash
pip install pixano-inference-sam
# the upstream SAM2 library is only on git, so install it too:
pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
```

The first command pulls in the Pixano Inference core and PyTorch; the second installs
Facebook's `sam-2` library (not available on PyPI).

Not on PyPI yet: install from the repository (`uv` also installs the core from it):

```bash
uv pip install "pixano-inference-sam @ git+https://github.com/pixano/pixano-inference#subdirectory=packages/pixano-inference-sam"
# or, from a clone:  uv pip install ./packages/pixano-inference-sam
```

## Models

| Registered name  | Capability     | Params class      | Default checkpoint                |
| ---------------- | -------------- | ----------------- | --------------------------------- |
| `Sam2ImageModel` | `segmentation` | `Sam2ImageParams` | `facebook/sam2-hiera-base-plus`   |
| `Sam2VideoModel` | `tracking`     | `Sam2VideoParams` | `facebook/sam2-hiera-large`       |

## Use

Reference the model by name in a config file (no import needed once installed):

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig
from pixano_inference_sam import Sam2ImageParams

models = [
    ModelConfig(
        name="sam2-image",
        model_class="Sam2ImageModel",
        model_params=Sam2ImageParams(path="facebook/sam2-hiera-base-plus"),
        deployment=DeploymentConfig(num_gpus=1, num_cpus=2),
    ),
]
```

## Development

The package is self-contained, with its own lock file:

```bash
cd packages/pixano-inference-sam
uv sync          # the dev group includes the git-only sam-2 library
uv run pytest
```
