<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# pixano-inference-sam3

[SAM 3](https://github.com/facebookresearch/sam3) open-vocabulary detection for Pixano Inference,
distributed as a standalone **plugin package**. It registers one detection model,
`Sam3DetectionModel`, which finds objects:

- **by text**: each class the request names is its own prompt (`classes=["screw", "nut"]`);
- **by exemplars** ("more like these"): `exemplar_boxes`, boxes around a few instances in the
  image, and optionally `negative_exemplar_boxes`, around look-alikes not to find. The class the
  request names (at most one) is used as text alongside them; with none the prompt is visual only.

The image is encoded once per request, whatever the number of prompts. Detections carry a box
(`[x1, y1, x2, y2]`, pixels), a score, the class and, unless `return_masks` is off, the instance
mask. The model declares all of it in its `interface` (`exemplars: true`), which is how a client
such as Pixano knows to offer exemplars.

## Install

```bash
pip install pixano-inference-sam3
```

It brings Meta's `sam3` package (0.1.4, from PyPI) and PyTorch. For a CPU-only machine install
PyTorch's CPU build first (`--torch-backend=cpu` with uv); SAM 3 is slow on CPU, a GPU is what it
is meant for.

## Weights: gated, and fetched once

- **The `facebook/sam3` checkpoint is gated** on Hugging Face. Request access on
  [its page](https://huggingface.co/facebook/sam3), then make a token of that account available to
  the server (`HF_TOKEN`, or `huggingface-cli login`). The first start downloads `sam3.pt` into the
  Hugging Face cache; later starts, including offline ones (`HF_HUB_OFFLINE=1`), read it from there.
  `path` may instead name another repository holding a `sam3.pt` (a fine-tuned one) or a local
  checkpoint file.
- **The `sam3` 0.1.4 wheel omits its text tokenizer's vocabulary.** On first start the plugin
  fetches it from SAM 3's repository at a pinned commit, checks its SHA-256 and keeps it in
  `$HF_HOME/sam3/`. On a machine without internet access, put the file there beforehand, or point
  `bpe_path` at it.

## Configure

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig
from pixano_inference_sam3 import Sam3DetectionParams

models = [
    ModelConfig(
        name="sam3",
        model_class="Sam3DetectionModel",
        model_params=Sam3DetectionParams(),  # path="facebook/sam3", return_masks=True
        deployment=DeploymentConfig(num_gpus=1),
    ),
]
```

```bash
pixano-inference --config models.py
```

In the server's Docker image, add it to the bundled packages:

```bash
docker build --build-arg MODEL_PACKAGES="pixano-inference-sam3" -t pixano-inference:sam3 .
```

## Call

```python
from pixano_inference.client import SyncPixanoInferenceClient
from pixano_inference.schemas import DetectionRequest

client = SyncPixanoInferenceClient("http://localhost:7463")
result = client.detection(
    DetectionRequest(
        model="sam3",
        image="https://example.com/workbench.jpg",
        classes=["screw"],
        exemplar_boxes=[[412, 230, 448, 301], [610, 245, 641, 318]],  # two screws, in pixels
        box_threshold=0.4,
    )
)
print(result.data.boxes, result.data.scores)
```

`box_threshold` is SAM 3's score threshold (`text_threshold` is not used). Boxes, exemplars
included, are in pixels of the image as the server decodes it, like every Pixano Inference model.

## Licence

This plugin's code is under the CeCILL-C licence (`LICENSE`). **SAM 3 itself, its code and its
weights, are distributed by Meta under the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)**,
which installing this package makes you subject to. Among other terms it prohibits uses related to
military, warfare or nuclear applications; check it against your own use before deploying.

## Tests

The tests replace SAM 3's processor with a fake, so they need neither a GPU nor the weights:

```bash
uv sync
uv run pytest
```
