<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Quickstart: Deploy Built-in Models

This guide walks you through deploying built-in models and running your first
inference requests.

## Prerequisites

- Python 3.10+
- A GPU is recommended for production use but not required for testing

## Installation

Install Pixano-Inference with the model-specific extras you need:

=== "SAM2 (segmentation)"

    ```bash
    uv sync --extra sam2
    ```

    or with `pip`:

    ```bash
    pip install pixano-inference[sam2]
    ```

=== "Transformers (detection, VQA)"

    ```bash
    uv sync --extra transformers
    ```

    or with `pip`:

    ```bash
    pip install pixano-inference[transformers]
    ```

=== "All extras"

    ```bash
    uv sync --extra sam2 --extra transformers --extra vllm
    ```

    or with `pip`:

    ```bash
    pip install pixano-inference[sam2,transformers,vllm]
    ```

## Write a Python config

Create a file called `models.py` that declares which models to deploy.
Here is a minimal example deploying SAM2 for image segmentation:

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig, Sam2ImageParams


models = [
    ModelConfig(
        name="sam2-image",
        task="image_mask_generation",
        model_class="Sam2ImageModel",
        model_params=Sam2ImageParams(
            path="facebook/sam2-hiera-base-plus",
            torch_dtype="bfloat16",
        ),
        deployment=DeploymentConfig(
            num_gpus=1,
            min_replicas=0,
            max_replicas=2,
            max_batch_size=8,
        ),
    )
]
```

!!! note "CPU-only testing"
    Set `num_gpus: 0` and `torch_dtype: float32` to run on CPU.
    This is useful for testing but not recommended for production.

## Start the server

### From the CLI

```bash
pixano-inference --config models.py
```

### Programmatically

```python
from pixano_inference.ray import InferenceServer

server = InferenceServer()
server.register_from_config("models.py")
server.start(blocking=True)
```

## Verify the deployment

Check the health endpoint:

```bash
curl http://localhost:7463/health
```

List deployed models:

```bash
curl http://localhost:7463/app/models/
```

You should see your model in the response:

```json
{
  "models": [
    {
      "name": "sam2-image",
      "task": "image_mask_generation",
    "model_class": "Sam2ImageModel"
    }
  ]
}
```

## Send inference requests

### With curl

```bash
# Encode an image to base64
IMAGE_B64=$(base64 -i your_image.png)

curl -X POST http://localhost:7463/tasks/image/mask_generation/ \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sam2-image",
    "image": "data:image/png;base64,'"$IMAGE_B64"'",
    "points": [[[200, 175]]],
    "labels": [[1]]
  }'
```

### With the Python client

```python
import asyncio
from pixano_inference.client import PixanoInferenceClient
from pixano_inference.schemas import SegmentationRequest


async def main():
    client = PixanoInferenceClient.connect("http://localhost:7463")

    request = SegmentationRequest(
        model="sam2-image",
        image="data:image/png;base64,...",  # Base64-encoded image
        points=[[[200, 175]]],              # Point prompt (x, y)
        labels=[[1]],                       # 1 = foreground
    )
    response = await client.segmentation(request)

    print(f"Status: {response.status}")
    print(f"Processing time: {response.processing_time:.3f}s")
    print(f"Masks: {len(response.data.masks[0])}")
    print(f"Scores: {response.data.scores.to_numpy()}")

    # Decode a mask to a numpy array
    mask = response.data.masks[0][0].to_mask()
    print(f"Mask shape: {mask.shape}")


asyncio.run(main())
```

## Built-in models

The following model classes are available out of the box:

| Model class | Task | Extra required | Example `model_params` |
|---|---|---|---|
| `Sam2ImageModel` | `image_mask_generation` | `sam2` | `path: facebook/sam2-hiera-base-plus` |
| `Sam2VideoModel` | `video_mask_generation` | `sam2` | `path: facebook/sam2-hiera-large` |
| `GroundingDINOModel` | `image_zero_shot_detection` | `transformers` | `path: IDEA-Research/grounding-dino-base` |
| `TransformersVLMModel` | `text_image_conditional_generation` | `transformers` | `path: llava-hf/llava-1.5-7b-hf` |
| `VLLMVLMModel` | `text_image_conditional_generation` | `vllm` | `path: Qwen/Qwen2-VL-7B-Instruct` |

## Multi-model config

You can deploy multiple models in a single config file. Each model gets its own
Ray actor with dedicated resources:

```python
from pixano_inference.configs import (
    DeploymentConfig,
    GroundingDINOParams,
    ModelConfig,
    Sam2ImageParams,
    Sam2VideoParams,
)


models = [
    ModelConfig(
        name="sam2-image",
        task="image_mask_generation",
        model_class="Sam2ImageModel",
        model_params=Sam2ImageParams(path="facebook/sam2-hiera-base-plus"),
        deployment=DeploymentConfig(num_gpus=1, max_batch_size=8),
    ),
    ModelConfig(
        name="sam2-video",
        task="video_mask_generation",
        model_class="Sam2VideoModel",
        model_params=Sam2VideoParams(path="facebook/sam2-hiera-large"),
        deployment=DeploymentConfig(num_gpus=1, max_batch_size=1),
    ),
    ModelConfig(
        name="grounding-dino",
        task="image_zero_shot_detection",
        model_class="GroundingDINOModel",
        model_params=GroundingDINOParams(path="IDEA-Research/grounding-dino-base"),
        deployment=DeploymentConfig(num_gpus=1, max_batch_size=8),
    ),
]
```

!!! tip
    See [custom_models.md](custom_models.md) for external model modules via
    `model_module`, and [`deploy/sam2_example.py`](https://github.com/pixano/pixano-inference/blob/main/deploy/sam2_example.py)
    for a typed config example.

## Deployment configuration reference

Each `ModelConfig(...)` entry supports the following fields:

### Model fields

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *required* | Unique model name |
| `task` | `str` | *required* | Task string (e.g. `image_mask_generation`) |
| `model_class` | `str \| type` | *required* | Registered model class name or class object |
| `model_module` | `str` | `None` | Python module to import before resolving `model_class` (e.g. `my_package.models`). Used for [external custom models](custom_models.md). |
| `model_params` | `dict \| BaseModelParams` | `{}` | Parameters passed to the model (e.g. `path`, `torch_dtype`) |

### Deployment fields (under `deployment:`)

| Field | Type | Default | Description |
|---|---|---|---|
| `num_gpus` | `float` | `0.0` | GPUs per replica |
| `num_cpus` | `float` | `1.0` | CPUs per replica |
| `memory_mb` | `int` | `None` | Memory limit in MB (`None` = no limit) |
| `min_replicas` | `int` | `0` | Minimum replicas (0 = scale to zero) |
| `max_replicas` | `int` | `4` | Maximum replicas |
| `target_num_ongoing_requests_per_replica` | `int` | `2` | Autoscaling target |
| `downscale_delay_s` | `float` | `60.0` | Seconds to wait before scaling down |
| `upscale_delay_s` | `float` | `5.0` | Seconds to wait before scaling up |
| `max_batch_size` | `int` | `8` | Maximum batch size for inference |
| `batch_wait_timeout_s` | `float` | `0.1` | Timeout for filling a batch (seconds) |

### Available tasks

| Task string | Description |
|---|---|
| `image_mask_generation` | Generate masks from images (SAM2) |
| `video_mask_generation` | Generate masks from video frames (SAM2) |
| `text_image_conditional_generation` | Generate text from images and prompts (LLaVA) |
| `image_zero_shot_detection` | Detect objects without class-specific training (Grounding DINO) |
| `instance_segmentation` | Detect objects and return masks when the model supports it |

## Next steps

To deploy your own custom models (PyTorch, JAX, TensorFlow, or any framework),
see the [Custom Models Guide](custom_models.md).
