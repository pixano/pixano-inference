<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Getting started with Pixano-Inference

## Installation

=== "uv (recommended)"

    ```bash
    uv add pixano-inference
    ```

    For a development install, clone the repo and sync:

    ```bash
    cd pixano-inference
    uv sync
    ```

=== "pip"

    ```bash
    pip install pixano-inference
    ```

    For a development install:

    ```bash
    cd pixano-inference
    pip install -e .
    ```

Install the extras for the model backends you need:

=== "uv"

    ```bash
    uv sync --extra sam2 --extra transformers --extra vllm
    ```

=== "pip"

    ```bash
    pip install pixano-inference[sam2,transformers,vllm]
    ```

## Usage

Pixano-Inference serves models with GPU-aware deployment, autoscaling, and request
batching. Models are declared in a Python config file, the server deploys them, and you
interact via the Python client or HTTP API.

The workflow is:

1. Write a Python config
2. Start the server
3. Connect the client
4. Run inference

See the [examples](../examples/index.md) section for detailed use cases.

### Step 1: Write a Python config

Create a file called `models.py` that declares which models to deploy.
Here is an example deploying Grounding DINO for zero-shot detection:

```python
from pixano_inference.configs import DeploymentConfig, GroundingDINOParams, ModelConfig


models = [
    ModelConfig(
        name="grounding-dino",
        model_class="GroundingDINOModel",
        model_params=GroundingDINOParams(path="IDEA-Research/grounding-dino-tiny"),
        deployment=DeploymentConfig(num_gpus=1, min_replicas=0, max_replicas=2),
    )
]
```

See the [Server Deployment documentation](../ray_serve/index.md) for all configuration
options, deploying built-in and custom models, and configuring autoscaling.

### Step 2: Start the server

```bash
pixano-inference --config models.py
```

The default address is `http://127.0.0.1:7463`. You can override it with
`--host` and `--port`.

### Step 3: Connect the client

The easiest way to interact with Pixano-Inference is through the Python client.
You can also use the Swagger UI at `http://localhost:7463/docs`.

```python
from pixano_inference.client import PixanoInferenceClient


client = PixanoInferenceClient.connect(url="http://localhost:7463")
```

### Step 4: Run an inference

Build a request and call the appropriate client method:

```python
import asyncio

from pixano_inference.client import PixanoInferenceClient
from pixano_inference.schemas import DetectionRequest


async def main():
    client = PixanoInferenceClient.connect(url="http://localhost:7463")

    request = DetectionRequest(
        model="grounding-dino",
        image="http://images.cocodataset.org/val2017/000000039769.jpg",
        classes=["cat", "remote control"],
        box_threshold=0.3,
        text_threshold=0.2,
    )
    response = await client.detection(request)
    print(response.data.boxes)
    print(response.data.classes)


asyncio.run(main())
```
