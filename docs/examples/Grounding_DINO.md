<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Grounding DINO

## Python config

Deploy Grounding DINO for zero-shot object detection:

```python
from pixano_inference.configs import DeploymentConfig, GroundingDINOParams, ModelConfig


models = [
    ModelConfig(
        name="grounding-dino",
        model_class="GroundingDINOModel",
        model_params=GroundingDINOParams(path="IDEA-Research/grounding-dino-tiny"),
        deployment=DeploymentConfig(num_gpus=1),
    )
]
```

## Start the server

```bash
pixano-inference --config models.py
```

## Connect the client

```python
from pixano_inference.client import PixanoInferenceClient


client = PixanoInferenceClient.connect(url="http://localhost:7463")
```

## Prepare the request

```python
from pixano_inference.schemas import DetectionRequest

request = DetectionRequest(
    model="grounding-dino",
    classes=["a cat", "a remote control"],
    box_threshold=0.3,
    text_threshold=0.2,
    image="http://images.cocodataset.org/val2017/000000039769.jpg",
)
```

## Run inference

```python
import asyncio

async def main():
    response = await client.detection(request)
    print(f"Boxes: {response.data.boxes}")
    print(f"Scores: {response.data.scores}")
    print(f"Classes: {response.data.classes}")
    return response


response = asyncio.run(main())
```
