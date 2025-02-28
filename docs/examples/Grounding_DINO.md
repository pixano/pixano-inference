<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Grounding DINO

## Connect to the client

```python
from pixano_inference.client import PixanoInferenceClient


client = PixanoInferenceClient.connect(url="http://localhost:8000")
client.models
```

## Instantiate the model

In this example, we will use the transformers implementation of Grounding DINO.

```python
from pixano_inference.pydantic import ModelConfig
from pixano_inference.tasks import ImageTask


await client.instantiate_model(
    provider="transformers",
    config=ModelConfig(
        name="grounding_dino",
        task=ImageTask.ZERO_SHOT_DETECTION.value,
        path="IDEA-Research/grounding-dino-tiny",
        config={},
    ),
)
```

## Prepare the request

```python
from pixano_inference.pydantic import ImageZeroShotDetectionRequest
request = ImageZeroShotDetectionRequest(
    model="grounding_dino",
    classes=["a cat", "a remote control"],
    box_threshold=0.3,
    text_threshold=0.2,
    image="http://images.cocodataset.org/val2017/000000039769.jpg",
)
```

## Call the model for inference (synchronous)

```python
from pixano_inference.pydantic import ImageZeroShotDetectionResponse


sync_response: ImageZeroShotDetectionResponse = await client.image_zero_shot_detection(request)
```

## Call the model for inference (asynchronous)

```python
from time import sleep

from pixano_inference.pydantic import CeleryTask, ImageZeroShotDetectionResponse

celery_response: CeleryTask = await client.image_zero_shot_detection(request, asynchronous=True)

sleep(5)

async_response: ImageZeroShotDetectionResponse = await client.image_zero_shot_detection(
    request=None, asynchronous=True, task_id=celery_response.id
)
```

## Delete the model (optional)

```python
await client.delete_model("grounding_dino")
```
