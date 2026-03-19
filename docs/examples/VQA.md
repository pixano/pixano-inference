<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Visual Question Answering

## Python config

Deploy a Transformers-backed vision-language model:

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig, TransformersVLMParams


models = [
    ModelConfig(
        name="llava-qwen",
        model_class="TransformersVLMModel",
        model_params=TransformersVLMParams(
            path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            config={"torch_dtype": "bfloat16"},
        ),
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

## Run inference

Build a request with an image and a text prompt, then call the model:

```python
import asyncio

from pixano_inference.schemas import VLMRequest


async def main():
    request = VLMRequest(
        model="llava-qwen",
        prompt=[
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/9/9e/Ours_brun_parcanimalierpyrenees_1.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is displayed in this image? Answer with a high level of description.",
                    },
                ],
                "role": "user",
            }
        ],
        images=None,
        max_new_tokens=1000,
    )

    response = await client.vlm(request)
    print(response.data.generated_text)


asyncio.run(main())
```
