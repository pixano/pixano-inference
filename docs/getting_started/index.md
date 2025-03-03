<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Getting started with Pixano-Inference

## Installation

To install the library, simply execute the following command

```bash
pip install pixano-inference
```

If you want to dynamically make changes to the library to develop and test, make a dev install by cloning the repo and executing the following commands

```bash
cd pixano-inference
pip install -e .
```

To have access to the providers available in Pixano-Inference, make sure to install the relevant dependencies. For example to use the vLLM provider run:

```bash
pip install pixano-inference[vllm]
```

## Usage

Pixano-Inference can be used as:

- a Python library to instantiate models from providers and call inference
- a server with the same features as the Library that can run **asynchronously**

See the [examples](../examples/index.md) sections that covers several use cases.

### Run in a python script

#### Instantiate a model

To instantiate a model, you first need to instantiate its provider and then call its `load_model` method.

```python
from pixano_inference.providers import VLLMProvider
from pixano_inference.tasks import MultimodalImageNLPTask

vllm_provider = VLLMProvider()
llava_qwen = vllm_provider.load_model(
    name="llava-qwen",
    task=MultimodalImageNLPTask.CONDITIONAL_GENERATION,
    device=torch.device("cuda"),
    path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    config={
        "dtype": "bfloat16",
    }
)
```

#### Call the model for inference

For the inference, call the method associated to the task you wish to perform. For VQA, it is the `text_image_conditional_generation` method.

```python
from pixano_inference.pydantic import TextImageConditionalGenerationOutput


prompt=[
    {
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/9/9e/Ours_brun_parcanimalierpyrenees_1.jpg"
                },
            },
            {"type": "text", "text": "What is displayed in this image ? Answer with high level of description like 1000 words. "},
        ],
        "role": "user",
    }
]

output: TextImageConditionalGenerationOutput = llava_qwen.text_image_conditional_generation(
    prompt=prompt, max_new_tokens=1000
)
output.generated_text
```

### Run the server

#### Launch the server

Pixano-Inference as a server requires a running [Redis server](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/install-redis-on-linux/). Its URL can be configured in a `.env` file at the root of the server. By default it will serve the localhost at the 6379 port.

Pixano-Inference can invoke a server that will serve the API. To do so, simply execute the following command:

```bash
pixano-inference --port 8000
```

The default port is `8000`. You can change it by passing the `--port` argument.

#### Instantiate the client

The easiest way to interact with Pixano-Inference is through the Python client. However you can take a look at the swagger `http://pixano_inference_url/docs` for using the API REST by yourself.

```python
from pixano_inference.client import PixanoInferenceClient


client = PixanoInferenceClient.connect(url="http://localhost:8000")
```

#### Instantiate a model

To instantiate a model, you need to provide the path of the model file and the task that the model will perform. For example to run a Llava model from the vLLM provider:

```python
from pixano_inference.pydantic import ModelConfig
from pixano_inference.tasks import MultimodalImageNLPTask


await client.instantiate_model(
    provider="vllm",
    config=ModelConfig(
        name="llava-qwen",
        task=MultimodalImageNLPTask.CONDITIONAL_GENERATION.value,
        path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        config={
            "dtype": "bfloat16",
        },
    ),
)
```

#### Run an inference

The client provide methods to run the different models on various tasks. For image-text conditional generation using Llava here is the relevant method:

```python
from pixano_inference.pydantic import TextImageConditionalGenerationRequest, TextImageConditionalGenerationResponse


request = TextImageConditionalGenerationRequest(
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
                {"type": "text", "text": "What is displayed in this image ? Answer concisely. "},
            ],
            "role": "user",
        }
    ],
    images=None,
    max_new_tokens=100,
)

response: TextImageConditionalGenerationResponse = await client.text_image_conditional_generation(request)
print(response.data.generated_text)
```
