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

### Run the application

Pixano-Inference can invoke a server that will serve the API. To do so, simply execute the following command:

```bash
pixano_inference --port 8000
```

The default port is `8000`. You can change it by passing the `--port` argument.

### Instantiate the client

The easiest way to interact with Pixano-Inference is through the Python client.

```python
from pixano_inference import PixanoInferenceClient


client = PixanoInferenceClient(url="http://localhost:8000")
```

### Instantiate a model

To instantiate a model, you need to provide the path of the model file and the task that the model will perform. For example to run a Llava model from the vLLM provider:

```python
from pixano_inference.pydantic import ModelConfig
from pixano_inference.tasks import MultimodalImageNLPTask


client.instantiate_model(
    provider="vllm",
    config=ModelConfig(
        name="llava",
        task=MultimodalImageNLPTask.CONDITIONAL_GENERATION.value,
        path="llava-hf/llava-v1.6-mistral-7b-hf",
        config={
            "dtype": "bfloat16",
        },
    ),
)
```

### Run an inference

The client provide methods to run the different models on various tasks. For image-text conditional generation using Llava here is the relevant method:

```python
from pixano_inference.pydantic import TextImageConditionalGenerationRequest, TextImageConditionalGenerationResponse


request = TextImageConditionalGenerationRequest(
    prompt=[
        {'content': [
            {'type': 'image_url',
             'image_url': {
                'url': 'https://upload.wikimedia.org/wikipedia/commons/9/9e/Ours_brun_parcanimalierpyrenees_1.jpg'}
            },
            {'type': 'text',
             'text': 'What is displayed in this image ? Answer concisely. '}
        ],
         'role': 'user'
        }
    ],
    image_path="/path/to/image.jpg",
)


response: TextImageConditionalGenerationResponse = client.text_image_conditional_generation(request)
print(response.data.generated_text)
```
