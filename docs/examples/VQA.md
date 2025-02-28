<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Visual Question Answering

## Instantiate the model

In this example, we will use the LLaVa model with Qwen 2 for its LLM component.

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

## Call the model for inference

For the VQA inference, call the `text_image_conditional_generation` method.

In this example, we use the picture of a brown bear and ask the model to provide a high level of description of the image.

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
