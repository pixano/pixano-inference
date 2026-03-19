# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""VLM (Vision-Language Model) base class and I/O types."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from .base import InferenceModel


class UsageInfo(BaseModel):
    """Usage metadata for generation.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total number of tokens.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class VLMInput(BaseModel):
    """Input for vision-language model generation.

    Attributes:
        prompt: Prompt for the generation. Can be a string or a list of dicts for chat templates.
        images: Images for the generation. Can be None if images are passed in the prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Temperature for the generation.
    """

    prompt: str | list[dict[str, Any]]
    images: list[str | Path] | None = None
    max_new_tokens: int
    temperature: float = 1.0


class VLMOutput(BaseModel):
    """Output for vision-language model generation.

    Attributes:
        generated_text: Generated text.
        usage: Usage metadata.
        generation_config: Configuration used for the generation.
    """

    generated_text: str
    usage: UsageInfo
    generation_config: dict[str, Any] = {}


class VLMModel(InferenceModel):
    """Base class for vision-language models.

    Example:
        ```python
        @register_model("my-vlm")
        class MyVLM(VLMModel):
            def load_model(self):
                self.model = load_weights(self.config.model_params["path"])

            def predict(self, input: VLMInput) -> VLMOutput:
                text = self.model.generate(input.prompt, input.images)
                return VLMOutput(generated_text=text, usage=..., generation_config=...)
        ```
    """

    capability_name: ClassVar[str] = "vlm"

    @abstractmethod
    def predict(self, input: VLMInput) -> VLMOutput:
        """Run vision-language generation.

        Args:
            input: VLM input with prompt, images, and generation parameters.

        Returns:
            VLM output with generated text, usage info, and generation config.
        """
