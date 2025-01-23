# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for text-image conditional generation task."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pixano_inference.pydantic.base import BaseRequest, BaseResponse


class TextImageConditionalGenerationInput(BaseModel):
    """Input for text-image conditional generation.

    Attributes:
        prompt: Prompt for the generation.
        image: Image for the generation.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Temperature for the generation.
    """

    prompt: str
    image: str | Path
    max_new_tokens: int
    temperature: float = 1.0


class TextImageConditionalGenerationRequest(BaseRequest, TextImageConditionalGenerationInput):
    """Request for text-image conditional generation."""

    def to_input(self) -> TextImageConditionalGenerationInput:
        """Convert the request to the input."""
        return TextImageConditionalGenerationInput(
            prompt=self.prompt,
            image=self.image,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )


class UsageConditionalGeneration(BaseModel):
    """Usage metadata of the model for text-image conditional generation.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total number of tokens.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TextImageConditionalGenerationOutput(BaseModel):
    """Output for text-image conditional generation.

    Attributes:
        generated_text: Generated text.
        usage: Usage of the model for the generation.
        generation_config: Configuration for the generation.
    """

    generated_text: str
    usage: UsageConditionalGeneration
    generation_config: dict[str, Any] = {}


class TextImageConditionalGenerationResponse(BaseResponse):
    """Response for text-image conditional generation.

    Attributes:
        data: Output of the generation.
    """

    data: TextImageConditionalGenerationOutput
