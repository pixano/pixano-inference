# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""LLM (Large Language Model) base class and I/O types (stub)."""

from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from .base import InferenceModel
from .vlm import UsageInfo


class LLMInput(BaseModel):
    """Input for LLM text generation.

    Attributes:
        prompt: Text prompt or chat messages.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
    """

    prompt: str | list[dict[str, Any]]
    max_new_tokens: int = 256
    temperature: float = 1.0


class LLMOutput(BaseModel):
    """Output for LLM text generation.

    Attributes:
        generated_text: Generated text.
        usage: Usage metadata.
        generation_config: Configuration used for the generation.
    """

    generated_text: str
    usage: UsageInfo
    generation_config: dict[str, Any] = {}


class LLMModel(InferenceModel):
    """Base class for large language models (stub)."""

    @abstractmethod
    def predict(self, input: LLMInput) -> LLMOutput:
        """Run text generation.

        Args:
            input: LLM input with prompt and generation parameters.

        Returns:
            LLM output with generated text, usage info, and generation config.
        """
