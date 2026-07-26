# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""VLM (Vision-Language Model) I/O types."""

from pathlib import Path
from typing import Any

from .base import CamelModel


class UsageInfo(CamelModel):
    """Usage metadata for generation.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total number of tokens.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class VLMInput(CamelModel):
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


class VLMOutput(CamelModel):
    """Output for vision-language model generation.

    Attributes:
        generated_text: Generated text.
        usage: Usage metadata.
        generation_config: Configuration used for the generation.
    """

    generated_text: str
    usage: UsageInfo
    generation_config: dict[str, Any] = {}
