# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Typed model params for vLLM-based models."""

from pydantic import Field

from .base import BaseModelParams, register_model_params


@register_model_params("VLLMVLMModel")
class VLLMVLMParams(BaseModelParams):
    """Typed parameters for ``VLLMVLMModel``.

    Attributes:
        path: HuggingFace model ID or local checkpoint path.
        config: Kwargs for ``vllm.LLM``.
        processor_config: Kwargs for ``vllm.LLM`` processor options.
    """

    path: str
    config: dict = Field(default_factory=dict)
    processor_config: dict = Field(default_factory=dict)
