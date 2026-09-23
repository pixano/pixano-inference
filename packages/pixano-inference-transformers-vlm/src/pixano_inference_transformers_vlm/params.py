# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Typed model params for ``TransformersVLMModel``."""

from pydantic import Field

from pixano_inference.configs.base import BaseModelParams, register_model_params


@register_model_params("TransformersVLMModel")
class TransformersVLMParams(BaseModelParams):
    """Typed parameters for ``TransformersVLMModel``.

    Attributes:
        path: HuggingFace model ID or local checkpoint path.
        processor_config: Kwargs for ``AutoProcessor.from_pretrained``.
        config: Kwargs for model ``from_pretrained``.
        model_type: Model type hint (e.g. ``"llava"``, ``"llava-next"``).
    """

    path: str
    processor_config: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    model_type: str | None = None
