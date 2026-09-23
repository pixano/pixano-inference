# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Typed model params for ``GroundingDINOModel``."""

from pydantic import Field

from pixano_inference.configs.base import BaseModelParams, register_model_params


@register_model_params("GroundingDINOModel")
class GroundingDINOParams(BaseModelParams):
    """Typed parameters for ``GroundingDINOModel``.

    Attributes:
        path: HuggingFace model ID or local checkpoint path.
        processor_config: Kwargs for ``AutoProcessor.from_pretrained``.
        config: Kwargs for ``AutoModelForZeroShotObjectDetection.from_pretrained``.
    """

    path: str
    processor_config: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
