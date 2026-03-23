# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Typed model params for SAM2 models."""

from typing import Literal

from .base import BaseModelParams, register_model_params


@register_model_params("Sam2ImageModel")
class Sam2ImageParams(BaseModelParams):
    """Typed parameters for ``Sam2ImageModel``.

    Attributes:
        path: HuggingFace model ID or local checkpoint path.
        torch_dtype: Torch dtype for autocast.
        compile: Whether to ``torch.compile`` the model.
    """

    path: str = "facebook/sam2-hiera-base-plus"
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    compile: bool = True


@register_model_params("Sam2VideoModel")
class Sam2VideoParams(BaseModelParams):
    """Typed parameters for ``Sam2VideoModel``.

    Attributes:
        path: HuggingFace model ID or local checkpoint path.
        torch_dtype: Torch dtype for autocast.
        compile: Whether to ``torch.compile`` the model.
        vos_optimized: Use VOS-optimised predictor.
        propagate: Whether to propagate masks across the full video.
    """

    path: str = "facebook/sam2-hiera-large"
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    compile: bool = True
    vos_optimized: bool = True
    propagate: bool = True
