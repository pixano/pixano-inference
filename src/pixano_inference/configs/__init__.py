# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Typed configuration objects for model deployment.

This module provides Pydantic-based config classes that validate model
parameters, task strings, and deployment settings at creation time.

Example:
    .. code-block:: python

        from pixano_inference.configs import ModelConfig, Sam2ImageParams

        config = ModelConfig(
            name="sam2-image",
            task="image_mask_generation",
            model_class="Sam2ImageModel",
            model_params=Sam2ImageParams(path="facebook/sam2-hiera-base-plus"),
        )
"""

# ruff: noqa: F401

from .base import (
    BaseModelParams,
    DeploymentConfig,
    ModelConfig,
    ModelParamsRegistry,
    ServerConfig,
    register_model_params,
)
from .sam2 import Sam2ImageParams, Sam2VideoParams
from .transformers import GroundingDINOParams, TransformersVLMParams
from .vllm import VLLMVLMParams
