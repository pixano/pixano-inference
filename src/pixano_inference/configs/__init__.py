# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Typed configuration objects for model deployment.

This module provides Pydantic-based config classes that validate model
parameters, model capabilities, and deployment settings at creation time.

Example:
    .. code-block:: python

        from pixano_inference.configs import ModelConfig, DeploymentConfig

        config = ModelConfig(
            name="my-detector",
            model_class="MyDetector",
            model_params={"path": "/models/weights.pt"},
            deployment=DeploymentConfig(num_gpus=1),
        )

Model-specific params (e.g. ``Sam2ImageParams``) are provided by the corresponding model
package (``pixano-inference-sam``) and registered when it is imported.
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
