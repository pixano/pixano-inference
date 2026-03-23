# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Ray Serve infrastructure for Pixano Inference.

This module provides Ray Serve-based infrastructure for model serving.
It offers GPU utilization, request batching, and autoscaling capabilities.
Users can subclass model base classes to deploy custom models (JAX, PyTorch,
TensorFlow, etc.) on Ray Serve.

Example:
    Basic usage with the inference server:

    ```python
    from pixano_inference.ray import InferenceServer, RayServeConfig

    config = RayServeConfig(host="0.0.0.0", port=7463, num_gpus=2)
    server = InferenceServer(config)
    server.start(blocking=True)
    ```

    Using Python configuration:

    ```python
    from pixano_inference.ray import InferenceServer

    server = InferenceServer()
    server.register_from_config("models.py")
    server.start(blocking=True)
    ```

    Custom model deployment:

    ```python
    from pixano_inference.models import (
        SegmentationInput,
        SegmentationModel,
        SegmentationOutput,
        register_model,
    )

    @register_model("my_sam2")
    class MySam2Model(SegmentationModel):
        def load_model(self) -> None:
            ...

        def predict(self, input: SegmentationInput) -> SegmentationOutput:
            ...
    ```
"""

from .config import AutoscalingConfig, ModelDeploymentConfig, RayServeConfig, ResourceConfig
from .config_loader import ConfigLoader
from .deployment import create_model_deployment
from .server import InferenceServer
from .utils import build_runtime_env, detect_optional_packages


__all__ = [
    "RayServeConfig",
    "ModelDeploymentConfig",
    "ResourceConfig",
    "AutoscalingConfig",
    "ConfigLoader",
    "create_model_deployment",
    "InferenceServer",
    "build_runtime_env",
    "detect_optional_packages",
]
