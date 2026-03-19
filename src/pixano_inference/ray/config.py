# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Ray Serve configuration models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResourceConfig(BaseModel):
    """Resource configuration for a model deployment.

    Attributes:
        num_gpus: Number of GPUs per replica.
        num_cpus: Number of CPUs per replica.
        memory_mb: Memory limit in MB. None means no limit.
    """

    num_gpus: float = Field(default=0.0, ge=0)
    num_cpus: float = Field(default=1.0, ge=0)
    memory_mb: int | None = Field(default=None, ge=0)


class AutoscalingConfig(BaseModel):
    """Autoscaling configuration for Ray Serve deployments.

    Attributes:
        min_replicas: Minimum number of replicas. Can be 0 for scale-to-zero.
        max_replicas: Maximum number of replicas.
        target_num_ongoing_requests_per_replica: Target number of ongoing requests
            per replica before scaling up.
        downscale_delay_s: Delay in seconds before scaling down.
        upscale_delay_s: Delay in seconds before scaling up.
    """

    min_replicas: int = Field(default=0, ge=0)
    max_replicas: int = Field(default=4, ge=1)
    target_num_ongoing_requests_per_replica: int = Field(default=2, ge=1)
    downscale_delay_s: float = Field(default=60.0, gt=0)
    upscale_delay_s: float = Field(default=5.0, gt=0)


class ModelDeploymentConfig(BaseModel):
    """Configuration for deploying a single model.

    Attributes:
        name: Unique model name. Optional for HuggingFace models (auto-derived from path).
        capability: Capability string (e.g. "segmentation").
        model_class: Registered class name (e.g. "Sam2ImageModel").
        model_module: Python module path to import before resolving model_class
            (e.g. "my_package.models"). Used for external custom models.
        model_params: Parameters passed to model __init__ via config.
        resources: Resource configuration for the deployment.
        autoscaling: Autoscaling configuration for the deployment.
        max_batch_size: Maximum batch size for inference.
        batch_wait_timeout_s: Timeout for waiting to fill batch.
    """

    name: str
    capability: str
    model_class: str
    model_module: str | None = Field(
        default=None,
        description="Python module path to import before resolving model_class (e.g. 'my_package.models').",
    )
    model_params: dict = Field(default_factory=dict)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
    max_batch_size: int = Field(default=8, ge=1)
    batch_wait_timeout_s: float = Field(default=0.1, ge=0)


class RayServeConfig(BaseModel):
    """Top-level Ray Serve configuration.

    Attributes:
        host: Host to bind to.
        port: Port to serve on.
        num_cpus: Total number of CPUs available to Ray. None means auto-detect.
        num_gpus: Total number of GPUs available to Ray. None means auto-detect.
        pip_packages: List of pip packages to install in Ray workers runtime environment.
        working_dir: Working directory for Ray workers.
        models: List of models to deploy at startup.
        default_resources: Default resource configuration for deployments.
        default_autoscaling: Default autoscaling configuration for deployments.
    """

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7463)
    num_cpus: int | None = Field(default=None)
    num_gpus: int | None = Field(default=None)
    pip_packages: list[str] | None = Field(default=None)
    working_dir: str | None = Field(default=None)
    models: list[ModelDeploymentConfig] = Field(default_factory=list)
    default_resources: ResourceConfig = Field(default_factory=ResourceConfig)
    default_autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
