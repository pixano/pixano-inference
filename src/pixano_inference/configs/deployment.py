# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deployment configuration of a single model: the object every model is constructed with.

These classes are plain pydantic models. They belong to the model API, not to the Ray Serve
runtime, so a model package imports them without loading the server stack.
"""

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

    min_replicas: int = Field(default=1, ge=0)
    max_replicas: int = Field(default=4, ge=1)
    target_num_ongoing_requests_per_replica: int = Field(default=2, ge=1)
    downscale_delay_s: float = Field(default=60.0, gt=0)
    upscale_delay_s: float = Field(default=5.0, gt=0)


class ModelDeploymentConfig(BaseModel):
    """Configuration for deploying a single model.

    Attributes:
        name: Unique model name. Optional for HuggingFace models (auto-derived from path).
        capability: Capability string (e.g. "segmentation").
        model_class: Registered class name (e.g. "MyDetector"), provided by an installed model package.
        model_params: Parameters passed to model __init__ via config.
        resources: Resource configuration for the deployment.
        autoscaling: Autoscaling configuration for the deployment.
        max_batch_size: Maximum batch size for inference (1 disables batching).
        batch_wait_timeout_s: Timeout for waiting to fill batch.
        max_ongoing_requests: Max concurrent requests Serve routes to one replica before
            queueing / triggering autoscaling.
        health_check_period_s: How often Serve calls the replica health check.
        timeout_s: Per-request inference timeout. None uses the capability default.
    """

    name: str
    capability: str
    model_class: str
    model_params: dict = Field(default_factory=dict)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
    max_batch_size: int = Field(default=1, ge=1)
    batch_wait_timeout_s: float = Field(default=0.1, ge=0)
    max_ongoing_requests: int = Field(default=2, ge=1)
    health_check_period_s: float = Field(default=10.0, gt=0)
    timeout_s: float | None = Field(default=None, gt=0)
