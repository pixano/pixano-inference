# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Ray Serve configuration models.

``ResourceConfig``, ``AutoscalingConfig`` and ``ModelDeploymentConfig`` live in
:mod:`pixano_inference.configs.deployment`; they are re-exported here so
``from pixano_inference.ray.config import ModelDeploymentConfig`` keeps working.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pixano_inference.configs.deployment import (  # noqa: F401
    AutoscalingConfig,
    ModelDeploymentConfig,
    ResourceConfig,
)


class RayServeConfig(BaseModel):
    """Top-level Ray Serve configuration.

    Attributes:
        host: Host to bind to. Defaults to loopback; bind 0.0.0.0 explicitly to expose the
            server (do so only with API-key auth enabled — see pixano_inference.security).
        port: Port to serve on.
        num_cpus: Total number of CPUs available to Ray. None means auto-detect.
        num_gpus: Total number of GPUs available to Ray. None means auto-detect.
        pip_packages: List of pip packages to install in Ray workers runtime environment.
        working_dir: Working directory for Ray workers.
        models: List of models to deploy at startup.
        default_resources: Default resource configuration for deployments.
        default_autoscaling: Default autoscaling configuration for deployments.
        serve_start_timeout_s: How long to wait for Ray Serve's controller to come up before
            aborting startup. Ray's own ``serve.start()`` waits forever, so this is the only
            bound on a cluster that can never schedule the controller actor.
        deploy_timeout_s: How long to wait for a model's Serve application to reach RUNNING.
    """

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=7463)
    num_cpus: int | None = Field(default=None)
    num_gpus: int | None = Field(default=None)
    pip_packages: list[str] | None = Field(default=None)
    working_dir: str | None = Field(default=None)
    models: list[ModelDeploymentConfig] = Field(default_factory=list)
    default_resources: ResourceConfig = Field(default_factory=ResourceConfig)
    default_autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
    strict_startup: bool = Field(default=True)
    ray_address: str | None = Field(default=None)
    ray_namespace: str = Field(default="pixano-inference")
    graceful_shutdown_s: float = Field(default=30.0, gt=0)
    serve_start_timeout_s: float = Field(default=120.0, gt=0)
    deploy_timeout_s: float = Field(default=600.0, gt=0)
