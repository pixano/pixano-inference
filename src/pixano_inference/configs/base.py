# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Core typed config classes and model params registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from pixano_inference.models import ModelClassRegistry, infer_http_capability

from ..ray.config import AutoscalingConfig, ModelDeploymentConfig, RayServeConfig, ResourceConfig


class ModelParamsRegistry:
    """Registry mapping model class names to their params Pydantic schemas.

    This lightweight registry allows validation of ``model_params`` without
    importing heavy ML dependencies.
    """

    _registry: dict[str, type[BaseModelParams]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[BaseModelParams]], type[BaseModelParams]]:
        """Decorator to register a params schema for a model class.

        Args:
            name: The model class name (e.g. ``"Sam2ImageModel"``).

        Returns:
            The decorator function.
        """

        def decorator(params_cls: type[BaseModelParams]) -> type[BaseModelParams]:
            if name in cls._registry:
                raise ValueError(f"Model params schema '{name}' already registered.")
            cls._registry[name] = params_cls
            return params_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseModelParams] | None:
        """Get the params schema for a model class, or None if not registered.

        Args:
            name: The model class name.

        Returns:
            The params Pydantic model class, or None.
        """
        return cls._registry.get(name)

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a params schema is registered.

        Args:
            name: The model class name.

        Returns:
            True if registered.
        """
        return name in cls._registry

    @classmethod
    def list_all(cls) -> dict[str, type[BaseModelParams]]:
        """List all registered params schemas.

        Returns:
            Dictionary of model class name to params schema.
        """
        return cls._registry.copy()


def register_model_params(name: str) -> Callable[[type[BaseModelParams]], type[BaseModelParams]]:
    """Convenience decorator for registering model params schemas.

    Args:
        name: The model class name to register under.

    Returns:
        The decorator function.
    """
    return ModelParamsRegistry.register(name)


class BaseModelParams(BaseModel):
    """Base class for typed model parameters.

    All models require at least a ``path`` (HuggingFace model ID or local checkpoint).
    Extra fields are forbidden so that typos are caught at config-creation time.

    Attributes:
        path: HuggingFace model ID or local checkpoint path.
    """

    model_config = {"extra": "forbid"}

    path: str


class DeploymentConfig(BaseModel):
    """User-facing deployment settings.

    Merges resource, autoscaling, and batch settings into a single flat class
    for convenience in Python config files.

    Attributes:
        num_gpus: Number of GPUs per replica.
        num_cpus: Number of CPUs per replica.
        memory_mb: Memory limit in MB.
        min_replicas: Minimum number of replicas (0 for scale-to-zero).
        max_replicas: Maximum number of replicas.
        target_num_ongoing_requests_per_replica: Target ongoing requests per replica before scaling up.
        downscale_delay_s: Delay in seconds before scaling down.
        upscale_delay_s: Delay in seconds before scaling up.
        max_batch_size: Maximum batch size for inference.
        batch_wait_timeout_s: Timeout for waiting to fill batch.
    """

    num_gpus: float = Field(default=0.0, ge=0)
    num_cpus: float = Field(default=1.0, ge=0)
    memory_mb: int | None = Field(default=None, ge=0)
    min_replicas: int = Field(default=0, ge=0)
    max_replicas: int = Field(default=4, ge=1)
    target_num_ongoing_requests_per_replica: int = Field(default=2, ge=1)
    downscale_delay_s: float = Field(default=60.0, gt=0)
    upscale_delay_s: float = Field(default=5.0, gt=0)
    max_batch_size: int = Field(default=8, ge=1)
    batch_wait_timeout_s: float = Field(default=0.1, ge=0)


class ModelConfig(BaseModel):
    """Full model configuration with typed validation.

    This class resolves the model capability from the configured model class and
    resolves ``model_params`` through the ``ModelParamsRegistry`` when a schema is
    registered for the given ``model_class``.

    Accepts both strings and typed Python objects so config files can keep IDE
    support while still allowing concise string-based declarations.

    Attributes:
        name: Unique model name. Optional for HuggingFace models (auto-derived from path).
        model_class: Registered model class name or class type (e.g. ``"Sam2ImageModel"``
            or ``Sam2ImageModel``).
        model_params: Typed params or raw dict, auto-resolved via registry.
        deployment: Deployment settings.
    """

    name: str | None = None
    model_class: str | type
    model_params: dict[str, Any] | BaseModelParams = Field(default_factory=dict)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

    model_config = {"arbitrary_types_allowed": True}
    _resolved_model_class: type | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _resolve_model_params(cls, data: Any) -> Any:
        """Resolve model_params through the registry when possible."""
        if not isinstance(data, dict):
            return data
        model_class = data.get("model_class")
        raw_params = data.get("model_params", {})
        # Extract string name for registry lookup (model_class may be a type)
        class_name = model_class.__name__ if isinstance(model_class, type) else model_class
        if isinstance(raw_params, dict) and class_name is not None:
            params_cls = ModelParamsRegistry.get(class_name)
            if params_cls is not None:
                data["model_params"] = params_cls(**raw_params)
        return data

    def model_post_init(self, __context: Any) -> None:
        """Resolve and validate the configured model class."""
        self._resolved_model_class = self._resolve_model_class()
        infer_http_capability(self._resolved_model_class)

    @property
    def capability(self) -> str:
        """Capability derived from the configured model class."""
        if self._resolved_model_class is None:
            self._resolved_model_class = self._resolve_model_class()
        return infer_http_capability(self._resolved_model_class)

    @property
    def resolved_name(self) -> str:
        """Resolved deployment name for the configured model."""
        if self.name is not None:
            return self.name

        raw_path: Any | None = None
        if isinstance(self.model_params, BaseModelParams):
            raw_path = self.model_params.path
        elif isinstance(self.model_params, dict):
            raw_path = self.model_params.get("path")

        if isinstance(raw_path, str):
            normalized_path = raw_path.rstrip("/")
            if normalized_path:
                return normalized_path.split("/")[-1]

        return self.model_class_name

    @property
    def model_class_name(self) -> str:
        """Registered name of the configured model class."""
        if isinstance(self.model_class, type):
            return self.model_class.__name__
        return self.model_class

    def _resolve_model_class(self) -> type:
        """Resolve ``model_class`` to a Python type and validate support."""
        import pixano_inference.impls  # noqa: F401
        from pixano_inference.models.base import InferenceModel

        if isinstance(self.model_class, type):
            if not issubclass(self.model_class, InferenceModel):
                raise ValueError(
                    f"Configured model_class '{self.model_class.__name__}' must inherit from InferenceModel."
                )
            ModelClassRegistry.ensure_registered(self.model_class)
            return self.model_class

        try:
            return ModelClassRegistry.get(self.model_class)
        except KeyError:
            pass

        raise ValueError(
            f"Unknown model_class '{self.model_class}'. Register the class with @register_model "
            "or pass the class type directly."
        )

    def to_deployment_config(self) -> ModelDeploymentConfig:
        """Convert to the internal ``ModelDeploymentConfig`` used by Ray Serve.

        Returns:
            A ``ModelDeploymentConfig`` instance.
        """
        dep = self.deployment
        if isinstance(self.model_params, BaseModel):
            model_params = self.model_params.model_dump()
        else:
            model_params = self.model_params
        return ModelDeploymentConfig(
            name=self.resolved_name,
            capability=self.capability,
            model_class=self.model_class_name,
            model_params=model_params,
            resources=ResourceConfig(
                num_gpus=dep.num_gpus,
                num_cpus=dep.num_cpus,
                memory_mb=dep.memory_mb,
            ),
            autoscaling=AutoscalingConfig(
                min_replicas=dep.min_replicas,
                max_replicas=dep.max_replicas,
                target_num_ongoing_requests_per_replica=dep.target_num_ongoing_requests_per_replica,
                downscale_delay_s=dep.downscale_delay_s,
                upscale_delay_s=dep.upscale_delay_s,
            ),
            max_batch_size=dep.max_batch_size,
            batch_wait_timeout_s=dep.batch_wait_timeout_s,
        )


class ServerConfig(BaseModel):
    """Top-level typed server configuration.

    Attributes:
        host: Host to bind to.
        port: Port to serve on.
        num_cpus: Total CPUs available to Ray. None means auto-detect.
        num_gpus: Total GPUs available to Ray. None means auto-detect.
        pip_packages: Pip packages for Ray workers runtime environment.
        working_dir: Working directory for Ray workers.
        models: List of model configurations.
    """

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7463)
    num_cpus: int | None = Field(default=None)
    num_gpus: int | None = Field(default=None)
    pip_packages: list[str] | None = Field(default=None)
    working_dir: str | None = Field(default=None)
    models: list[ModelConfig] = Field(default_factory=list)

    def to_ray_serve_config(self) -> RayServeConfig:
        """Convert to the internal ``RayServeConfig``.

        Returns:
            A ``RayServeConfig`` instance with all models converted.
        """
        return RayServeConfig(
            host=self.host,
            port=self.port,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            pip_packages=self.pip_packages,
            working_dir=self.working_dir,
            models=[m.to_deployment_config() for m in self.models],
        )
