# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""DeploymentManager and FastAPI app factory for Ray Serve."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import ray
from fastapi import FastAPI

from pixano_inference.__version__ import __version__
from pixano_inference.models.registry import ModelClassRegistry
from pixano_inference.schemas import ModelInfo

from .config import ModelDeploymentConfig, RayServeConfig
from .deployment import create_model_deployment
from .routes import register_inference_routes, register_service_routes


logger = logging.getLogger(__name__)


class DeploymentManager:
    """In-process manager for deployed models, handles, and GPU allocation.

    Manages the lifecycle of Ray Serve model deployments including:
    - Resolving model classes from the registry
    - Creating and running deployments
    - Storing deployment handles for inference
    - Tracking GPU allocations
    """

    def __init__(self, config: RayServeConfig) -> None:
        """Initialize the deployment manager.

        Args:
            config: Ray Serve configuration.
        """
        self._config = config
        self._handles: dict[str, Any] = {}  # model_name -> DeploymentHandle
        self._configs: dict[str, ModelDeploymentConfig] = {}  # model_name -> config
        self._metadata_cache: dict[str, dict[str, Any]] = {}  # model_name -> metadata

    @property
    def config(self) -> RayServeConfig:
        """Server configuration."""
        return self._config

    def deploy_model(self, config: ModelDeploymentConfig) -> None:
        """Deploy a model.

        Resolves the model class from the registry, creates a Ray actor,
        and stores the handle.

        Args:
            config: Model deployment configuration.

        Raises:
            ValueError: If the model is already deployed.
            KeyError: If the model class is not found in the registry.
        """
        if config.name in self._handles:
            raise ValueError(f"Model '{config.name}' is already deployed.")

        # Import external model module to trigger @register_model decorators
        if config.model_module is not None:
            try:
                importlib.import_module(config.model_module)
                logger.info(f"Imported model module: {config.model_module}")
            except ImportError as e:
                raise ImportError(
                    f"Failed to import model module '{config.model_module}' for model '{config.name}': {e}"
                ) from e

        # Resolve model class
        model_class = ModelClassRegistry.get(config.model_class)

        # Create Ray actor
        handle = create_model_deployment(model_class, config)

        self._handles[config.name] = handle
        self._configs[config.name] = config
        logger.info(f"Deployed model '{config.name}' (class={config.model_class}, capability={config.capability})")

    def undeploy_model(self, name: str) -> None:
        """Undeploy a model.

        Kills the Ray actor, frees GPU, and removes the handle.

        Args:
            name: Model name.

        Raises:
            ValueError: If the model is not deployed.
        """
        if name not in self._handles:
            raise ValueError(f"Model '{name}' is not deployed.")

        handle = self._handles[name]
        try:
            ray.kill(handle)
        except Exception as e:
            logger.warning(f"Error killing actor for '{name}': {e}")

        del self._handles[name]
        del self._configs[name]
        self._metadata_cache.pop(name, None)
        logger.info(f"Undeployed model '{name}'")

    def get_handle(self, name: str) -> Any | None:
        """Get a deployment handle by model name.

        Args:
            name: Model name.

        Returns:
            DeploymentHandle or None if not found.
        """
        return self._handles.get(name)

    def get_model_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a deployed model.

        Args:
            name: Model name.

        Returns:
            Model metadata dictionary.
        """
        if name in self._metadata_cache:
            return self._metadata_cache[name]

        config = self._configs.get(name)
        if config is None:
            return {}

        metadata = {
            "model_name": config.name,
            "capability": config.capability,
            "model_class": config.model_class,
        }

        self._metadata_cache[name] = metadata
        return metadata

    def list_models(self) -> list[ModelInfo]:
        """List all deployed models.

        Returns:
            List of ModelInfo objects.
        """
        return [
            ModelInfo(
                name=config.name,
                capability=config.capability,
                model_path=config.model_params.get("path")
                if isinstance(config.model_params.get("path"), str)
                else None,
                model_class=config.model_class,
            )
            for config in self._configs.values()
        ]

    def get_model_capability(self, name: str) -> str | None:
        """Get the deployed capability for a model."""
        config = self._configs.get(name)
        if config is None:
            return None
        return config.capability

    def get_gpu_info(self) -> dict[str, Any]:
        """Get GPU resource information from Ray.

        Returns:
            GPU info dictionary.
        """
        if not ray.is_initialized():
            return {"num_gpus": 0, "available_gpus": [], "gpus_used": 0}

        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()

        total_gpus = float(cluster_resources.get("GPU", 0.0))
        available_gpus = float(available_resources.get("GPU", 0.0))
        used_gpus = max(0.0, total_gpus - available_gpus)

        return {
            "num_gpus": int(total_gpus),
            "available_gpus": available_gpus,
            "gpus_used": used_gpus,
        }


def create_ray_serve_app(
    config: RayServeConfig | None = None,
) -> tuple[FastAPI, DeploymentManager]:
    """Create the FastAPI application and DeploymentManager for Ray Serve.

    Args:
        config: Ray Serve configuration. If None, uses defaults.

    Returns:
        Tuple of (FastAPI app, DeploymentManager).
    """
    if config is None:
        config = RayServeConfig()

    # Trigger model registration from installed backends
    import pixano_inference.impls  # noqa: F401

    app = FastAPI(
        title="Pixano Inference (Ray)",
        description="Pixano Inference API powered by Ray Serve",
        version=__version__,
    )

    deployment_manager = DeploymentManager(config)

    # Register all route groups
    register_service_routes(app, deployment_manager)
    register_inference_routes(app, deployment_manager)

    # Store references in app state for access by routes
    app.state.config = config
    app.state.deployment_manager = deployment_manager

    return app, deployment_manager
