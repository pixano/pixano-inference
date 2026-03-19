# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Inference server entry point for Ray Serve."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ray
import uvicorn
from ray import serve

from .app import create_ray_serve_app
from .config import RayServeConfig
from .config_loader import ConfigLoader
from .utils import build_runtime_env


if TYPE_CHECKING:
    from pixano_inference.configs.base import ModelConfig

    from .app import DeploymentManager


logger = logging.getLogger(__name__)


class InferenceServer:
    """Server for managing Ray Serve deployments.

    The FastAPI application runs via uvicorn in-process while model
    deployments run as separate Ray Serve actors. This avoids serialization
    issues with Pydantic models while keeping Ray's GPU management.

    Example:
        ```python
        from pixano_inference.ray import InferenceServer, RayServeConfig

        config = RayServeConfig(host="0.0.0.0", port=7463, num_gpus=2)
        server = InferenceServer(config)
        server.start(blocking=True)
        ```

        Using a Python config file:

        ```python
        server = InferenceServer()
        server.register_from_config("models.py")
        server.start(blocking=True)
        ```
    """

    def __init__(self, config: RayServeConfig | None = None) -> None:
        """Initialize the inference server.

        Args:
            config: Ray Serve configuration. If None, uses defaults.
        """
        self._config = config or RayServeConfig()
        self._running = False
        self._deployment_manager: DeploymentManager | None = None

    @property
    def config(self) -> RayServeConfig:
        """Server configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Whether the server is running."""
        return self._running

    def register_models(self, models: list[ModelConfig]) -> None:
        """Register typed model configs for deployment at startup.

        Each ``ModelConfig`` is validated and converted to the internal
        ``ModelDeploymentConfig`` format.

        Args:
            models: List of typed model configurations.
        """
        deployment_configs = [m.to_deployment_config() for m in models]
        self._config.models.extend(deployment_configs)
        logger.info(f"Added {len(models)} models programmatically")

    def register_from_config(self, config_path: str | Path) -> None:
        """Load config file and add models to startup list.

        Supports Python (``.py``) config files.

        Args:
            config_path: Path to the configuration file (.py).
        """
        loader = ConfigLoader(config_path)
        models = loader.load()
        self._config.models.extend(models)
        logger.info(f"Added {len(models)} models from config: {config_path}")

    def start(
        self,
        host: str | None = None,
        port: int | None = None,
        blocking: bool = True,
    ) -> None:
        """Start the inference server.

        This method:
        1. Initializes Ray for model deployment actors
        2. Creates the FastAPI application with DeploymentManager
        3. Deploys startup models from config via Ray Serve
        4. Runs the FastAPI app via uvicorn

        Args:
            host: Host to bind to. Uses config value if not specified.
            port: Port to serve on. Uses config value if not specified.
            blocking: Whether to block until server is stopped.
        """
        host = host or self._config.host
        port = port or self._config.port

        # Build runtime environment
        runtime_env = build_runtime_env(
            pip_packages=self._config.pip_packages,
            working_dir=self._config.working_dir,
            auto_detect=False,
        )

        # Initialize Ray if not already running
        if not ray.is_initialized():
            init_kwargs: dict[str, Any] = {}
            if runtime_env:
                init_kwargs["runtime_env"] = runtime_env
            if self._config.num_cpus is not None:
                init_kwargs["num_cpus"] = self._config.num_cpus
            if self._config.num_gpus is not None:
                init_kwargs["num_gpus"] = self._config.num_gpus

            ray.init(**init_kwargs)
            logger.info(f"Ray initialized with runtime_env: {runtime_env}")

        # Create FastAPI app and deployment manager
        fastapi_app, deployment_manager = create_ray_serve_app(self._config)
        self._deployment_manager = deployment_manager

        # Deploy startup models
        for model_config in self._config.models:
            try:
                deployment_manager.deploy_model(model_config)
                logger.info(f"Startup model '{model_config.name}' deployed")
            except Exception as e:
                logger.error(f"Failed to deploy startup model '{model_config.name}': {e}")

        self._running = True
        logger.info(f"Inference server starting on {host}:{port}")

        # Run FastAPI via uvicorn (model deployments run as Ray actors)
        if blocking:
            uvicorn.run(fastapi_app, host=host, port=port)
        else:
            uvicorn_config = uvicorn.Config(fastapi_app, host=host, port=port)
            self._uvicorn_server = uvicorn.Server(uvicorn_config)
            thread = threading.Thread(target=self._uvicorn_server.run, daemon=True)
            thread.start()

    def stop(self) -> None:
        """Stop the inference server."""
        try:
            if hasattr(self, "_uvicorn_server"):
                self._uvicorn_server.should_exit = True

            if ray.is_initialized():
                try:
                    serve.shutdown()
                except Exception:
                    pass

            self._running = False
            logger.info("Inference server stopped")
        except Exception as e:
            logger.error(f"Error stopping server: {e}")

    def __enter__(self) -> InferenceServer:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._running:
            self.stop()
