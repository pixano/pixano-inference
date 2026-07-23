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
from typing import TYPE_CHECKING

import uvicorn

from .app import create_ray_serve_app
from .config import RayServeConfig
from .config_loader import ConfigLoader


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

        Builds the FastAPI application and runs it via uvicorn. Ray + Serve startup,
        deployment of the configured models, and graceful drain on shutdown are all handled
        by the app's lifespan (so SIGTERM triggers a clean ``serve.shutdown`` +
        ``ray.shutdown`` after in-flight requests drain).

        Args:
            host: Host to bind to. Uses config value if not specified.
            port: Port to serve on. Uses config value if not specified.
            blocking: Whether to block until the server is stopped.
        """
        host = host or self._config.host
        port = port or self._config.port

        fastapi_app, deployment_manager = create_ray_serve_app(self._config)
        self._deployment_manager = deployment_manager
        self._running = True
        logger.info(f"Inference server starting on {host}:{port}")

        # Bound the graceful-shutdown drain so it (plus serve/ray teardown in the lifespan)
        # completes before an orchestrator's stop grace period elapses and sends SIGKILL.
        if blocking:
            uvicorn.run(fastapi_app, host=host, port=port, timeout_graceful_shutdown=self._config.graceful_shutdown_s)
        else:
            uvicorn_config = uvicorn.Config(
                fastapi_app, host=host, port=port, timeout_graceful_shutdown=self._config.graceful_shutdown_s
            )
            self._uvicorn_server = uvicorn.Server(uvicorn_config)
            thread = threading.Thread(target=self._uvicorn_server.run, daemon=True)
            thread.start()

    def stop(self) -> None:
        """Stop the inference server.

        For a non-blocking server, signals uvicorn to exit; the app lifespan then drains
        Serve and Ray. In blocking mode this is driven by SIGTERM/SIGINT instead.
        """
        try:
            if hasattr(self, "_uvicorn_server"):
                self._uvicorn_server.should_exit = True
            self._running = False
            logger.info("Inference server stopping")
        except Exception as e:
            logger.error(f"Error stopping server: {e}")

    def __enter__(self) -> InferenceServer:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._running:
            self.stop()
