# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Service endpoints: health, readiness, settings, model listing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pixano_inference.__version__ import __version__
from pixano_inference.schemas import ModelInfo


if TYPE_CHECKING:
    from pixano_inference.ray.app import DeploymentManager


def register_service_routes(app: FastAPI, deployment_manager: DeploymentManager) -> None:
    """Register health, readiness, settings, and model listing endpoints.

    Args:
        app: FastAPI application.
        deployment_manager: The deployment manager instance.
    """

    @app.get("/health")
    async def health():
        """Liveness check: the ingress process is up (cheap, always 200)."""
        return {"status": "healthy"}

    @app.get("/ready")
    async def ready():
        """Readiness check: 503 unless every configured model is RUNNING in Serve."""
        result = deployment_manager.readiness()
        if not result["ready"]:
            return JSONResponse(status_code=503, content=result)
        return result

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Pixano Inference API (Ray Serve)",
            "version": __version__,
            "docs": "/docs",
        }

    @app.get("/app/settings/")
    async def get_settings():
        """Get application settings and live cluster/model status."""
        gpu_info = deployment_manager.get_gpu_info()
        models = deployment_manager.list_models()
        return {
            "app_name": "Pixano Inference",
            "app_version": __version__,
            "app_description": "Pixano Inference API powered by Ray Serve",
            "num_cpus": deployment_manager.config.num_cpus,
            "num_gpus": gpu_info.get("num_gpus", 0),
            "num_nodes": deployment_manager.num_nodes(),
            "gpus_used": gpu_info.get("gpus_used", 0.0),
            "gpu_to_model": {},
            "models": [m.name for m in models],
            "models_to_capability": {m.name: m.capability for m in models},
            "models_status": deployment_manager.model_statuses(),
        }

    @app.get("/app/models/")
    async def list_models() -> list[ModelInfo]:
        """List all deployed models."""
        return deployment_manager.list_models()
