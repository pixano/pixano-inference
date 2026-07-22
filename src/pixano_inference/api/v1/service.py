# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""/v1 service routes: health, readiness, and server info."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pixano_inference.__version__ import __version__


if TYPE_CHECKING:
    from pixano_inference.ray.app import DeploymentManager


def build_service_router(deployment_manager: DeploymentManager) -> APIRouter:
    """Build the health/readiness/info router bound to *deployment_manager*."""
    router = APIRouter(tags=["service"])

    @router.get("/health")
    async def health() -> dict[str, str]:
        """Liveness check: the ingress process is up (cheap, always 200)."""
        return {"status": "healthy"}

    @router.get("/ready")
    async def ready() -> Any:
        """Readiness check: 503 unless every configured model is RUNNING in Serve."""
        result = deployment_manager.readiness()
        if not result["ready"]:
            return JSONResponse(status_code=503, content=result)
        return result

    @router.get("/info")
    async def info() -> dict[str, Any]:
        """Server and cluster information."""
        gpu_info = deployment_manager.get_gpu_info()
        models = deployment_manager.list_models()
        return {
            "appName": "Pixano Inference",
            "appVersion": __version__,
            "numCpus": deployment_manager.config.num_cpus,
            "numGpus": gpu_info.get("num_gpus", 0),
            "numNodes": deployment_manager.num_nodes(),
            "gpusUsed": gpu_info.get("gpus_used", 0.0),
            "models": [m.name for m in models],
            "modelsStatus": deployment_manager.model_statuses(),
        }

    return router
