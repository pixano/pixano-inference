# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Assemble and mount the /v1 API onto a FastAPI app."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI

from .admin import build_admin_router
from .inference import build_inference_router
from .jobs import build_jobs_router
from .service import build_service_router


if TYPE_CHECKING:
    from pixano_inference.ray.app import DeploymentManager


def register_v1_api(app: FastAPI, deployment_manager: DeploymentManager, auth_dependency: Any | None = None) -> None:
    """Mount the /v1 API on *app*.

    Health/readiness/info are unauthenticated (probes); inference, jobs, and admin routes
    require the API key when one is configured. A top-level ``/health`` alias is also mounted
    for container health checks.

    Args:
        app: FastAPI application.
        deployment_manager: The deployment manager instance.
        auth_dependency: Optional auth dependency applied to the guarded routers.
    """
    guarded = [Depends(auth_dependency)] if auth_dependency is not None else []

    service_router = build_service_router(deployment_manager)
    app.include_router(service_router, prefix="/v1")
    # Unversioned aliases for container health checks / load balancers.
    app.include_router(build_service_router(deployment_manager))

    app.include_router(build_inference_router(deployment_manager), prefix="/v1", dependencies=guarded)
    app.include_router(build_jobs_router(deployment_manager), prefix="/v1", dependencies=guarded)
    app.include_router(build_admin_router(deployment_manager), prefix="/v1", dependencies=guarded)
