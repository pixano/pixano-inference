# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""/v1 admin routes: runtime model deploy / undeploy / list / stats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anyio import to_thread
from fastapi import APIRouter, HTTPException

from .schemas import DeployModelRequest, ModelStatusInfo


if TYPE_CHECKING:
    from pixano_inference.ray.app import DeploymentManager


def build_admin_router(deployment_manager: DeploymentManager) -> APIRouter:
    """Build the `/models` admin router bound to *deployment_manager*."""
    router = APIRouter(prefix="/models", tags=["admin"])

    @router.get("", response_model=list[ModelStatusInfo])
    async def list_models() -> Any:
        statuses = deployment_manager.model_statuses()
        return [
            ModelStatusInfo(
                name=m.name,
                capability=m.capability,
                model_class=m.model_class,
                model_path=m.model_path,
                status=statuses.get(m.name, "NOT_STARTED"),
            )
            for m in deployment_manager.list_models()
        ]

    @router.post("", status_code=201, response_model=ModelStatusInfo)
    async def deploy_model(request: DeployModelRequest) -> Any:
        # Import here to keep heavy config validation off the module import path.
        from pixano_inference.configs.base import ModelConfig

        try:
            model_config = ModelConfig(
                name=request.name,
                model_class=request.model_class,
                model_params=request.model_params,
                deployment=request.deployment,  # type: ignore[arg-type]
            ).to_deployment_config()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid model configuration: {exc}") from exc

        try:
            await to_thread.run_sync(deployment_manager.deploy_model, model_config)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown model class: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Deployment failed: {exc}") from exc

        statuses = deployment_manager.model_statuses()
        return ModelStatusInfo(
            name=model_config.name,
            capability=model_config.capability,
            model_class=model_config.model_class,
            model_path=model_config.model_params.get("path")
            if isinstance(model_config.model_params.get("path"), str)
            else None,
            status=statuses.get(model_config.name, "NOT_STARTED"),
        )

    @router.delete("/{name}")
    async def undeploy_model(name: str) -> dict[str, str]:
        try:
            await to_thread.run_sync(deployment_manager.undeploy_model, name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "deleted", "model": name}

    @router.get("/{name}/stats")
    async def model_stats(name: str) -> dict[str, Any]:
        handle = deployment_manager.get_handle(name)
        if handle is None:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        try:
            return await handle.get_stats.remote()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not fetch stats: {exc}") from exc

    return router
