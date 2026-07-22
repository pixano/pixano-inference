# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""/v1 asynchronous tracking-job routes (submit / poll / cancel)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request

from pixano_inference.jobs import serialize_job

from .helpers import build_binary_request_from_request, get_validated_handle
from .schemas import TrackingRequestV1


if TYPE_CHECKING:
    from pixano_inference.ray.app import DeploymentManager


def build_jobs_router(deployment_manager: DeploymentManager) -> APIRouter:
    """Build the tracking-jobs router bound to *deployment_manager*."""
    router = APIRouter(tags=["jobs"])

    def _submit(model: str, input_obj: Any) -> dict[str, Any]:
        get_validated_handle(deployment_manager, model, "tracking")
        job_id = deployment_manager.submit_tracking_job(model, input_obj)
        job = deployment_manager.get_tracking_job(job_id)
        if job is None:
            raise HTTPException(status_code=500, detail=f"Job '{job_id}' was not created.")
        return serialize_job(job_id, job)

    @router.post("/inference/tracking/jobs", status_code=202)
    async def submit_tracking_job(request: TrackingRequestV1) -> dict[str, Any]:
        return _submit(request.model, request.to_input())

    @router.post("/inference/tracking/jobs/binary", status_code=202)
    async def submit_tracking_job_binary(request: Request) -> dict[str, Any]:
        parsed = await build_binary_request_from_request(
            request, TrackingRequestV1, file_field="frames", payload_key="video"
        )
        return _submit(parsed.model, parsed.to_input())

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = deployment_manager.get_tracking_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return serialize_job(job_id, job)

    @router.delete("/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        job = deployment_manager.cancel_tracking_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return serialize_job(job_id, job)

    return router
