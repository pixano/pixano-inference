# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Capability-based inference routes with schema bridging."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from pixano_inference.schemas.inference import (
    DetectionRequest,
    SegmentationRequest,
    TrackingRequest,
    VLMRequest,
)


if TYPE_CHECKING:
    from pixano_inference.ray.app import DeploymentManager, TrackingJobRecord

logger = logging.getLogger(__name__)
_LEGACY_BINARY_METADATA_MAX_PART_SIZE = 32 * 1024 * 1024


def _build_binary_request(request_type: type[BaseModel], metadata: str, **extra_fields: Any) -> BaseModel:
    try:
        payload = json.loads(metadata)
    except JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid binary inference metadata: {exc.msg}") from exc

    payload.update(extra_fields)
    try:
        return request_type.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


async def _read_metadata_upload(upload: UploadFile | None) -> str | None:
    if upload is None:
        return None

    payload = await upload.read()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Binary inference metadata must be valid UTF-8 JSON.") from exc


async def _build_binary_request_from_request(
    request: Request,
    request_type: type[BaseModel],
    *,
    max_part_size: int,
    file_field: str,
    payload_key: str,
    **extra_fields: Any,
) -> BaseModel:
    form = await request.form(max_part_size=max_part_size)
    metadata_upload = form.get("metadata")
    payload_field = form.get(file_field)

    metadata_text: str | None
    if isinstance(metadata_upload, (UploadFile, StarletteUploadFile)):
        metadata_text = await _read_metadata_upload(metadata_upload)
    elif isinstance(metadata_upload, str):
        metadata_text = metadata_upload
    else:
        metadata_text = None

    if metadata_text is None:
        raise HTTPException(status_code=400, detail="Missing binary inference metadata.")

    if file_field == "image":
        if not isinstance(payload_field, (UploadFile, StarletteUploadFile)):
            raise HTTPException(status_code=400, detail="Missing binary image payload.")
        extra_fields[payload_key] = await payload_field.read()
    else:
        files = form.getlist(file_field)
        if not files or not all(isinstance(file, (UploadFile, StarletteUploadFile)) for file in files):
            raise HTTPException(status_code=400, detail="Missing binary frame payload.")
        extra_fields[payload_key] = [await file.read() for file in files]

    return _build_binary_request(request_type, metadata_text, **extra_fields)


def _get_validated_handle(
    deployment_manager: DeploymentManager,
    model_name: str,
    expected_capability: str,
) -> Any:
    handle = deployment_manager.get_handle(model_name)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    actual_capability = deployment_manager.get_model_capability(model_name)
    if actual_capability != expected_capability:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' does not support '{expected_capability}' inference. "
                f"It is deployed as '{actual_capability}'."
            ),
        )

    return handle


async def _run_inference(
    deployment_manager: DeploymentManager,
    model_name: str,
    input_data: BaseModel,
    expected_capability: str,
) -> dict[str, Any]:
    """Run inference on a deployed model.

    Args:
        deployment_manager: The deployment manager instance.
        model_name: Name of the model to run inference on.
        input_data: Typed Input object to pass to the model.
        expected_capability: Capability expected by the endpoint.

    Returns:
        Response dictionary.
    """
    handle = _get_validated_handle(deployment_manager, model_name, expected_capability)
    start_time = time.time()

    try:
        import ray

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: ray.get(handle.predict.remote(input_data))
        )
    except Exception as e:
        logger.exception(f"Inference error for model '{model_name}' on capability '{expected_capability}': {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    processing_time = time.time() - start_time

    return {
        "id": f"ray-{model_name}-{int(time.time() * 1000)}",
        "status": "SUCCESS",
        "timestamp": datetime.now(),
        "processing_time": processing_time,
        "metadata": deployment_manager.get_model_metadata(model_name),
        "data": result.model_dump(),
    }


def _serialize_tracking_job(job_id: str, job: TrackingJobRecord) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": job.status,
        "detail": job.detail,
        "data": job.result if job.status == "completed" else None,
        "metadata": job.metadata,
        "timestamp": job.timestamp,
        "processing_time": job.processing_time,
    }


def register_inference_routes(app: FastAPI, deployment_manager: DeploymentManager) -> None:
    """Register capability-based inference endpoints.

    Args:
        app: FastAPI application.
        deployment_manager: The deployment manager instance.
    """

    @app.post("/inference/segmentation/")
    async def segmentation(request: SegmentationRequest) -> dict[str, Any]:
        """Run segmentation inference."""
        input_obj = request.to_input()
        return await _run_inference(deployment_manager, request.model, input_obj, "segmentation")

    @app.post("/inference/segmentation/binary")
    async def segmentation_binary(
        request: Request,
    ) -> dict[str, Any]:
        """Run segmentation inference from a binary image upload."""
        parsed_request = await _build_binary_request_from_request(
            request,
            SegmentationRequest,
            max_part_size=_LEGACY_BINARY_METADATA_MAX_PART_SIZE,
            file_field="image",
            payload_key="image",
        )
        input_obj = parsed_request.to_input()
        return await _run_inference(deployment_manager, parsed_request.model, input_obj, "segmentation")

    @app.post("/inference/tracking/")
    async def tracking(request: TrackingRequest) -> dict[str, Any]:
        """Run tracking inference."""
        input_obj = request.to_input()
        return await _run_inference(deployment_manager, request.model, input_obj, "tracking")

    @app.post("/inference/tracking/binary")
    async def tracking_binary(
        request: Request,
    ) -> dict[str, Any]:
        """Run tracking inference from uploaded frame binaries."""
        parsed_request = await _build_binary_request_from_request(
            request,
            TrackingRequest,
            max_part_size=_LEGACY_BINARY_METADATA_MAX_PART_SIZE,
            file_field="frames",
            payload_key="video",
        )
        input_obj = parsed_request.to_input()
        return await _run_inference(deployment_manager, parsed_request.model, input_obj, "tracking")

    @app.post("/inference/tracking/jobs/")
    async def tracking_job_submit(request: TrackingRequest) -> dict[str, Any]:
        """Submit an asynchronous tracking job."""
        _get_validated_handle(deployment_manager, request.model, "tracking")
        input_obj = request.to_input()
        job_id = deployment_manager.submit_tracking_job(request.model, input_obj)
        job = deployment_manager.get_tracking_job(job_id)
        if job is None:
            raise HTTPException(status_code=500, detail=f"Tracking job '{job_id}' was not created.")
        return _serialize_tracking_job(job_id, job)

    @app.post("/inference/tracking/jobs/binary")
    async def tracking_job_submit_binary(request: Request) -> dict[str, Any]:
        """Submit an asynchronous tracking job from uploaded frame binaries."""
        parsed_request = await _build_binary_request_from_request(
            request,
            TrackingRequest,
            max_part_size=_LEGACY_BINARY_METADATA_MAX_PART_SIZE,
            file_field="frames",
            payload_key="video",
        )
        _get_validated_handle(deployment_manager, parsed_request.model, "tracking")
        input_obj = parsed_request.to_input()
        job_id = deployment_manager.submit_tracking_job(parsed_request.model, input_obj)
        job = deployment_manager.get_tracking_job(job_id)
        if job is None:
            raise HTTPException(status_code=500, detail=f"Tracking job '{job_id}' was not created.")
        return _serialize_tracking_job(job_id, job)

    @app.get("/inference/tracking/jobs/{job_id}")
    async def tracking_job_status(job_id: str) -> dict[str, Any]:
        """Poll the current status of an asynchronous tracking job."""
        job = deployment_manager.get_tracking_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Tracking job '{job_id}' not found")
        return _serialize_tracking_job(job_id, job)

    @app.delete("/inference/tracking/jobs/{job_id}")
    async def tracking_job_cancel(job_id: str) -> dict[str, Any]:
        """Cancel an asynchronous tracking job."""
        job = deployment_manager.cancel_tracking_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Tracking job '{job_id}' not found")
        return _serialize_tracking_job(job_id, job)

    @app.post("/inference/vlm/")
    async def vlm(request: VLMRequest) -> dict[str, Any]:
        """Run VLM inference."""
        input_obj = request.to_input()
        return await _run_inference(deployment_manager, request.model, input_obj, "vlm")

    @app.post("/inference/detection/")
    async def detection(request: DetectionRequest) -> dict[str, Any]:
        """Run detection inference."""
        input_obj = request.to_input()
        return await _run_inference(deployment_manager, request.model, input_obj, "detection")
