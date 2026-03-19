# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Task inference routes with schema bridging."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pixano_inference.models.detection import DetectionInput
from pixano_inference.models.segmentation import SegmentationInput
from pixano_inference.models.tracking import TrackingInput
from pixano_inference.models.vlm import VLMInput
from pixano_inference.schemas.tasks import (
    DetectionRequest,
    SegmentationRequest,
    TrackingRequest,
    VLMRequest,
)


if TYPE_CHECKING:
    from pixano_inference.ray.app import DeploymentManager

logger = logging.getLogger(__name__)


async def _run_inference(
    deployment_manager: DeploymentManager,
    model_name: str,
    input_data: BaseModel,
    task_name: str,
) -> dict[str, Any]:
    """Run inference on a deployed model.

    Args:
        deployment_manager: The deployment manager instance.
        model_name: Name of the model to run inference on.
        input_data: Typed Input object to pass to the model.
        task_name: Task name for error messages.

    Returns:
        Response dictionary.
    """
    handle = deployment_manager.get_handle(model_name)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    start_time = time.time()

    try:
        import ray

        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: ray.get(handle.predict.remote(input_data))
        )
    except Exception as e:
        logger.exception(f"Inference error for model '{model_name}' on task '{task_name}': {e}")
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


def register_inference_routes(app: FastAPI, deployment_manager: DeploymentManager) -> None:
    """Register task inference endpoints.

    Args:
        app: FastAPI application.
        deployment_manager: The deployment manager instance.
    """

    @app.post("/tasks/image/mask_generation/")
    async def image_mask_generation(request: SegmentationRequest) -> dict[str, Any]:
        """Run image mask generation inference."""
        input_obj = request.to_base_model(SegmentationInput)
        return await _run_inference(deployment_manager, request.model, input_obj, "image_mask_generation")

    @app.post("/tasks/video/mask_generation/")
    async def video_mask_generation(request: TrackingRequest) -> dict[str, Any]:
        """Run video mask generation inference."""
        input_obj = request.to_base_model(TrackingInput)
        return await _run_inference(deployment_manager, request.model, input_obj, "video_mask_generation")

    @app.post("/tasks/multimodal/text-image/conditional_generation/")
    async def text_image_conditional_generation(
        request: VLMRequest,
    ) -> dict[str, Any]:
        """Run text-image conditional generation inference."""
        input_obj = request.to_base_model(VLMInput)
        return await _run_inference(deployment_manager, request.model, input_obj, "text_image_conditional_generation")

    @app.post("/tasks/image/zero_shot_detection/")
    async def image_zero_shot_detection(request: DetectionRequest) -> dict[str, Any]:
        """Run image zero-shot detection inference."""
        input_obj = request.to_base_model(DetectionInput)
        return await _run_inference(deployment_manager, request.model, input_obj, "image_zero_shot_detection")

    @app.post("/tasks/image/instance_segmentation/")
    async def image_instance_segmentation(request: DetectionRequest) -> dict[str, Any]:
        """Run image instance segmentation inference."""
        input_obj = request.to_base_model(DetectionInput)
        return await _run_inference(deployment_manager, request.model, input_obj, "image_instance_segmentation")
