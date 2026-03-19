# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Capability-based inference routes with schema bridging."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pixano_inference.schemas.inference import (
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

    @app.post("/inference/tracking/")
    async def tracking(request: TrackingRequest) -> dict[str, Any]:
        """Run tracking inference."""
        input_obj = request.to_input()
        return await _run_inference(deployment_manager, request.model, input_obj, "tracking")

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
