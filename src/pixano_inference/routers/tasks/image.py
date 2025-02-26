# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for Image tasks."""

from typing import Annotated

from fastapi import APIRouter, Depends

from pixano_inference.pydantic import (
    CeleryTask,
    ImageMaskGenerationRequest,
    ImageMaskGenerationResponse,
    ImageZeroShotDetectionResponse,
)
from pixano_inference.pydantic.tasks.image.zero_shot_detection import ImageZeroShotDetectionRequest
from pixano_inference.routers.utils import delete_task, execute_task_request, get_task_result
from pixano_inference.settings import Settings, get_pixano_inference_settings
from pixano_inference.tasks import ImageTask


router = APIRouter(prefix="/image", tags=["Image"])


@router.post("/mask_generation/", response_model=CeleryTask)
async def mask_generation(
    request: ImageMaskGenerationRequest,
    settings: Annotated[Settings, Depends(get_pixano_inference_settings)],
) -> CeleryTask:
    """Generate mask from an image and optionnaly points and bboxes.

    Args:
        request: Request for image mask generation.
        settings: Settings of the app.

    Returns:
        Id and status of the task.
    """
    return await execute_task_request(request=request, task=ImageTask.MASK_GENERATION, settings=settings)


@router.get("/mask_generation/{task_id}", response_model=ImageMaskGenerationResponse | CeleryTask)
async def get_mask_generation(task_id: str) -> ImageMaskGenerationResponse | CeleryTask:
    """Get the result of a image mask generation task.

    Args:
        task_id: ID of the task to retrieve.

    Returns:
        Response for image mask generation.
    """
    return await get_task_result(task_id, response_type=ImageMaskGenerationResponse)


@router.delete("/mask_generation/{task_id}", response_model=None)
async def delete_mask_generation(task_id: str) -> None:
    """Delete a image mask generation task."""
    return await delete_task(task_id=task_id)


@router.post("/zero_shot_detection/", response_model=CeleryTask)
async def zero_shot_detection(
    request: ImageZeroShotDetectionRequest,
    settings: Annotated[Settings, Depends(get_pixano_inference_settings)],
) -> CeleryTask:
    """Perform zero shot detection on an image.

    Args:
        request: Request for image zero shot detection.
        settings: Settings of the app.

    Returns:
        Id and status of the task.
    """
    return await execute_task_request(request=request, task=ImageTask.ZERO_SHOT_DETECTION, settings=settings)


@router.get("/zero_shot_detection/{task_id}", response_model=ImageZeroShotDetectionResponse | CeleryTask)
async def get_zero_shot_detection(task_id: str) -> ImageZeroShotDetectionResponse | CeleryTask:
    """Get the result of a image zero shot detection task.

    Args:
        task_id: ID of the task to retrieve.

    Returns:
        Response for image zero shot detection.
    """
    return await get_task_result(task_id, response_type=ImageZeroShotDetectionResponse)


@router.delete("/zero_shot_detection/{task_id}", response_model=None)
async def delete_zero_shot_detection(task_id: str) -> None:
    """Delete a image zero shot detection task."""
    return await delete_task(task_id=task_id)
