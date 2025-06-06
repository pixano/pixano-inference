# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for Image tasks."""

from typing import Annotated

from fastapi import APIRouter, Depends

from pixano_inference.pydantic.base import CeleryTask
from pixano_inference.pydantic.tasks.video.mask_generation import (
    VideoMaskGenerationRequest,
    VideoMaskGenerationResponse,
)
from pixano_inference.routers.utils import delete_task, execute_task_request, get_task_result
from pixano_inference.settings import Settings, get_pixano_inference_settings
from pixano_inference.tasks.video import VideoTask


router = APIRouter(prefix="/video", tags=["Video"])


@router.post("/mask_generation/", response_model=CeleryTask)
async def video_mask_generation(
    request: VideoMaskGenerationRequest,
    settings: Annotated[Settings, Depends(get_pixano_inference_settings)],
) -> CeleryTask:
    """Generate mask from a video and optionnaly points and bboxes.

    Args:
        request: Request for mask generation.
        settings: Settings of the app.

    Returns:
        Response for mask generation.
    """
    return await execute_task_request(request=request, task=VideoTask.MASK_GENERATION, settings=settings)


@router.get("/mask_generation/{task_id}", response_model=VideoMaskGenerationResponse | CeleryTask)
async def get_video_mask_generation(task_id: str) -> VideoMaskGenerationResponse | CeleryTask:
    """Get the result of a mask generation task.

    Args:
        task_id: ID of the task to retrieve.

    Returns:
        Response for mask generation.
    """
    result = await get_task_result(task_id, response_type=VideoMaskGenerationResponse)
    return result


@router.delete("/mask_generation/{task_id}", response_model=None)
async def delete_video_mask_generation(task_id: str) -> None:
    """Delete a mask generation task."""
    return await delete_task(task_id=task_id)
