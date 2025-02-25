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
)
from pixano_inference.routers.utils import execute_task_request
from pixano_inference.settings import Settings, get_pixano_inference_settings
from pixano_inference.tasks.video import VideoTask


router = APIRouter(prefix="/video", tags=["Video"])


@router.post("/mask_generation", response_model=CeleryTask)
async def mask_generation(
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
