# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for Image tasks."""

from fastapi import APIRouter

from pixano_inference.pydantic.tasks.video.mask_generation import (
    VideoMaskGenerationRequest,
    VideoMaskGenerationResponse,
)
from pixano_inference.routers.utils import execute_task_request
from pixano_inference.tasks.video import VideoTask


router = APIRouter(prefix="/tasks/video", tags=["Image Tasks"])


@router.post("/mask_generation/", response_model=VideoMaskGenerationResponse)
async def mask_generation(
    request: VideoMaskGenerationRequest,
) -> VideoMaskGenerationResponse:
    """Generate mask from a video and optionnaly points and bboxes.

    Args:
        request: Request for mask generation.

    Returns:
        Response for mask generation.
    """
    return await execute_task_request(request, VideoTask.MASK_GENERATION, VideoMaskGenerationResponse)
