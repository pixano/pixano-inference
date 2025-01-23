# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for Image tasks."""

from fastapi import APIRouter

from pixano_inference.pydantic.tasks.image.mask_generation import (
    ImageMaskGenerationRequest,
    ImageMaskGenerationResponse,
)
from pixano_inference.routers.utils import execute_task_request
from pixano_inference.tasks.image import ImageTask


router = APIRouter(prefix="/tasks/image", tags=["Image Tasks"])


@router.post("/mask_generation/", response_model=ImageMaskGenerationResponse)
async def mask_generation(
    request: ImageMaskGenerationRequest,
) -> ImageMaskGenerationResponse:
    """Generate mask from an image and optionnaly points and bboxes.

    Args:
        request: Request for mask generation.

    Returns:
        Response for mask generation.
    """
    return await execute_task_request(request, ImageTask.MASK_GENERATION, ImageMaskGenerationResponse)
