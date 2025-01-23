# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for NLP tasks."""

from fastapi import APIRouter

from pixano_inference.pydantic.tasks.multimodal.conditional_generation import (
    TextImageConditionalGenerationRequest,
    TextImageConditionalGenerationResponse,
)
from pixano_inference.routers.utils import execute_task_request
from pixano_inference.tasks import MultimodalImageNLPTask


router = APIRouter(prefix="/tasks/multimodal/image-text", tags=["Multi-Modal Image-Text Tasks"])


@router.post("/conditional_generation/", response_model=TextImageConditionalGenerationResponse)
async def image_text_conditional_generation(
    request: TextImageConditionalGenerationRequest,
) -> TextImageConditionalGenerationResponse:
    """Generate text from an image and a prompt.

    Args:
        request: Request for text-image conditional generation.

    Returns:
        Response for text-image conditional generation.
    """
    return await execute_task_request(
        request, MultimodalImageNLPTask.CONDITIONAL_GENERATION, TextImageConditionalGenerationResponse
    )
