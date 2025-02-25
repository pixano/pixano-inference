# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for NLP tasks."""

from typing import Annotated

from fastapi import APIRouter, Depends

from pixano_inference.pydantic import (
    CeleryTask,
    TextImageConditionalGenerationRequest,
    TextImageConditionalGenerationResponse,
)
from pixano_inference.routers.utils import delete_task, execute_task_request, get_task_result
from pixano_inference.settings import Settings, get_pixano_inference_settings
from pixano_inference.tasks import MultimodalImageNLPTask


router = APIRouter(prefix="/multimodal", tags=["Multi-Modal"])


@router.post("/text-image/conditional_generation/", response_model=CeleryTask)
async def text_image_conditional_generation(
    request: TextImageConditionalGenerationRequest,
    settings: Annotated[Settings, Depends(get_pixano_inference_settings)],
) -> CeleryTask:
    """Generate text from an image and a prompt.

    Args:
        request: Request for text-image conditional generation.
        settings: Settings of the app.

    Returns:
        Response for text-image conditional generation.
    """
    return await execute_task_request(
        request=request, task=MultimodalImageNLPTask.CONDITIONAL_GENERATION, settings=settings
    )


@router.get(
    "/text-image/conditional_generation/{task_id}", response_model=TextImageConditionalGenerationResponse | CeleryTask
)
async def get_text_image_conditional_generation(task_id: str) -> TextImageConditionalGenerationResponse | CeleryTask:
    """Get the result of a text image conditional generation task.

    Args:
        task_id: ID of the task to retrieve.

    Returns:
        Response for text image conditional generation.
    """
    result = await get_task_result(task_id, response_type=TextImageConditionalGenerationResponse)
    return result


@router.delete("/text-image/conditional_generation/{task_id}", response_model=None)
async def delete_text_image_conditional_generation(task_id: str) -> None:
    """Delete a text image conditional generation task."""
    return await delete_task(task_id=task_id)
