# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for the app."""

from typing import Annotated

from fastapi import APIRouter, Depends

from pixano_inference.pydantic.models import ModelInfo
from pixano_inference.settings import Settings, get_pixano_inference_settings


router = APIRouter(prefix="/app", tags=["App"])


@router.get("/settings/", response_model=Settings)
async def get_settings(
    settings: Annotated[Settings, Depends(get_pixano_inference_settings)],
) -> Settings:
    """Get the current settings of the app."""
    return settings


@router.get("/models/", response_model=list[ModelInfo])
async def get_list_models(
    settings: Annotated[Settings, Depends(get_pixano_inference_settings)],
) -> list[ModelInfo]:
    """List all models available in the app."""
    models = [ModelInfo(name=model_name, task=task) for model_name, task in settings.models_to_task.items()]
    return models
