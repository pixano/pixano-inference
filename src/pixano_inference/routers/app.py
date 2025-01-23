# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for the app."""

from fastapi import APIRouter

from pixano_inference.models_registry import list_models
from pixano_inference.pydantic.models import ModelInfo
from pixano_inference.settings import PIXANO_INFERENCE_SETTINGS, Settings


router = APIRouter(prefix="/app", tags=["App"])


@router.get("/settings/", response_model=Settings)
async def get_settings() -> Settings:
    """Get the current settings of the app."""
    return PIXANO_INFERENCE_SETTINGS


@router.get("/models/", response_model=list[ModelInfo])
async def get_list_models() -> list[ModelInfo]:
    """List all models available in the app."""
    models = [
        ModelInfo(name=model_name, provider=provider_name, task=task.value)
        for (model_name, provider_name), (model, provider, task) in list_models()
    ]
    return models
