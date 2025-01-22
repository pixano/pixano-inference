# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for model providers."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from pixano_inference.providers.registry import get_provider
from pixano_inference.providers.transformers import TransformersProvider
from pixano_inference.pydantic import ModelConfig
from pixano_inference.settings import Settings, get_pixano_inference_settings


router = APIRouter(prefix="/providers", tags=["models"])


@router.post("/transformers/instantiate/")
async def instantiate_transformer_model(
    config: ModelConfig, settings: Annotated[Settings, Depends(get_pixano_inference_settings)]
):
    """Instantiate a model from transformers.

    Args:
        config: Model configuration for instantiation.
        settings: Settings for the instantiation.
    """
    try:
        provider: TransformersProvider = get_provider("transformers")()  # type: ignore[operator]
    except ImportError:
        raise HTTPException(status_code=500, detail="Transformers library is not installed")

    provider.load_model(settings=settings, **config.model_dump())
    return
