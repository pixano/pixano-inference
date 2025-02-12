# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for model providers."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException

from pixano_inference.models_registry import get_model_from_registry
from pixano_inference.providers.base import ModelProvider
from pixano_inference.providers.registry import get_provider, is_provider
from pixano_inference.pydantic import ModelConfig
from pixano_inference.settings import Settings, get_pixano_inference_settings


router = APIRouter(prefix="/providers", tags=["models"])


@router.post("/instantiate")
async def instantiate_model(
    config: ModelConfig,
    provider: Annotated[str, Body()],
    settings: Annotated[Settings, Depends(get_pixano_inference_settings)],
):
    """Instantiate a model from a provider.

    Args:
        config: Model configuration for instantiation.
        provider: The model provider.
        settings: Settings for the instantiation.
    """
    if not is_provider(provider):
        raise HTTPException(status_code=404, detail=f"Provider {provider} does not exist.")
    elif provider == "transformers":
        try:
            p: ModelProvider = get_provider("transformers")()  # type: ignore[assignment]
        except ImportError:
            raise HTTPException(status_code=500, detail="Transformers library is not installed.")
    elif provider == "sam2":
        try:
            p = get_provider("sam2")()  # type: ignore[assignment]
        except ImportError:
            raise HTTPException(status_code=500, detail="Sam2 is not installed.")
    elif provider == "vllm":
        try:
            p = get_provider("vllm")()  # type: ignore[assignment]
        except ImportError:
            raise HTTPException(status_code=500, detail="vLLM library is not installed.")
    else:
        p = get_provider(provider)()  # type: ignore[assignment]

    p.load_model(settings=settings, **config.model_dump())
    return


@router.post("/transformers/instantiate/")
async def instantiate_transformer_model(
    config: ModelConfig, settings: Annotated[Settings, Depends(get_pixano_inference_settings)]
):
    """Instantiate a model from transformers.

    Args:
        config: Model configuration for instantiation.
        settings: Settings for the instantiation.
    """
    return instantiate_model(config=config, provider="transformers", settings=settings)


@router.post("/sam2/instantiate/")
async def instantiate_sam2_model(
    config: ModelConfig, settings: Annotated[Settings, Depends(get_pixano_inference_settings)]
):
    """Instantiate a model from sam2.

    Args:
        config: Model configuration for instantiation.
        settings: Settings for the instantiation.
    """
    return instantiate_model(config=config, provider="sam2", settings=settings)


@router.delete("/model/{model_name}")
async def delete_model(model_name: str):
    """Delete a model from the system.

    Args:
        model_name: The name of the model to be deleted.
    """
    try:
        model = get_model_from_registry(model_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found") from e
    model.delete()
    return
