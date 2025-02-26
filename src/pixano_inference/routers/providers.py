# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""API routes for model providers."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException

from pixano_inference.celery import add_celery_worker_and_queue, celery_app, delete_celery_worker_and_queue
from pixano_inference.providers.utils import instantiate_provider
from pixano_inference.pydantic import CeleryTask, ModelConfig
from pixano_inference.settings import Settings, get_pixano_inference_settings
from pixano_inference.tasks.utils import is_task


router = APIRouter(prefix="/providers", tags=["models"])


@router.post("/instantiate", response_model=CeleryTask)
async def instantiate_model(
    config: ModelConfig,
    provider: Annotated[str, Body()],
    settings: Annotated[Settings, Depends(get_pixano_inference_settings)],
):
    """Instantiate a model from a provider.

    Args:
        config: Model configuration for instantiation.
        provider: The model provider.
        settings: Settings of the app.
    """
    try:
        instantiate_provider(provider)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Provider {provider} does not exist.") from e
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Provider {provider} is not installed.") from e

    if not is_task(config.task):
        raise HTTPException(status_code=404, detail=f"Task {config.task} is not a supported task.")

    gpu = settings.add_model(config.name, config.task)
    try:
        task = add_celery_worker_and_queue(model_config=config, provider=provider, gpu=gpu)
    except Exception as e:
        settings.remove_model(config.name)
        delete_celery_worker_and_queue(config.name)
        raise HTTPException(status_code=400, detail=f"Error while instantiating {config.task} - {str(e)}") from e
    return task


@router.get("/instantiate/{task_id}", response_model=CeleryTask)
async def get_instantiate_model_status(
    task_id: str,
) -> CeleryTask:
    """Return status of model instantiation."""
    task_result = celery_app.AsyncResult(task_id)
    return CeleryTask(id=task_result.id, status=task_result.status)


@router.delete("/model/{model_name}")
async def delete_model(
    model_name: str,
    settings: Annotated[Settings, Depends(get_pixano_inference_settings)],
):
    """Delete a model from the system.

    Args:
        model_name: The name of the model to be deleted.
        settings: Settings of the app.
    """
    if model_name not in settings.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    settings.remove_model(model_name)
    delete_celery_worker_and_queue(model_name)
    return
