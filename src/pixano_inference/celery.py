# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Celery configuration file."""

import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from subprocess import Popen
from typing import Any, Tuple, cast

from celery import Celery
from celery.result import AsyncResult
from fastapi.encoders import jsonable_encoder

from pixano_inference.models import BaseInferenceModel
from pixano_inference.providers.base import BaseProvider, ModelProvider
from pixano_inference.providers.utils import instantiate_provider
from pixano_inference.pydantic import (
    CeleryTask,
    ImageMaskGenerationRequest,
    ImageZeroShotDetectionRequest,
    ModelConfig,
    TextImageConditionalGenerationRequest,
    VideoMaskGenerationRequest,
)
from pixano_inference.tasks import ImageTask, MultimodalImageNLPTask, Task, VideoTask, str_to_task
from pixano_inference.utils.package import assert_torch_installed, is_torch_installed


if is_torch_installed():
    import torch

# Maps model_name -> (provider, model, task)
model_registry: dict[str, Tuple[BaseProvider, BaseInferenceModel, Task]] = {}

uvicorn_logger = logging.getLogger("uvicorn.error")

queues_to_workers: dict[str, Popen] = {}


def create_celery_app() -> Celery:
    """Create a new celery app."""
    redis_url: str = os.environ.get("REDIS_URL", "localhost")
    redis_port: int = int(os.environ.get("REDIS_PORT", 6379))
    redis_db_number: int = int(os.environ.get("REDIS_DB_NUMBER", 0))

    redis_url = f"redis://{redis_url}:{redis_port}/{redis_db_number}"

    app = Celery(__name__, broker=redis_url, backend=redis_url)

    return app


celery_app = create_celery_app()


@celery_app.task
def instantiate_model(provider: str, model_config: dict[str, Any], gpu: int | None) -> None:
    """Instantiate a model."""
    assert_torch_installed()
    device = torch.device(f"cuda:{gpu}") if gpu is not None else torch.device("cpu")
    provider_instance = instantiate_provider(provider)
    provider_instance = cast(ModelProvider, provider_instance)

    model_instance = provider_instance.load_model(**model_config, device=device)
    task = str_to_task(model_config["task"])
    model_registry[model_config["name"]] = (provider_instance, model_instance, task)


@celery_app.task
def delete_model(model_name: str) -> None:
    """Delete model."""
    provider, model, _ = model_registry.pop(model_name, (None, None, None))
    if model is not None:
        try:
            model.delete()
        except NameError:
            pass


@celery_app.task
def predict(model_name: str, request: dict[str, Any]) -> dict[str, Any]:
    """Run a model inference from the request."""
    if model_name not in model_registry:
        raise ValueError(f"Model '{model_name}' not found")

    provider, model, task = model_registry[model_name]

    start_time = time.time()
    match task:
        case ImageTask.MASK_GENERATION:
            output = provider.image_mask_generation(
                request=ImageMaskGenerationRequest.model_construct(**request), model=model
            )
        case MultimodalImageNLPTask.CONDITIONAL_GENERATION:
            output = provider.text_image_conditional_generation(
                request=TextImageConditionalGenerationRequest.model_construct(**request), model=model
            )
        case VideoTask.MASK_GENERATION:
            output = provider.video_mask_generation(
                request=VideoMaskGenerationRequest.model_construct(**request), model=model
            )
        case ImageTask.ZERO_SHOT_DETECTION:
            output = provider.image_zero_shot_detection(
                request=ImageZeroShotDetectionRequest.model_construct(**request), model=model
            )
        case _:
            raise ValueError(f"Unknown task: {task}")
    response = {
        "timestamp": datetime.now(),
        "processing_time": time.time() - start_time,
        "metadata": model.metadata,
        "data": output.model_dump(),
    }
    return response


def model_queue_name(model_name: str) -> str:
    """Get the name of the queue for a given model."""
    return f"{model_name}_queue"


def add_celery_worker_and_queue(provider: str, model_config: ModelConfig, gpu: int | None) -> CeleryTask:
    """Add a new worker and a queue to the celery app to handle model."""
    queue = model_queue_name(model_config.name)
    celery_app.control.add_consumer(queue=model_queue_name(model_config.name), reply=True)

    command = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "pixano_inference.celery.celery_app",
        "worker",
        "--loglevel=INFO",
        "-Q",
        queue,
        "--pool=solo",
    ]
    worker = Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    uvicorn_logger.info(f"Spawned Celery worker {worker.pid} handling model {model_config.name}.")
    queues_to_workers[queue] = worker

    task: AsyncResult = instantiate_model.apply_async(
        (provider, jsonable_encoder(model_config.model_dump()), gpu), queue=queue
    )
    task_result, result = list(task.collect())[0]

    return CeleryTask(id=task.id, status=task_result.status)


def delete_celery_worker_and_queue(model_name: str):
    """Delete a worker and a queue of the celery app that handled a model."""
    queue = model_queue_name(model_name)

    command = [
        sys.executable,
        "-m",
        "celery",
        "-Q",
        queue,
        "purge",
    ]
    purge_process = Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    purge_process.wait()
    purge_process.kill()
    try:
        worker = queues_to_workers.pop(queue)
    except KeyError:  # Instantiation failed before storing the worker
        pass
    else:
        delete_model.apply_async((model_name,), queue=queue).get()
        os.killpg(os.getpgid(worker.pid), signal.SIGTERM)
        worker.wait()
        uvicorn_logger.info(f"Killed Celery worker {worker.pid} handling model {model_name}.")
    celery_app.control.cancel_consumer(queue=queue)
