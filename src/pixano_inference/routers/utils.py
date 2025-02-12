# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Utils for the routers."""

import time
from datetime import datetime
from typing import TypeVar

from fastapi import HTTPException

from pixano_inference.models.base import ModelStatus
from pixano_inference.models_registry import get_values_from_model_registry
from pixano_inference.pydantic.base import BaseRequest, BaseResponse
from pixano_inference.tasks.image import ImageTask
from pixano_inference.tasks.multimodal import MultimodalImageNLPTask
from pixano_inference.tasks.task import Task


T = TypeVar("T", bound=BaseResponse)


async def execute_task_request(request: BaseRequest, task: Task, response_type: type[T]) -> T:
    """Execute a task request.

    Args:
        request: Request to execute.
        task: Task to execute
        response_type: Type of the response to return.

    Returns:
        Response of the request
    """
    start_time = time.time()
    model_str = request.model
    try:
        model, provider, model_task = get_values_from_model_registry(model_str)
    except KeyError:
        raise HTTPException(404, detail=f"Model {model_str} is not registered.")
    if task != model_task:
        raise HTTPException(400, detail=f"Model {model} does not support the {task.value} task.")

    if model.status == ModelStatus.RUNNING:
        raise HTTPException(503, "Model is running please wait for it to finish and try again.")

    model.status = ModelStatus.RUNNING
    try:
        match task:
            case ImageTask.MASK_GENERATION:
                output = provider.image_mask_generation(request, model)
            case MultimodalImageNLPTask.CONDITIONAL_GENERATION:
                output = provider.text_image_conditional_generation(request, model)
            case _:
                raise HTTPException(400, f"Task {task.value} is not yet supported.")
    except Exception as e:
        model.status = ModelStatus.IDLE
        raise e
    else:
        model.status = ModelStatus.IDLE

    response = response_type(
        timestamp=datetime.now(),
        processing_time=time.time() - start_time,
        metadata=model.metadata,
        data=output,
    )
    return response
