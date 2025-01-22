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
    model_str, provider_str = request.model, request.provider
    try:
        model, provider, model_task = get_values_from_model_registry(model_str, provider_str)
    except KeyError:
        return HTTPException(404, detail=f"Model {model_str} from provider {provider_str} is not registered.")
    if task != model_task:
        return HTTPException(400, detail=f"Model {model} does not support the {task.value} task.")

    sleeping_loops = 0
    while model.status == ModelStatus.RUNNING and sleeping_loops < 100:
        time.sleep(0.1)
        sleeping_loops += 1
        if sleeping_loops == 100:
            return HTTPException(503, "Model is running please resend your request.")

    model.status = ModelStatus.RUNNING
    match task:
        case ImageTask.MASK_GENERATION:
            output = provider.image_mask_generation(request, model)
        case MultimodalImageNLPTask.CONDITIONAL_GENERATION:
            output = provider.text_image_conditional_generation(request, model)
        case _:
            raise NotImplementedError(f"Task {task.value} is not yet supported.")
    model.status = ModelStatus.IDLE
    response = response_type(
        timestamp=datetime.now(),
        processing_time=time.time() - start_time,
        metadata=model.metadata,
        data=output,
    )
    return response
