# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Utils for the routers."""

from celery import states
from celery.result import AsyncResult
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from pixano_inference.celery import celery_app, model_queue_name, predict
from pixano_inference.pydantic import BaseRequest, BaseResponse, CeleryTask
from pixano_inference.settings import Settings
from pixano_inference.tasks import Task


async def execute_task_request(request: BaseRequest, task: Task, settings: Settings) -> CeleryTask:
    """Execute a task request.

    Args:
        request: Request to execute.
        task: Task to execute
        settings: Settings of the app.

    Returns:
        Response of the request
    """
    model_name = request.model
    if model_name not in settings.models:
        raise HTTPException(404, detail=f"Model {model_name} is not registered.")
    model_task = settings.models_to_task[model_name]
    if task.value != model_task:
        raise HTTPException(400, detail=f"Model {model_name} does not support the {task.value} task.")

    queue = model_queue_name(model_name)
    celery_task: AsyncResult = predict.apply_async(
        (
            model_name,
            jsonable_encoder(request),
        ),
        queue=queue,
    )
    return CeleryTask(id=celery_task.id, status=states.PENDING)


async def get_task_result(task_id: str, response_type: type[BaseResponse]) -> CeleryTask | BaseResponse:
    """Get the result of a task.

    Args:
        task_id: ID of the task to retrieve.
        response_type: Type of response to return.

    Returns:
        Response for the task generation.
    """
    task_result = celery_app.AsyncResult(task_id)
    status, result = task_result.status, task_result.result
    if status in [states.PENDING, states.REVOKED, states.STARTED, states.FAILURE, states.REVOKED]:
        return CeleryTask(id=task_id, status=status)
    elif status != states.SUCCESS:
        raise HTTPException(status_code=500, detail=f"Unknown task status {status}")
    result["id"] = task_id
    result["status"] = status
    response = response_type.model_construct(**result)
    task_result.forget()
    return response


async def delete_task(task_id: str) -> None:
    """Delete a task."""
    celery_app.control.revoke(task_id, terminate=True)
    return
