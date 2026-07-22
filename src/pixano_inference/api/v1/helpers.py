# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Shared helpers for the /v1 inference routes: handle validation, dispatch, multipart."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

from fastapi import HTTPException, Request, UploadFile
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile


if TYPE_CHECKING:
    from pixano_inference.ray.app import DeploymentManager


logger = logging.getLogger(__name__)

_RequestT = TypeVar("_RequestT", bound=BaseModel)

BINARY_METADATA_MAX_PART_SIZE = 32 * 1024 * 1024


def get_validated_handle(deployment_manager: DeploymentManager, model_name: str, expected_capability: str) -> Any:
    """Resolve a deployment handle and check its capability, or raise 404/400."""
    handle = deployment_manager.get_handle(model_name)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    actual_capability = deployment_manager.get_model_capability(model_name)
    if actual_capability != expected_capability:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_name}' does not support '{expected_capability}' inference. "
                f"It is deployed as '{actual_capability}'."
            ),
        )
    return handle


async def _await_response(response: Any) -> Any:
    """Await a Serve DeploymentResponse (so asyncio.wait_for accepts it)."""
    return await response


async def run_inference(
    deployment_manager: DeploymentManager,
    model_name: str,
    input_data: BaseModel,
    expected_capability: str,
) -> dict[str, Any]:
    """Dispatch inference through the Serve handle with a per-capability timeout."""
    handle = get_validated_handle(deployment_manager, model_name, expected_capability)
    timeout_s = deployment_manager.get_timeout(model_name, expected_capability)
    start_time = time.time()

    response = handle.predict.remote(input_data)
    try:
        result = await asyncio.wait_for(_await_response(response), timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            response.cancel()
        except Exception:
            pass
        raise HTTPException(status_code=504, detail=f"Inference timed out after {timeout_s:g}s.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference error for model '%s' on '%s': %s", model_name, expected_capability, exc)
        raise HTTPException(status_code=500, detail="Inference error.")

    return {
        "id": f"ray-{model_name}-{uuid4().hex[:12]}",
        "status": "SUCCESS",
        "timestamp": datetime.now(timezone.utc),
        "processing_time": time.time() - start_time,
        "metadata": deployment_manager.get_model_metadata(model_name),
        "data": result.model_dump(by_alias=True),
    }


def build_binary_request(request_type: type[_RequestT], metadata: str, **extra_fields: Any) -> _RequestT:
    """Parse JSON metadata + binary media fields into a typed request model."""
    try:
        payload = json.loads(metadata)
    except JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid binary inference metadata: {exc.msg}") from exc
    payload.update(extra_fields)
    try:
        return request_type.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


async def _read_metadata_upload(upload: Any) -> str | None:
    if upload is None:
        return None
    payload = await upload.read()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Binary inference metadata must be valid UTF-8 JSON.") from exc


async def build_binary_request_from_request(
    request: Request,
    request_type: type[_RequestT],
    *,
    file_field: str,
    payload_key: str,
    max_part_size: int = BINARY_METADATA_MAX_PART_SIZE,
    **extra_fields: Any,
) -> _RequestT:
    """Parse a multipart request (metadata JSON part + one or more binary media parts)."""
    form = await request.form(max_part_size=max_part_size)
    metadata_upload = form.get("metadata")
    payload_field = form.get(file_field)

    metadata_text: str | None
    if isinstance(metadata_upload, (UploadFile, StarletteUploadFile)):
        metadata_text = await _read_metadata_upload(metadata_upload)
    elif isinstance(metadata_upload, str):
        metadata_text = metadata_upload
    else:
        metadata_text = None
    if metadata_text is None:
        raise HTTPException(status_code=400, detail="Missing binary inference metadata.")

    if file_field == "image":
        if not isinstance(payload_field, (UploadFile, StarletteUploadFile)):
            raise HTTPException(status_code=400, detail="Missing binary image payload.")
        extra_fields[payload_key] = await payload_field.read()
    else:
        files = form.getlist(file_field)
        if not files or not all(isinstance(file, (UploadFile, StarletteUploadFile)) for file in files):
            raise HTTPException(status_code=400, detail="Missing binary frame payload.")
        uploads: list[Any] = [f for f in files if isinstance(f, (UploadFile, StarletteUploadFile))]
        extra_fields[payload_key] = [await f.read() for f in uploads]

    return build_binary_request(request_type, metadata_text, **extra_fields)
