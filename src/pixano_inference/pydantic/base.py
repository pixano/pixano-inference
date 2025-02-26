# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic base models for request and response."""

from abc import ABC
from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class BaseRequest(BaseModel, ABC):
    """Base request model.

    Attributes:
        model: Name of the model.
    """

    model: str

    def to_base_model(self, base_model: type[T]) -> T:
        """Convert request to input type."""
        if not issubclass(base_model, BaseModel):
            raise ValueError(f"base_model must be a subclass of pydantic's BaseModel, got {base_model.__name__}.")
        return base_model.model_validate(self.model_dump(include=list(base_model.model_fields.keys())))


class APIRequest(BaseRequest, ABC):
    """API request model.

    Attributes:
        api_key: API key.
        secret_key: Secret key.
    """

    api_key: str
    secret_key: str


class CeleryTask(BaseModel):
    """Celery task model.

    Attributes:
        task_id: ID of the celery task.
        status: Status of the celery task.
    """

    id: str
    status: str


class BaseResponse(BaseModel, ABC):
    """Base response model.

    Attributes:
        task_id: ID of the celery task.
        status: Status of the celery task.
        timestamp: Timestamp of the response.
        processing_time: Processing time of the response.
        metadata: Metadata of the response.
        data: Data of the response.
    """

    id: str
    status: str
    timestamp: datetime
    processing_time: float = 0.0
    metadata: dict[str, Any]
    data: Any
