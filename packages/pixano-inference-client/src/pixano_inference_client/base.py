# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic base models for request and response."""

from abc import ABC
from datetime import datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


T = TypeVar("T", bound=BaseModel)


class CamelModel(BaseModel):
    """Base model whose fields serialize as camelCase on the wire.

    Fields stay snake_case in Python; ``populate_by_name=True`` means both the camelCase
    alias and the snake_case name are accepted on input, so existing Python callers keep
    working while the HTTP contract (and generated TypeScript client) is camelCase.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, arbitrary_types_allowed=True)


class BaseRequest(CamelModel, ABC):
    """Base request model.

    Attributes:
        model: Name of the model.
    """

    model: str

    def to_base_model(self, base_model: type[T]) -> T:
        """Convert request to input type."""
        if not issubclass(base_model, BaseModel):
            raise ValueError(f"base_model must be a subclass of pydantic's BaseModel, got {base_model.__name__}.")
        return base_model.model_validate(self.model_dump(include=set(base_model.model_fields.keys())))


class BaseResponse(CamelModel, ABC):
    """Base response envelope.

    Attributes:
        id: ID of the task.
        status: Status of the task.
        timestamp: Timestamp of the response.
        processing_time: Processing time of the response.
        metadata: Metadata of the response.
        data: Task-specific output payload.
    """

    id: str
    status: str
    timestamp: datetime
    processing_time: float = 0.0
    metadata: dict[str, Any]
    data: Any
