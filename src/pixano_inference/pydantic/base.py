# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic base models for request and response."""

from abc import ABC
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class BaseRequest(BaseModel):
    """Base request model.

    Attributes:
        provider: Name of the provider.
        model: Name of the model.
    """

    provider: str
    model: str


class APIRequest(BaseRequest):
    """API request model.

    Attributes:
        api_key: API key.
        secret_key: Secret key.
    """

    api_key: str
    secret_key: str


class BaseResponse(BaseModel, ABC):
    """Base response model.

    Attributes:
        timestamp: Timestamp of the response.
        processing_time: Processing time of the response.
        metadata: Metadata of the response.
        data: Data of the response.
    """

    timestamp: datetime
    processing_time: float = 0.0
    metadata: dict[str, Any]
    data: Any
