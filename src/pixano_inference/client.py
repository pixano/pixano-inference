# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pixano inference client."""

from typing import Any, Literal

import httpx
import requests  # type: ignore[import-untyped]
from fastapi import HTTPException
from httpx import Response
from pydantic import field_validator

from .schemas import (
    BaseRequest,
    BaseResponse,
    DetectionRequest,
    DetectionResponse,
    ModelInfo,
    SegmentationRequest,
    SegmentationResponse,
    TrackingRequest,
    TrackingResponse,
    VLMRequest,
    VLMResponse,
)
from .settings import Settings
from .utils import is_url


def raise_if_error(response: Response) -> None:
    """Raise an error from a response."""
    if response.is_success:
        return
    error_out = f"HTTP {response.status_code}: {response.reason_phrase}"
    try:
        json_detail = response.json()
    except Exception:
        json_detail = {}

    detail = json_detail.get("detail", None)
    if detail is not None:
        error_out += f" - {detail}"
    error = json_detail.get("error", None)
    if error is not None:
        error_out += f" - {error}"
    raise HTTPException(response.status_code, detail=error_out)


class PixanoInferenceClient(Settings):
    """Pixano Inference Client."""

    url: str

    @field_validator("url", mode="after")
    def _validate_url(cls, v):
        if not is_url(v):
            raise ValueError(f"Invalid URL, got '{v}'.")
        if v.endswith("/"):
            v = v[:-1]
        return v

    @staticmethod
    def connect(url: str) -> "PixanoInferenceClient":
        """Connect to pixano inference.

        Args:
            url: The URL of the pixano inference server.
        """
        server_settings = requests.get(f"{url}/app/settings/").json()
        server_settings = {k: v for k, v in server_settings.items() if v is not None}
        client = PixanoInferenceClient(url=url, **server_settings)
        return client

    async def _rest_call(
        self,
        path: str,
        method: Literal["GET", "POST", "PUT", "DELETE"],
        timeout: int = 60,
        **kwargs,
    ) -> Response:
        """Perform a REST call to the pixano inference server."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            match method:
                case "GET":
                    request_fn = client.get
                case "POST":
                    request_fn = client.post
                case "PUT":
                    request_fn = client.put
                case "DELETE":
                    request_fn = client.delete
                case _:
                    raise ValueError(
                        f"Invalid REST call method. Expected one of ['GET', 'POST', 'PUT', 'DELETE'], but got "
                        f"'{method}'."
                    )

            if path.startswith("/"):
                path = path[1:]

            url = f"{self.url}/{path}"
            response = await request_fn(url, **kwargs)
            raise_if_error(response)

            return response

    async def get_settings(self) -> Settings:
        """Get the settings for the pixano inference server."""
        response = await self.get("app/settings/")
        raise_if_error(response)
        return Settings(**response.json())

    async def get(self, path: str, **kwargs: Any) -> Response:
        """Perform a GET request to the pixano inference server.

        Args:
            path: The path of the request.
            kwargs: The keyword arguments to pass to the request or httpx client.
        """
        return await self._rest_call(path=path, method="GET", **kwargs)

    async def post(self, path: str, **kwargs: Any) -> Response:
        """Perform a POST request to the pixano inference server.

        Args:
            path: The path of the request.
            kwargs: The keyword arguments to pass to the request or httpx client.
        """
        return await self._rest_call(path=path, method="POST", **kwargs)

    async def put(self, path: str, **kwargs: Any) -> Response:
        """Perform a PUT request to the pixano inference server.

        Args:
            path: The path of the request.
            kwargs: The keyword arguments to pass to the request or httpx client.
        """
        return await self._rest_call(path=path, method="PUT", **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> Response:
        """Perform a DELETE request to the pixano inference server.

        Args:
            path: The path of the request.
            kwargs: The keyword arguments to pass to the request or httpx client.
        """
        return await self._rest_call(path=path, method="DELETE", **kwargs)

    async def list_models(self) -> list[ModelInfo]:
        """List all models."""
        response = await self.get("app/models/")
        return [ModelInfo.model_construct(**model) for model in response.json()]

    async def inference(
        self,
        route: str,
        request: BaseRequest,
        response_type: type[BaseResponse],
    ) -> BaseResponse:
        """Perform inference via a POST request to the pixano inference server.

        Args:
            route: The route for the request.
            request: The request payload.
            response_type: The expected response type.

        Returns:
            The parsed response from the server.
        """
        response = await self.post(route, json=request.model_dump())
        return response_type.model_validate(response.json())

    async def segmentation(
        self,
        request: SegmentationRequest,
    ) -> SegmentationResponse:
        """Perform an inference to perform image segmentation."""
        return await self.inference(
            route="inference/segmentation/",
            request=request,
            response_type=SegmentationResponse,
        )

    async def tracking(
        self,
        request: TrackingRequest,
    ) -> TrackingResponse:
        """Perform an inference to perform video tracking."""
        return await self.inference(
            route="inference/tracking/",
            request=request,
            response_type=TrackingResponse,
        )

    async def vlm(
        self,
        request: VLMRequest,
    ) -> VLMResponse:
        """Perform an inference for vision-language model generation."""
        return await self.inference(
            route="inference/vlm/",
            request=request,
            response_type=VLMResponse,
        )

    async def detection(
        self,
        request: DetectionRequest,
    ) -> DetectionResponse:
        """Perform an inference to perform zero-shot detection."""
        return await self.inference(
            route="inference/detection/",
            request=request,
            response_type=DetectionResponse,
        )
