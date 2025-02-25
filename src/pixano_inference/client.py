# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pixano inference client."""

import asyncio
from typing import Any, Literal

import httpx
import requests  # type: ignore[import-untyped]
from celery import states
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from httpx import Response
from pydantic import field_validator

from .pydantic import BaseRequest, BaseResponse, CeleryTask
from .pydantic.models import ModelConfig, ModelInfo
from .pydantic.tasks import (
    ImageMaskGenerationRequest,
    ImageMaskGenerationResponse,
    TextImageConditionalGenerationRequest,
    TextImageConditionalGenerationResponse,
    VideoMaskGenerationRequest,
    VideoMaskGenerationResponse,
)
from .settings import Settings
from .utils import is_url


class InferenceTooLongError(Exception):
    """Exeption when inference took too long."""

    pass


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
        settings = Settings.model_validate(requests.get(f"{url}/app/settings/").json())
        client = PixanoInferenceClient(url=url, **settings.model_dump())
        return client

    async def _rest_call(self, path: str, method: Literal["GET", "POST", "PUT", "DELETE"], **kwargs) -> Response:
        """Perform a REST call to the pixano inference server."""
        async with httpx.AsyncClient(timeout=60) as client:
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

    async def get(self, path: str) -> Response:
        """Perform a GET request to the pixano inference server.

        Args:
            path: The path of the request.
        """
        return await self._rest_call(path=path, method="GET")

    async def post(self, path: str, **kwargs: Any) -> Response:
        """Perform a POST request to the pixano inference server.

        Args:
            path: The path of the request.
            kwargs: The keyword arguments to pass to the request.
        """
        return await self._rest_call(path=path, method="POST", **kwargs)

    async def put(self, path: str, **kwargs: Any) -> Response:
        """Perform a PUT request to the pixano inference server.

        Args:
            path: The path of the request.
            kwargs: The keyword arguments to pass to the request.
        """
        return await self._rest_call(path=path, method="PUT", **kwargs)

    async def delete(self, path: str) -> Response:
        """Perform a DELETE request to the pixano inference server.

        Args:
            path: The path of the request.
        """
        return await self._rest_call(path=path, method="DELETE")

    async def list_models(self) -> list[ModelInfo]:
        """List all models."""
        response = await self.get("app/models/")
        return [ModelInfo.model_construct(**model) for model in response.json()]

    async def instantiate_model(self, provider: str, config: ModelConfig) -> None:
        """Instantiate a model.

        Args:
            provider: The model provider.
            config: The configuration of the model.
        """
        json_content = jsonable_encoder({"provider": provider, "config": config})
        await self.post("providers/instantiate", json=json_content)
        return

    async def delete_model(self, model_name: str) -> None:
        """Delete a model.

        Args:
            model_name: The name of the model.
        """
        await self.delete(f"providers/model/{model_name}")

    async def inference(
        self,
        route: str,
        request: BaseRequest,
        response_type: type[BaseResponse],
        poll_interval: float = 0.1,
        timeout: float = 60,
    ) -> BaseResponse:
        """Perform a POST request to the pixano inference server.

        Args:
            route: The root to the request.
            request: The request of the model.
            response_type: The type of the response.
            poll_interval: waiting time between subsequent requests to server to retrieve task results.
            timeout: Time to wait for response.

        Returns:
            A response from the pixano inference server.
        """
        celery_response = await self.post(route, json=request.model_dump())
        celery_task = CeleryTask.model_construct(**celery_response.json())
        time = 0.0
        has_slash = route.endswith("/")
        task_route = route + f"{'' if has_slash else '/'}{celery_task.id}"
        while time < timeout:
            response: dict[str, Any] = (await self.get(task_route)).json()
            if response["status"] == states.SUCCESS:
                return response_type.model_validate(response)
            elif response["status"] == states.FAILURE:
                raise ValueError("The inference failed. Please check your inputs.")
            time += poll_interval
            await asyncio.sleep(poll_interval)
        await self.delete(task_route)
        raise InferenceTooLongError("The model is either busy or the task takes too long to perform.")

    async def text_image_conditional_generation(
        self, request: TextImageConditionalGenerationRequest
    ) -> TextImageConditionalGenerationResponse:
        """Perform an inference to perform text-image conditional generation."""
        return await self.inference(
            route="tasks/multimodal/text-image/conditional_generation/",
            request=request,
            response_type=TextImageConditionalGenerationResponse,
        )

    async def image_mask_generation(self, request: ImageMaskGenerationRequest) -> ImageMaskGenerationResponse:
        """Perform an inference to perform image mask generation."""
        return await self.inference(
            route="tasks/image/mask_generation/", request=request, response_type=ImageMaskGenerationResponse
        )

    async def video_mask_generation(self, request: VideoMaskGenerationRequest) -> VideoMaskGenerationResponse:
        """Perform an inference to perform video mask generation."""
        return await self.inference(
            route="tasks/video/mask_generation/", request=request, response_type=VideoMaskGenerationResponse
        )
