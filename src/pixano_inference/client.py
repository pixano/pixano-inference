# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pixano inference client."""

import asyncio
from typing import Any, Literal, cast, overload

import httpx
import requests  # type: ignore[import-untyped]
from celery import states
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from httpx import Response
from pydantic import field_validator

from .pydantic import (
    BaseRequest,
    BaseResponse,
    CeleryTask,
    ImageMaskGenerationRequest,
    ImageMaskGenerationResponse,
    ImageZeroShotDetectionRequest,
    ImageZeroShotDetectionResponse,
    ModelConfig,
    ModelInfo,
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


def _validate_task_id_asynchronous_request_response_type(
    task_id: str | None, asynchronous: bool, request: BaseRequest | None, response_type: type[BaseResponse] | None
) -> None:
    if not asynchronous and task_id is not None:
        raise ValueError("Task id must be None for synchronous calls.")
    elif not asynchronous and (request is None or response_type is None):
        raise ValueError("Request and response type must be provided for synchronous calls.")
    elif asynchronous and task_id is None and (request is None or response_type is None):
        raise ValueError("Either request and response_type or task id must be provided for asynchronous calls.")
    elif asynchronous and isinstance(task_id, str) and response_type is None:
        raise TypeError("Response type must be provided to retrieve results from asynchronous calls.")


def _validate_poll_interval_timeout(poll_interval: float, timeout: float) -> None:
    if not isinstance(poll_interval, (float, int)):
        raise ValueError("Poll interval should be a number that define an interval in seconds.")
    if not isinstance(timeout, (float, int)):
        raise ValueError("Timeout should be a number that define a number of seconds.")
    if poll_interval <= 0 or timeout <= 0 or poll_interval > timeout:
        raise ValueError("Poll interval and timeout should follow this rule (0 < poll interval <= timeout).")


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

    async def instantiate_model(self, provider: str, config: ModelConfig, timeout: int = 60) -> None:
        """Instantiate a model.

        Args:
            provider: The model provider.
            config: The configuration of the model.
            timeout: The timeout to wait for a response. Please note that even if the timeout is reached, the request
                will not be aborted.
        """
        json_content = jsonable_encoder({"provider": provider, "config": config})
        await self.post("providers/instantiate", json=json_content, timeout=timeout)
        return

    async def delete_model(self, model_name: str) -> None:
        """Delete a model.

        Args:
            model_name: The name of the model.
        """
        await self.delete(f"providers/model/{model_name}")

    @overload
    async def inference(
        self,
        route: str,
        request: BaseRequest | None,
        response_type: type[BaseResponse] | None,
        poll_interval: float,
        timeout: float,
        task_id: str,
        asynchronous: Literal[True],
    ) -> BaseResponse | CeleryTask: ...
    @overload
    async def inference(
        self,
        route: str,
        request: BaseRequest | None,
        response_type: type[BaseResponse] | None,
        poll_interval: float,
        timeout: float,
        task_id: None,
        asynchronous: Literal[True],
    ) -> CeleryTask: ...
    @overload
    async def inference(
        self,
        route: str,
        request: BaseRequest,
        response_type: type[BaseResponse],
        poll_interval: float,
        timeout: float,
        task_id: str | None,
        asynchronous: Literal[False],
    ) -> BaseResponse: ...
    @overload
    async def inference(
        self,
        route: str,
        request: BaseRequest | None,
        response_type: type[BaseResponse] | None,
        poll_interval: float,
        timeout: float,
        task_id: str | None,
        asynchronous: bool,
    ) -> BaseResponse | CeleryTask: ...
    async def inference(
        self,
        route: str,
        request: BaseRequest | None = None,
        response_type: type[BaseResponse] | None = None,
        poll_interval: float = 0.1,
        timeout: float = 60.0,
        task_id: str | None = None,
        asynchronous: bool = False,
    ) -> BaseResponse | CeleryTask:
        """Perform a POST request to the pixano inference server.

        Args:
            route: The root to the request.
            request: The request of the model.
            response_type: The type of the response.
            poll_interval: waiting time between subsequent requests to server to retrieve task results for synchronous
                requests.
            timeout: Time to wait for response for synchronous requests. If reached, the request will be aborted.
            task_id: The id of the task to poll for results.
            asynchronous: If True then the function will be called asynchronously and returns a CeleryTask object or
                poll results when task id is provided.

        Returns:
            A response from the pixano inference server.
        """
        _validate_task_id_asynchronous_request_response_type(
            task_id=task_id, asynchronous=asynchronous, request=request, response_type=response_type
        )
        _validate_poll_interval_timeout(poll_interval=poll_interval, timeout=timeout)

        if not asynchronous or task_id is None:
            request = cast(BaseRequest, request)
            celery_response: Response = await self.post(route, json=request.model_dump())
            celery_task: CeleryTask = CeleryTask.model_construct(**celery_response.json())

        # Asynchronous calls
        if asynchronous and task_id is None:
            return celery_task
        elif asynchronous and task_id is not None:
            response_type = cast(type[BaseResponse], response_type)
            has_slash = route.endswith("/")
            task_route = route + f"{'' if has_slash else '/'}{task_id}"
            response: dict[str, Any] = (await self.get(task_route)).json()
            if response["status"] == states.SUCCESS:
                return response_type.model_validate(response)
            return CeleryTask.model_construct(**response)

        # Synchronous calls with polling for result retrieval and deletion of celery tasks after timeout.
        response_type = cast(type[BaseResponse], response_type)

        time = 0.0
        has_slash = route.endswith("/")
        task_route = route + f"{'' if has_slash else '/'}{celery_task.id}"
        while time < timeout:
            response = (await self.get(task_route)).json()
            if response["status"] == states.SUCCESS:
                return response_type.model_validate(response)
            elif response["status"] == states.FAILURE:
                raise ValueError("The inference failed. Please check your inputs.")
            time += poll_interval
            await asyncio.sleep(poll_interval)
        await self.delete(task_route)
        raise InferenceTooLongError("The model is either busy or the task takes too long to perform.")

    @overload
    async def text_image_conditional_generation(
        self,
        request: TextImageConditionalGenerationRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: str,
        asynchronous: Literal[True],
    ) -> TextImageConditionalGenerationResponse | CeleryTask: ...
    @overload
    async def text_image_conditional_generation(
        self,
        request: TextImageConditionalGenerationRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: None,
        asynchronous: Literal[True],
    ) -> CeleryTask: ...
    @overload
    async def text_image_conditional_generation(
        self,
        request: TextImageConditionalGenerationRequest,
        poll_interval: float,
        timeout: float,
        task_id: str | None,
        asynchronous: Literal[False],
    ) -> TextImageConditionalGenerationResponse | CeleryTask: ...
    @overload
    async def text_image_conditional_generation(
        self,
        request: TextImageConditionalGenerationRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: str | None,
        asynchronous: bool,
    ) -> TextImageConditionalGenerationResponse | CeleryTask: ...
    async def text_image_conditional_generation(
        self,
        request: TextImageConditionalGenerationRequest | None = None,
        poll_interval: float = 0.1,
        timeout: float = 60,
        task_id: str | None = None,
        asynchronous: bool = False,
    ) -> TextImageConditionalGenerationResponse | CeleryTask:
        """Perform an inference to perform text-image conditional generation."""
        return await self.inference(
            route="tasks/multimodal/text-image/conditional_generation/",
            request=request,
            response_type=TextImageConditionalGenerationResponse,
            poll_interval=poll_interval,
            timeout=timeout,
            task_id=task_id,
            asynchronous=asynchronous,
        )

    @overload
    async def image_mask_generation(
        self,
        request: ImageMaskGenerationRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: str,
        asynchronous: Literal[True],
    ) -> ImageMaskGenerationResponse | CeleryTask: ...
    @overload
    async def image_mask_generation(
        self,
        request: ImageMaskGenerationRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: None,
        asynchronous: Literal[True],
    ) -> CeleryTask: ...
    @overload
    async def image_mask_generation(
        self,
        request: ImageMaskGenerationRequest,
        poll_interval: float,
        timeout: float,
        task_id: str | None,
        asynchronous: Literal[False],
    ) -> ImageMaskGenerationResponse: ...
    @overload
    async def image_mask_generation(
        self,
        request: ImageMaskGenerationRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: str | None,
        asynchronous: bool,
    ) -> ImageMaskGenerationResponse | CeleryTask: ...
    async def image_mask_generation(
        self,
        request: ImageMaskGenerationRequest | None = None,
        poll_interval: float = 0.1,
        timeout: float = 60,
        task_id: str | None = None,
        asynchronous: bool = False,
    ) -> ImageMaskGenerationResponse | CeleryTask:
        """Perform an inference to perform image mask generation."""
        return await self.inference(
            route="tasks/image/mask_generation/",
            request=request,
            response_type=ImageMaskGenerationResponse,
            poll_interval=poll_interval,
            timeout=timeout,
            task_id=task_id,
            asynchronous=asynchronous,
        )

    @overload
    async def video_mask_generation(
        self,
        request: VideoMaskGenerationRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: str,
        asynchronous: Literal[True],
    ) -> VideoMaskGenerationResponse | CeleryTask: ...
    @overload
    async def video_mask_generation(
        self,
        request: VideoMaskGenerationRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: None,
        asynchronous: Literal[True],
    ) -> CeleryTask: ...
    @overload
    async def video_mask_generation(
        self,
        request: VideoMaskGenerationRequest,
        poll_interval: float,
        timeout: float,
        task_id: str | None,
        asynchronous: Literal[False],
    ) -> VideoMaskGenerationResponse: ...
    @overload
    async def video_mask_generation(
        self,
        request: VideoMaskGenerationRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: str | None,
        asynchronous: bool,
    ) -> VideoMaskGenerationResponse | CeleryTask: ...
    async def video_mask_generation(
        self,
        request: VideoMaskGenerationRequest | None = None,
        poll_interval: float = 0.1,
        timeout: float = 60,
        task_id: str | None = None,
        asynchronous: bool = False,
    ) -> VideoMaskGenerationResponse | CeleryTask:
        """Perform an inference to perform video mask generation."""
        return await self.inference(
            route="tasks/video/mask_generation/",
            request=request,
            response_type=VideoMaskGenerationResponse,
            poll_interval=poll_interval,
            timeout=timeout,
            task_id=task_id,
            asynchronous=asynchronous,
        )

    @overload
    async def image_zero_shot_detection(
        self,
        request: ImageZeroShotDetectionRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: str,
        asynchronous: Literal[True],
    ) -> ImageZeroShotDetectionResponse | CeleryTask: ...
    @overload
    async def image_zero_shot_detection(
        self,
        request: ImageZeroShotDetectionRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: None,
        asynchronous: Literal[True],
    ) -> CeleryTask: ...
    @overload
    async def image_zero_shot_detection(
        self,
        request: ImageZeroShotDetectionRequest,
        poll_interval: float,
        timeout: float,
        task_id: str | None,
        asynchronous: Literal[False],
    ) -> ImageZeroShotDetectionResponse: ...
    @overload
    async def image_zero_shot_detection(
        self,
        request: ImageZeroShotDetectionRequest | None,
        poll_interval: float,
        timeout: float,
        task_id: str | None,
        asynchronous: bool,
    ) -> ImageZeroShotDetectionResponse | CeleryTask: ...
    async def image_zero_shot_detection(
        self,
        request: ImageZeroShotDetectionRequest | None = None,
        poll_interval: float = 0.1,
        timeout: float = 60,
        task_id: str | None = None,
        asynchronous: bool = False,
    ) -> ImageZeroShotDetectionResponse | CeleryTask:
        """Perform an inference to perform video mask generation."""
        return await self.inference(
            route="tasks/image/zero_shot_detection/",
            request=request,
            response_type=ImageZeroShotDetectionResponse,
            poll_interval=poll_interval,
            timeout=timeout,
            task_id=task_id,
            asynchronous=asynchronous,
        )
