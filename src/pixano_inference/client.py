# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pixano inference client."""

from typing import Any, Literal

import requests  # type: ignore[import-untyped]
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import field_validator
from requests import Response

from .pydantic import BaseRequest, BaseResponse
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


def raise_if_error(response: Response) -> None:
    """Raise an error from a response."""
    if response.ok:
        return
    error_out = f"HTTP {response.status_code}: {response.reason}"
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
        settings = Settings.model_validate(requests.get(f"{url}/app/settings").json())
        client = PixanoInferenceClient(url=url, **settings.model_dump())
        return client

    def _rest_call(self, path: str, method: Literal["GET", "POST", "PUT", "DELETE"], **kwargs) -> Response:
        """Perform a REST call to the pixano inference server."""
        match method:
            case "GET":
                request_fn = requests.get
            case "POST":
                request_fn = requests.post
            case "PUT":
                request_fn = requests.put
            case "DELETE":
                request_fn = requests.delete
            case _:
                raise ValueError(
                    f"Invalid REST call method. Expected one of ['GET', 'POST', 'PUT', 'DELETE'], but got '{method}'."
                )

        url = f"{self.url}/{path}"
        response = request_fn(url, **kwargs)
        raise_if_error(response)

        return response

    def get_settings(self) -> Settings:
        """Get the settings for the pixano inference server."""
        response = self.get("app/settings")
        raise_if_error(response)
        return Settings(**response.json())

    def get(self, path: str) -> Response:
        """Perform a GET request to the pixano inference server.

        Args:
            path: The path of the request.
        """
        return self._rest_call(path=path, method="GET")

    def post(self, path: str, **kwargs: Any) -> Response:
        """Perform a POST request to the pixano inference server.

        Args:
            path: The path of the request.
            kwargs: The keyword arguments to pass to the request.
        """
        return self._rest_call(path=path, method="POST", **kwargs)

    def put(self, path: str, **kwargs: Any) -> Response:
        """Perform a PUT request to the pixano inference server.

        Args:
            path: The path of the request.
            kwargs: The keyword arguments to pass to the request.
        """
        return self._rest_call(path=path, method="PUT", **kwargs)

    def delete(self, path: str) -> Response:
        """Perform a DELETE request to the pixano inference server.

        Args:
            path: The path of the request.
        """
        return self._rest_call(path=path, method="DELETE")

    def list_models(self) -> list[ModelInfo]:
        """List all models."""
        response = self.get("app/models")
        return [ModelInfo.model_validate(model) for model in response.json()]

    def instantiate_model(self, provider: str, config: ModelConfig) -> None:
        """Instantiate a model.

        Args:
            provider: The model provider.
            config: The configuration of the model.
        """
        json_content = jsonable_encoder({"provider": provider, "config": config})
        self.post("providers/instantiate", json=json_content)
        return

    def delete_model(self, model_name: str) -> None:
        """Delete a model.

        Args:
            model_name: The name of the model.
        """
        self.delete(f"providers/model/{model_name}")

    def inference(self, url: str, request: BaseRequest, response_type: type[BaseResponse]) -> BaseResponse:
        """Perform a POST request to the pixano inference server.

        Args:
            url: The path of the request.
            request: The request of the model.
            response_type: The type of the response.

        Returns:
            A response from the pixano inference server.
        """
        return response_type.model_validate(self.post(url, json=request.model_dump()).json())

    def text_image_conditional_generation(
        self, request: TextImageConditionalGenerationRequest
    ) -> TextImageConditionalGenerationResponse:
        """Perform an inference to perform text-image conditional generation."""
        return self.inference(
            url="/tasks/multimodal/image-text/conditional_generation/",
            request=request,
            response_type=TextImageConditionalGenerationResponse,
        )

    def image_mask_generation(self, request: ImageMaskGenerationRequest) -> ImageMaskGenerationResponse:
        """Perform an inference to perform image mask generation."""
        return self.inference(
            url="/tasks/image/mask_generation", request=request, response_type=ImageMaskGenerationResponse
        )

    def video_mask_generation(self, request: VideoMaskGenerationRequest) -> VideoMaskGenerationResponse:
        """Perform an inference to perform video mask generation."""
        return self.inference(
            url="/tasks/video/mask_generation", request=request, response_type=VideoMaskGenerationResponse
        )
