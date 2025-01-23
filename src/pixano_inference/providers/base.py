# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Base classes for providers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pixano_inference.models.base import BaseInferenceModel
from pixano_inference.pydantic.base import APIRequest, BaseResponse
from pixano_inference.pydantic.tasks.image.mask_generation import ImageMaskGenerationOutput, ImageMaskGenerationRequest
from pixano_inference.pydantic.tasks.multimodal.conditional_generation import (
    TextImageConditionalGenerationOutput,
    TextImageConditionalGenerationRequest,
)
from pixano_inference.pydantic.tasks.video.mask_generation import (
    VideoMaskGenerationRequest,
    VideoMaskGenerationResponse,
)
from pixano_inference.settings import PIXANO_INFERENCE_SETTINGS, Settings
from pixano_inference.tasks.task import Task
from pixano_inference.utils.package import is_torch_installed


if is_torch_installed():
    import torch


class BaseProvider(ABC):
    """Base class for providers."""

    def __init__(self, **kwargs: Any):
        """Initialize the provider."""
        pass

    @torch.inference_mode()
    def image_mask_generation(
        self, request: ImageMaskGenerationRequest, model: BaseInferenceModel, *args: Any, **kwargs: Any
    ) -> ImageMaskGenerationOutput:
        """Generate a mask from the image.

        Args:
            request: Request for the generation.
            model: Model to use for the generation.
            args: Additional arguments.
            kwargs: Additional keyword arguments.
        """
        raise NotImplementedError("This provider does not support image mask generation.")

    @torch.inference_mode()
    def text_image_conditional_generation(
        self, request: TextImageConditionalGenerationRequest, model: BaseInferenceModel, *args: Any, **kwargs: Any
    ) -> TextImageConditionalGenerationOutput:
        """Generate an image from the text and image.

        Args:
            request: Request for the generation.
            model: Model to use for the generation.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Output of the generation
        """
        raise NotImplementedError("This provider does not support text-image conditional generation.")

    @torch.inference_mode()
    def video_mask_generation(
        self, request: VideoMaskGenerationRequest, model: BaseInferenceModel, *args: Any, **kwargs: Any
    ) -> VideoMaskGenerationResponse:
        """Generate a mask from the video.

        Args:
            request: Request for the generation.
            model: Model to use for the generation.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Output of the generation
        """
        raise NotImplementedError("This provider does not support video mask generation.")


class APIProvider(BaseProvider):
    """Base class for API providers."""

    @property
    @abstractmethod
    def api_url(self) -> str:
        """URL of the API."""
        ...

    @abstractmethod
    def send_request(self, request: APIRequest) -> BaseResponse:
        """Send a request to the API and return the response.

        Args:
            request: Request to send.

        Returns:
            Response from the API.
        """
        ...


class ModelProvider(BaseProvider):
    """Base class for model providers."""

    @abstractmethod
    def load_model(
        self,
        name: str,
        task: Task | str,
        settings: Settings = PIXANO_INFERENCE_SETTINGS,
        path: Path | str | None = None,
        processor_config: dict = {},
        config: dict = {},
    ) -> Any:
        """Load the model from the provider.

        Args:
            name: Name of the model.
            task: Task of the model.
            settings: Settings to use for the provider.
            path: Path to the model.
            processor_config: Processor configuration.
            config: Configuration for the model.

        Returns:
            The loaded model.
        """
        ...
