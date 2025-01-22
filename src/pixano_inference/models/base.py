# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Base class for inference models."""

from abc import ABC, abstractmethod
from typing import Any

from pixano_inference.pydantic.tasks.image.mask_generation import ImageMaskGenerationOutput
from pixano_inference.pydantic.tasks.multimodal.conditional_generation import (
    TextImageConditionalGenerationOutput,
)
from pixano_inference.pydantic.tasks.video.mask_generation import VideoMaskGenerationOutput


class BaseInferenceModel(ABC):
    """Base class for inference models."""

    def __init__(self, name: str, provider: str):
        """Initialize the model.

        Args:
            name: Name of the model.
            provider: Provider of the model.
        """
        self.name = name
        self.provider = provider

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return the metadata of the model."""
        ...

    @abstractmethod
    def delete(self):
        """Delete the model."""
        ...

    def image_mask_generation(self, *args: Any, **kwargs) -> ImageMaskGenerationOutput:
        """Generate a mask from the image."""
        raise NotImplementedError("This model does not support image mask generation.")

    def text_image_conditional_generation(self, *args: Any, **kwargs) -> TextImageConditionalGenerationOutput:
        """Generate text from an image and a prompt."""
        raise NotImplementedError("This model does not support text-image conditional generation.")

    def video_mask_generation(self, *args: Any, **kwargs) -> VideoMaskGenerationOutput:
        """Generate a mask from the video."""
        raise NotImplementedError("This model does not support video mask generation.")
