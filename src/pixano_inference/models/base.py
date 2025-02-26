# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Base class for inference models."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pixano_inference.pydantic import (
    ImageMaskGenerationOutput,
    ImageZeroShotDetectionOutput,
    TextImageConditionalGenerationOutput,
    VideoMaskGenerationOutput,
)


class ModelStatus(Enum):
    """Current status of the model.

    Attributes:
    - IDLE: waiting for an input.
    - RUNNING: computing.
    """

    IDLE = 0
    RUNNING = 1


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
        self._status = ModelStatus.IDLE

    @property
    def status(self) -> ModelStatus:
        """Get the status of the model."""
        return self._status

    @status.setter
    def status(self, new_status: ModelStatus):
        """Set the status of the model.

        It should be handled outside of the Inference Model by a controller to handle requests sequentially.
        """
        if not isinstance(new_status, ModelStatus):
            raise ValueError(f"Status should be a ModelStatus, got {new_status}.")
        self._status = new_status

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

    def image_zero_shot_detection(self, *args: Any, **kwargs) -> ImageZeroShotDetectionOutput:
        """Perform zero shot detection on an image."""
        raise NotImplementedError("This model does not support image zero shot detection.")
