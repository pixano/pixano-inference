# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for image zero shot detection task."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pixano_inference.pydantic.base import BaseRequest, BaseResponse


class ImageZeroShotDetectionInput(BaseModel):
    """Input for image zero shot detection.

    Attributes:
        image: Image for image zero shot detection.
        classes: List of classes to detect.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: str | Path
    classes: str | list[str]
    box_threshold: float = 0.5
    text_threshold: float = 0.5


class ImageZeroShotDetectionRequest(BaseRequest, ImageZeroShotDetectionInput):
    """Request for image zero shot detection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> ImageZeroShotDetectionInput:
        """Convert the request to the input."""
        return self.to_base_model(ImageZeroShotDetectionInput)


class ImageZeroShotDetectionOutput(BaseModel):
    """Output for image zero shot detection.

    Attributes:
        boxes: List of boxes detected by the model.
        scores: List of scores detected by the model (higher is better).
        classes: List of class names associated with each box and score.
    """

    boxes: list[list[int]]
    scores: list[float]
    classes: list[str]


class ImageZeroShotDetectionResponse(BaseResponse):
    """Response for image zero shot detection.

    Attributes:
        data: Output of the generation.
    """

    data: ImageZeroShotDetectionOutput
