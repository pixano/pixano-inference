# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for image mask generation task."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from pixano_inference.pydantic.base import BaseRequest, BaseResponse
from pixano_inference.pydantic.data.vector_database import LanceVector
from pixano_inference.pydantic.nd_array import NDArrayFloat

from .utils import CompressedRLE


class ImageMaskGenerationInput(BaseModel):
    """Input for image mask generation.

    Attributes:
        image: Image for image mask generation.
        image_embedding: Image embedding for the image mask generation.
        high_resolution_features: High resolution features for the image mask generation.
        reset_predictor: True (default) for a new image. If False, keep current predictor if available.
        points: Points for the image mask generation. The first fimension is the number of prompts the second
            the number of points per mask and the third the coordinates of the points.
        labels: Labels for the image mask generation. The first fimension is the number of prompts, the second
            the number of labels per mask.
        boxes: Boxes for the image mask generation. The first fimension is the number of prompts, the second
            the coordinates of the boxes.
        num_multimask_outputs: Number of masks to generate per prediction.
        multimask_output: Whether to generate multiple masks per prediction.
        return_image_embedding: Whether to return the image embeddings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: str | Path
    image_embedding: NDArrayFloat | LanceVector | None = None
    high_resolution_features: list[NDArrayFloat] | list[LanceVector] | None = None
    reset_predictor: bool = True
    points: list[list[list[int]]] | None = None
    labels: list[list[int]] | None = None
    boxes: list[list[int]] | None = None
    num_multimask_outputs: int = 3
    multimask_output: bool = True
    return_image_embedding: bool = False

    @field_validator("points")
    @classmethod
    def _check_points(cls, v: list[list[list[int]]] | None) -> list[list[list[int]]] | None:
        if v is not None:
            for list_ in v:
                for point in list_:
                    if len(point) != 2:
                        raise ValueError("Each point should have 2 coordinates.")
        return v

    @field_validator("boxes")
    @classmethod
    def _check_boxes(cls, v: list[list[int]] | None) -> list[list[int]] | None:
        if v is not None:
            for box in v:
                if len(box) != 4:
                    raise ValueError("Each box should have 4 coordinates.")
        return v


class ImageMaskGenerationRequest(BaseRequest, ImageMaskGenerationInput):
    """Request for image mask generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> ImageMaskGenerationInput:
        """Convert the request to the input."""
        return self.to_base_model(ImageMaskGenerationInput)


class ImageMaskGenerationOutput(BaseModel):
    """Output for image mask generation.

    Attributes:
        masks: Generated masks. The first dimension is the number of predictions, the second the number of masks per
            prediction.
        scores: Scores of the masks. The first dimension is the number of predictions, the second the number of masks
            per prediction.
        image_embedding: Image embeddings.
        high_resolution_features: High resolution features.
    """

    masks: list[list[CompressedRLE]]
    scores: NDArrayFloat
    image_embedding: NDArrayFloat | None = None
    high_resolution_features: list[NDArrayFloat] | None = None


class ImageMaskGenerationResponse(BaseResponse):
    """Response for image mask generation.

    Attributes:
        data: Output of the generation.
    """

    data: ImageMaskGenerationOutput
