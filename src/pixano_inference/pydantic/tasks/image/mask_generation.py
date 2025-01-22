# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for mask generation task."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from pixano_inference.pydantic.base import BaseRequest, BaseResponse
from pixano_inference.pydantic.data.vector_database import LanceVector
from pixano_inference.pydantic.nd_array import NDArrayFloat

from .utils import RLEMask


class ImageMaskGenerationInput(BaseModel):
    """Input for mask generation.

    Attributes:
        image: Image for mask generation.
        image_embedding: Image embedding for the mask generation.
        high_resolution_features: High resolution features for the mask generation.
        points: Points for the mask generation. The first fimension is the number of prompts the second
            the number of points per mask and the third the coordinates of the points.
        labels: Labels for the mask generation. The first fimension is the number of prompts, the second
            the number of labels per mask.
        boxes: Boxes for the mask generation. The first fimension is the number of prompts, the second
            the coordinates of the boxes.
        num_multimask_outputs: Number of masks to generate per prediction.
        multimask_output: Whether to generate multiple masks per prediction.
        return_image_embedding: Whether to return the image embeddings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: str | Path
    image_embedding: NDArrayFloat | LanceVector | None = None
    high_resolution_features: NDArrayFloat | LanceVector | None = None
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
    def _check_boxes(cls, v: list[list[list[int]]] | None) -> list[list[list[int]]] | None:
        if v is not None:
            for list_ in v:
                for box in list_:
                    if len(box) != 4:
                        raise ValueError("Each box should have 4 coordinates.")
        return v


class ImageMaskGenerationRequest(BaseRequest, ImageMaskGenerationInput):
    """Request for mask generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> ImageMaskGenerationInput:
        """Convert the request to the input."""
        return ImageMaskGenerationInput(
            image=self.image,
            points=self.points,
            boxes=self.boxes,
            num_multimask_outputs=self.num_multimask_outputs,
            multimask_output=self.multimask_output,
            image_embedding=self.image_embedding,
            high_resolution_features=self.high_resolution_features,
            return_image_embedding=self.return_image_embedding,
        )


class ImageMaskGenerationOutput(BaseModel):
    """Output for mask generation.

    Attributes:
        masks: Generated masks. The first dimension is the number of predictions, the second the number of masks per
            prediction.
        scores: Scores of the masks. The first dimension is the number of predictions, the second the number of masks
            per prediction.
        image_embedding: Image embeddings.
        high_resolution_features: High resolution features.
    """

    masks: list[list[RLEMask]]
    scores: NDArrayFloat
    image_embedding: NDArrayFloat | None = None
    high_resolution_features: NDArrayFloat | None = None


class ImageMaskGenerationResponse(BaseResponse):
    """Response for mask generation.

    Attributes:
        data: Output of the generation.
    """

    data: ImageMaskGenerationOutput
