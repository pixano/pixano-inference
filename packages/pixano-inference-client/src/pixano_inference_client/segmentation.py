# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Segmentation I/O types."""

from pathlib import Path

from pydantic import field_validator

from .base import CamelModel
from .nd_array import NDArrayFloat
from .rle import CompressedRLE


class SegmentationInput(CamelModel):
    """Input for image segmentation.

    Attributes:
        image: Image for segmentation (path, URL, or base64).
        image_embedding: Pre-computed image embedding.
        high_resolution_features: High resolution features.
        mask_input: Low-resolution mask logits from a previous refinement round.
        reset_predictor: True (default) for a new image. If False, keep current predictor.
        points: Point prompts [num_prompts, num_points, 2].
        labels: Labels for points [num_prompts, num_points].
        boxes: Box prompts [num_prompts, 4].
        num_multimask_outputs: Number of masks to generate per prediction.
        multimask_output: Whether to generate multiple masks per prediction.
        return_image_embedding: Whether to return the image embeddings.
        return_logits: Whether to return low-resolution logits for iterative refinement.
    """

    image: str | Path | bytes
    image_embedding: NDArrayFloat | None = None
    high_resolution_features: list[NDArrayFloat] | None = None
    mask_input: NDArrayFloat | None = None
    reset_predictor: bool = True
    points: list[list[list[int]]] | None = None
    labels: list[list[int]] | None = None
    boxes: list[list[int]] | None = None
    num_multimask_outputs: int = 3
    multimask_output: bool = True
    return_image_embedding: bool = False
    return_logits: bool = False

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


class SegmentationOutput(CamelModel):
    """Output for image segmentation.

    Attributes:
        masks: Generated masks [num_predictions, num_masks_per_prediction].
        scores: Scores of the masks.
        image_embedding: Image embeddings (optional).
        high_resolution_features: High resolution features (optional).
        mask_logits: Low-resolution logits for iterative refinement (optional).
    """

    masks: list[list[CompressedRLE]]
    scores: NDArrayFloat
    image_embedding: NDArrayFloat | None = None
    high_resolution_features: list[NDArrayFloat] | None = None
    mask_logits: NDArrayFloat | None = None
