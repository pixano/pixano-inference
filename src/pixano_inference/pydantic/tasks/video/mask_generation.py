# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for mask generation task."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from pixano_inference.pydantic.base import BaseRequest, BaseResponse
from pixano_inference.pydantic.tasks.image.utils import CompressedRLE


class VideoMaskGenerationInput(BaseModel):
    """Input for mask generation.

    Attributes:
        video: Can be a path to the video or list of paths to the frames or a base64 encoded video or a list of base64
            encoded frames.
        points: Points for the mask generation. The first fimension is the number of objects the second
            the number of points for each object and the third the coordinates of the points.
        labels: Labels for the mask generation. The first fimension is the number of objects, the second
            the number of labels for each object.
        boxes: Boxes for the mask generation. The first fimension is the number of objects, the second
            the coordinates of the boxes.
        objects_ids: IDs of the objects to generate masks for.
        frame_indexes: Indexes of the frames where the objects are located.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    video: str | Path | list[str] | list[Path]
    points: list[list[list[int]]] | None = None
    labels: list[list[int]] | None = None
    boxes: list[list[int]] | None = None
    objects_ids: list[int]
    frame_indexes: list[int]

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

    @field_validator("objects_ids")
    @classmethod
    def _check_objects_ids(cls, v: list[int]) -> list[int]:
        if len(v) == 0:
            raise ValueError("At least one object ID should be provided.")
        elif len(v) != len(set(v)):
            raise ValueError("Object IDs should be unique.")
        return v


class VideoMaskGenerationRequest(BaseRequest, VideoMaskGenerationInput):
    """Request for mask generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> VideoMaskGenerationInput:
        """Convert the request to the input."""
        return self.to_base_model(VideoMaskGenerationInput)


class VideoMaskGenerationOutput(BaseModel):
    """Output for mask generation.

    Attributes:
        objects_ids: IDs of the objects.
        frame_indexes: Indexes of the frames where the objects are located.
        masks: Masks for the objects.
    """

    objects_ids: list[int]
    frame_indexes: list[int]
    masks: list[CompressedRLE]


class VideoMaskGenerationResponse(BaseResponse):
    """Response for mask generation.

    Attributes:
        data: Output of the generation.
    """

    data: VideoMaskGenerationOutput
