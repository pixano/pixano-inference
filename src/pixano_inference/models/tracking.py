# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tracking model base class and I/O types."""

from abc import abstractmethod
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from pixano_inference.schemas.rle import CompressedRLE

from .base import InferenceModel


class TrackingInput(BaseModel):
    """Input for video mask generation / tracking.

    Attributes:
        video: Path to the video, list of frame paths, or base64 encoded video/frames.
        points: Point prompts [num_objects, num_points, 2].
        labels: Labels for points [num_objects, num_points].
        boxes: Box prompts [num_objects, 4].
        objects_ids: IDs of the objects to generate masks for.
        frame_indexes: Indexes of the frames where the objects are located.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    video: list[str] | list[Path] | str | Path
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


class TrackingOutput(BaseModel):
    """Output for video mask generation / tracking.

    Attributes:
        objects_ids: IDs of the objects.
        frame_indexes: Indexes of the frames where the objects are located.
        masks: Masks for the objects.
    """

    objects_ids: list[int]
    frame_indexes: list[int]
    masks: list[CompressedRLE]


class TrackingModel(InferenceModel):
    """Base class for video mask generation / tracking models.

    Example:
        ```python
        @register_model("my-tracker")
        class MyTracker(TrackingModel):
            def load_model(self):
                self.model = load_weights(self.config.model_params["path"])

            def predict(self, input: TrackingInput) -> TrackingOutput:
                ...
                return TrackingOutput(objects_ids=..., frame_indexes=..., masks=...)
        ```
    """

    @abstractmethod
    def predict(self, input: TrackingInput) -> TrackingOutput:
        """Run video mask generation / tracking.

        Args:
            input: Tracking input with video, prompts, and object IDs.

        Returns:
            Tracking output with objects_ids, frame_indexes, and masks.
        """
