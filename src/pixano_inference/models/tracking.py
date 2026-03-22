# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tracking model base class and I/O types."""

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from pixano_inference.schemas.rle import CompressedRLE

from .base import InferenceModel


class TrackingPointPrompt(BaseModel):
    """Point prompt for a tracking keyframe."""

    x: int
    y: int
    label: Literal[0, 1]


class TrackingBoxPrompt(BaseModel):
    """Box prompt for a tracking keyframe."""

    x: int
    y: int
    width: int
    height: int


class TrackingInterval(BaseModel):
    """Optional propagation window relative to the provided video frames."""

    start_frame: int
    end_frame: int
    direction: Literal["forward", "backward"] = "forward"


class TrackingKeyframe(BaseModel):
    """Prompt payload for a single tracking keyframe."""

    frame_index: int
    points: list[TrackingPointPrompt] | None = None
    box: TrackingBoxPrompt | None = None
    mask: CompressedRLE | None = None

    @model_validator(mode="after")
    def _check_prompt_payload(self) -> "TrackingKeyframe":
        has_points_or_box = bool(self.points) or self.box is not None
        if self.mask is not None and has_points_or_box:
            raise ValueError("Keyframe prompts must use either points/box or a mask, not both.")
        if self.mask is None and not has_points_or_box:
            raise ValueError("Keyframe prompts require points, a box, or a mask.")
        return self


class TrackingInput(BaseModel):
    """Input for video mask generation / tracking.

    Attributes:
        video: Path to the video, list of frame paths, or base64 encoded video/frames.
        points: Legacy point prompts [num_objects, num_points, 2].
        labels: Legacy labels for points [num_objects, num_points].
        boxes: Legacy box prompts [num_objects, 4].
        propagate: Whether to propagate masks beyond the prompted frames.
        interval: Optional propagation interval relative to the provided frame window.
        keyframes: Optional structured prompt payloads for each object.
        objects_ids: IDs of the objects to generate masks for.
        frame_indexes: Indexes of the prompted frames.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    video: list[str | Path | bytes] | str | Path | bytes
    points: list[list[list[int]]] | None = None
    labels: list[list[int]] | None = None
    boxes: list[list[int]] | None = None
    propagate: bool | None = None
    interval: TrackingInterval | None = None
    keyframes: list[TrackingKeyframe] | None = None
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
        if len(v) != len(set(v)):
            raise ValueError("Object IDs should be unique.")
        return v

    @model_validator(mode="after")
    def _check_keyframe_count(self) -> "TrackingInput":
        if self.keyframes is not None and len(self.keyframes) != len(self.objects_ids):
            raise ValueError("When keyframes are provided, there must be exactly one keyframe per object ID.")
        return self


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

    capability_name: ClassVar[str] = "tracking"

    @abstractmethod
    def predict(self, input: TrackingInput) -> TrackingOutput:
        """Run video mask generation / tracking.

        Args:
            input: Tracking input with video, prompts, and object IDs.

        Returns:
            Tracking output with objects_ids, frame_indexes, and masks.
        """
