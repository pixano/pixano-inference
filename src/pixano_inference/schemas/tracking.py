# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tracking I/O types.

A tracking model follows objects through the frames of a video. It is either *prompted* (the
request names the objects with points, boxes or masks, as SAM2 expects) or *prompt-free* (tracking
by detection, as in ByteTrack: the model detects the objects and creates the tracks itself). Both
return the same :class:`TrackingOutput`: for each frame, the objects tracked in it.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .base import _BaseModel
from .rle import CompressedRLE


class TrackingPointPrompt(_BaseModel):
    """Point prompt for a tracking keyframe."""

    x: int
    y: int
    label: Literal[0, 1]


class TrackingBoxPrompt(_BaseModel):
    """Box prompt for a tracking keyframe."""

    x: int
    y: int
    width: int
    height: int


class TrackingInterval(_BaseModel):
    """Optional propagation window relative to the provided video frames."""

    start_frame: int
    end_frame: int
    direction: Literal["forward", "backward"] = "forward"


class TrackingKeyframe(_BaseModel):
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


class TrackingInput(_BaseModel):
    """Input for video tracking.

    A prompted request names the objects to track: ``objects_ids`` with one keyframe (or one legacy
    point/box prompt) per object. A prompt-free request gives only the video, and optionally
    ``classes`` and ``box_threshold``: the model detects the objects and assigns the track ids.

    Attributes:
        video: Path to the video, list of frame paths, or base64 encoded video/frames.
        points: Legacy point prompts [num_objects, num_points, 2].
        labels: Legacy labels for points [num_objects, num_points].
        boxes: Legacy box prompts [num_objects, 4].
        propagate: Whether to propagate masks beyond the prompted frames.
        interval: Optional propagation interval relative to the provided frame window.
        keyframes: Optional structured prompt payloads for each object.
        objects_ids: IDs of the prompted objects. Empty for a prompt-free request.
        frame_indexes: Indexes of the prompted frames. Empty for a prompt-free request.
        classes: Class names to detect and track (prompt-free tracking). ``None`` means the
            model's own class set.
        box_threshold: Minimum detection confidence for an object to be tracked (prompt-free
            tracking). ``None`` means the model's default.
    """

    video: list[str | Path | bytes] | str | Path | bytes
    points: list[list[list[int]]] | None = None
    labels: list[list[int]] | None = None
    boxes: list[list[int]] | None = None
    propagate: bool | None = None
    interval: TrackingInterval | None = None
    keyframes: list[TrackingKeyframe] | None = None
    objects_ids: list[int] = []
    frame_indexes: list[int] = []
    classes: list[str] | str | None = None
    box_threshold: float | None = None

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
        if len(v) != len(set(v)):
            raise ValueError("Object IDs should be unique.")
        return v

    @model_validator(mode="after")
    def _check_prompts(self) -> "TrackingInput":
        has_prompts = any(prompt is not None for prompt in (self.keyframes, self.points, self.labels, self.boxes))
        if has_prompts and not self.objects_ids:
            raise ValueError("Prompts require object IDs: provide one object ID per prompted object.")
        if self.keyframes is not None and len(self.keyframes) != len(self.objects_ids):
            raise ValueError("When keyframes are provided, there must be exactly one keyframe per object ID.")
        return self


class TrackedObject(_BaseModel):
    """One tracked object in one frame.

    A mask-based tracker (SAM2) fills ``mask``; a detection-based tracker (ByteTrack) fills ``box``
    and ``score``. At least one of ``box`` and ``mask`` is present.

    Attributes:
        track_id: Identity of the object across frames. A prompted model returns the object ID of
            the request; a prompt-free model assigns it.
        box: Bounding box ``[x1, y1, x2, y2]`` in pixels of the frame (top-left and bottom-right
            corners), or ``None``.
        score: Confidence of the object in this frame, or ``None``.
        class_name: Class name of the object, or ``None``. Serialized as ``class``.
        mask: Mask of the object in compressed-RLE format, or ``None``.
    """

    track_id: int
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    score: float | None = None
    class_name: str | None = Field(default=None, alias="class")
    mask: CompressedRLE | None = None

    @model_validator(mode="after")
    def _check_location(self) -> "TrackedObject":
        if self.box is None and self.mask is None:
            raise ValueError("A tracked object needs a box or a mask.")
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            if x2 < x1 or y2 < y1:
                raise ValueError("A box is [x1, y1, x2, y2] (two corners), not [x, y, width, height].")
        return self


class TrackedFrame(_BaseModel):
    """The objects tracked in one frame.

    Attributes:
        frame_index: Index of the frame, 0-based and relative to the submitted video frames.
        objects: Objects tracked in the frame, each track at most once.
    """

    frame_index: int
    objects: list[TrackedObject] = []

    @field_validator("objects")
    @classmethod
    def _check_unique_tracks(cls, v: list[TrackedObject]) -> list[TrackedObject]:
        track_ids = [tracked.track_id for tracked in v]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("A track appears at most once in a frame.")
        return v


class TrackingOutput(_BaseModel):
    """Output for video tracking: for each frame, the objects tracked in it.

    This is the shape a tracker produces, one frame at a time: append one :class:`TrackedFrame` per
    frame, holding one :class:`TrackedObject` per object alive in that frame. Use :meth:`tracks` to
    read the same result as trajectories.

    Attributes:
        frames: Tracked frames, in the order the model processed them. A frame appears at most once;
            a frame where nothing is tracked may be omitted or have no objects.
    """

    frames: list[TrackedFrame]

    @field_validator("frames")
    @classmethod
    def _check_unique_frames(cls, v: list[TrackedFrame]) -> list[TrackedFrame]:
        frame_indexes = [frame.frame_index for frame in v]
        if len(frame_indexes) != len(set(frame_indexes)):
            raise ValueError("A frame appears at most once in a tracking output.")
        return v

    def tracks(self) -> dict[int, list[tuple[int, TrackedObject]]]:
        """Group the result by track.

        Returns:
            For each track ID, its ``(frame_index, object)`` pairs sorted by frame index.
        """
        tracks: dict[int, list[tuple[int, TrackedObject]]] = {}
        for frame in sorted(self.frames, key=lambda frame: frame.frame_index):
            for tracked in frame.objects:
                tracks.setdefault(tracked.track_id, []).append((frame.frame_index, tracked))
        return tracks
