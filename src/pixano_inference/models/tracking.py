# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tracking model base class.

The I/O and prompt types are the wire contract and live in
:mod:`pixano_inference.schemas.tracking`; they are re-exported here so
``from pixano_inference.models.tracking import TrackingInput`` (and the prompt types) keep working.
"""

from typing import ClassVar

from pixano_inference.schemas.interface import TrackingInterface  # noqa: F401
from pixano_inference.schemas.tracking import (  # noqa: F401
    TrackedFrame,
    TrackedObject,
    TrackingBoxPrompt,
    TrackingInput,
    TrackingInterval,
    TrackingKeyframe,
    TrackingOutput,
    TrackingPointPrompt,
)

from .base import InferenceModel


class TrackingModel(InferenceModel[TrackingInput, TrackingOutput]):
    """Base class for video tracking models.

    ``predict`` receives a :class:`TrackingInput` and returns a :class:`TrackingOutput`: for each
    frame, the objects tracked in it. A model is either prompted (the request names the objects with
    points, boxes or masks, and the model returns their masks, as SAM2 does) or prompt-free
    (tracking by detection: the request has no object, and the model creates the tracks and returns
    their boxes and scores, as ByteTrack does). The ``interface`` property, when overridden, returns
    a :class:`TrackingInterface`: which prompts the model takes, whether it accepts a prompt-free
    request, its class set and what each tracked object carries.

    Example:
        ```python
        @register_model("my-tracker")
        class MyTracker(TrackingModel):
            def load_model(self):
                self.detector, self.tracker = load_detector(...), load_tracker(...)

            def predict(self, input: TrackingInput) -> TrackingOutput:
                frames = []
                for index, image in enumerate(load_frames(input.video)):
                    tracks = self.tracker.update(self.detector(image))
                    frames.append(
                        TrackedFrame(
                            frame_index=index,
                            objects=[TrackedObject(track_id=t.id, box=t.xyxy, score=t.score) for t in tracks],
                        )
                    )
                return TrackingOutput(frames=frames)
        ```
    """

    capability_name: ClassVar[str] = "tracking"
