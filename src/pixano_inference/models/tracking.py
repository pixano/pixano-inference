# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tracking model base class.

The I/O and prompt types are the wire contract and live in
:mod:`pixano_inference_client.tracking`; they are re-exported here so
``from pixano_inference.models.tracking import TrackingInput`` (and the prompt types) keep working.
"""

from abc import abstractmethod
from typing import ClassVar

from pixano_inference_client.tracking import (  # noqa: F401
    TrackingBoxPrompt,
    TrackingInput,
    TrackingInterval,
    TrackingKeyframe,
    TrackingOutput,
    TrackingPointPrompt,
)

from .base import InferenceModel


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
