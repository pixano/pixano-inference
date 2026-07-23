# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Detection model base class.

The I/O types (:class:`DetectionInput`/:class:`DetectionOutput`) are the wire contract and live
in :mod:`pixano_inference_client.detection`; they are re-exported here so
``from pixano_inference.models.detection import DetectionInput`` keeps working.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from pixano_inference_client.detection import DetectionInput, DetectionOutput  # noqa: F401

from .base import InferenceModel


class DetectionModel(InferenceModel):
    """Base class for detection and instance-segmentation models.

    Subclasses implement ``predict`` which receives a :class:`DetectionInput`
    and returns a :class:`DetectionOutput`.  The same pair of types covers
    both pure detection (no masks) and instance segmentation (with masks).

    Example:
        ```python
        @register_model("my-detector")
        class MyDetector(DetectionModel):
            def load_model(self):
                self.model = load_weights(self.config.model_params["path"])

            def predict(self, input: DetectionInput) -> DetectionOutput:
                boxes, scores, cls = self.model(input.image, input.classes)
                return DetectionOutput(boxes=boxes, scores=scores, classes=cls)
        ```
    """

    capability_name: ClassVar[str] = "detection"

    @abstractmethod
    def predict(self, input: DetectionInput) -> DetectionOutput:
        """Run detection or instance segmentation.

        Args:
            input: Detection input with image, optional classes, and thresholds.

        Returns:
            Detection output with boxes, scores, classes, and optional masks.
        """
