# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Detection model base class and I/O types."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from pixano_inference.schemas.rle import CompressedRLE

from .base import InferenceModel


class DetectionInput(BaseModel):
    """Input for detection.

    When ``classes`` is provided the model runs in **open-vocabulary** mode
    (e.g. GroundingDINO, YOLOE with text prompt).  When ``classes`` is
    ``None`` the model uses its built-in class set (**closed-vocabulary**
    mode, e.g. YOLO, prompt-free YOLOE).

    Attributes:
        image: Image for detection (path, URL, or base64).
        classes: Class names to detect.  ``None`` means closed-vocabulary.
        box_threshold: Confidence threshold for boxes.
        text_threshold: Confidence threshold for text matching (open-vocab only).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: str | Path
    classes: list[str] | str | None = None
    box_threshold: float = 0.5
    text_threshold: float = 0.5


class DetectionOutput(BaseModel):
    """Output for detection.

    ``masks`` is populated when the model performs instance segmentation
    (boxes + masks).  For detection-only models it is ``None``.

    Attributes:
        boxes: List of detected boxes.
        scores: List of confidence scores.
        classes: List of class names associated with each box.
        masks: Optional instance masks in compressed-RLE format.
    """

    boxes: list[list[int]]
    scores: list[float]
    classes: list[str]
    masks: list[CompressedRLE] | None = None


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
