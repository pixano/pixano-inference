# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Detection I/O types."""

from __future__ import annotations

from pathlib import Path

from .base import CamelModel
from .rle import CompressedRLE


class DetectionInput(CamelModel):
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

    image: str | Path
    classes: list[str] | str | None = None
    box_threshold: float = 0.5
    text_threshold: float = 0.5


class DetectionOutput(CamelModel):
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
