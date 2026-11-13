# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Detection I/O types."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator, model_validator

from .base import _BaseModel
from .rle import CompressedRLE


def class_names_of(classes: list[str] | str | None) -> list[str]:
    """The class names a request's ``classes`` holds.

    A list holds one name per item; a single string may hold several, separated by ``.`` (the
    Grounding DINO convention, ``"cat . dog"``). Empty names are dropped.
    """
    if classes is None:
        return []
    items = classes.split(".") if isinstance(classes, str) else classes
    return [name.strip() for name in items if name.strip()]


class DetectionInput(_BaseModel):
    """Input for detection.

    When ``classes`` is provided the model runs in **open-vocabulary** mode
    (e.g. GroundingDINO, YOLOE with text prompt).  When ``classes`` is
    ``None`` the model uses its built-in class set (**closed-vocabulary**
    mode, e.g. YOLO, prompt-free YOLOE).

    **Exemplars** ask for "more like these": ``exemplar_boxes`` are boxes, in the pixels of
    ``image``, around instances of what to find, and ``negative_exemplar_boxes`` around things
    that look alike but are not. Only a model whose interface sets ``exemplars`` (and
    ``negative_exemplars``) reads them; any other model ignores them, so a client checks the
    interface before sending them. With exemplars, ``classes`` names at most one class: the class
    of the exemplars, given to every detection and used as a text prompt alongside them. Without a
    class the prompt is visual only, and detections carry the class ``""``.

    Attributes:
        image: Image for detection (path, URL, base64, or raw bytes from the binary route).
        classes: Class names to detect.  ``None`` means closed-vocabulary.
        box_threshold: Confidence threshold for boxes.
        text_threshold: Confidence threshold for text matching (open-vocab only).
        exemplar_boxes: Boxes ``[x1, y1, x2, y2]`` (pixels) around instances of what to find.
        negative_exemplar_boxes: Boxes ``[x1, y1, x2, y2]`` (pixels) around what not to find.
            Requires ``exemplar_boxes``.
    """

    image: str | Path | bytes
    classes: list[str] | str | None = None
    box_threshold: float = 0.5
    text_threshold: float = 0.5
    exemplar_boxes: list[list[int]] | None = None
    negative_exemplar_boxes: list[list[int]] | None = None

    @field_validator("exemplar_boxes", "negative_exemplar_boxes")
    @classmethod
    def _check_boxes(cls, v: list[list[int]] | None) -> list[list[int]] | None:
        for box in v or []:
            if len(box) != 4:
                raise ValueError("An exemplar box is [x1, y1, x2, y2]: it has 4 coordinates.")
            if not (box[0] < box[2] and box[1] < box[3]):
                raise ValueError("An exemplar box is [x1, y1, x2, y2] (two corners), not [x, y, width, height].")
        return v

    @model_validator(mode="after")
    def _check_exemplars(self) -> DetectionInput:
        if self.negative_exemplar_boxes and not self.exemplar_boxes:
            raise ValueError("Negative exemplars require exemplar_boxes.")
        if self.exemplar_boxes and len(class_names_of(self.classes)) > 1:
            raise ValueError("With exemplars, 'classes' names at most one class: the class of the exemplars.")
        return self


class DetectionOutput(_BaseModel):
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
