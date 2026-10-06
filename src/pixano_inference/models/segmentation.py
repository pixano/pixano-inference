# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Segmentation model base class.

The I/O types (:class:`SegmentationInput`/:class:`SegmentationOutput`) are the wire contract and
live in :mod:`pixano_inference.schemas.segmentation`; they are re-exported here so
``from pixano_inference.models.segmentation import SegmentationInput`` keeps working.
"""

from typing import ClassVar

from pixano_inference.schemas.segmentation import SegmentationInput, SegmentationOutput  # noqa: F401

from .base import InferenceModel


class SegmentationModel(InferenceModel[SegmentationInput, SegmentationOutput]):
    """Base class for image segmentation models.

    ``predict`` receives a :class:`SegmentationInput` (image, prompts and options) and returns a
    :class:`SegmentationOutput` (masks, scores and, optionally, embeddings).

    Example:
        ```python
        @register_model("my-segmenter")
        class MySegmenter(SegmentationModel):
            def load_model(self):
                self.model = load_weights(self.config.model_params["path"])

            def predict(self, input: SegmentationInput) -> SegmentationOutput:
                masks, scores = self.model(input.image, input.points, input.labels, input.boxes)
                return SegmentationOutput(masks=masks, scores=scores)
        ```
    """

    capability_name: ClassVar[str] = "segmentation"
