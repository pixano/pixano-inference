# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Segmentation model base class.

The I/O types (:class:`SegmentationInput`/:class:`SegmentationOutput`) are the wire contract and
live in :mod:`pixano_inference_client.segmentation`; they are re-exported here so
``from pixano_inference.models.segmentation import SegmentationInput`` keeps working.
"""

from abc import abstractmethod
from typing import ClassVar

from pixano_inference_client.segmentation import SegmentationInput, SegmentationOutput  # noqa: F401

from .base import InferenceModel


class SegmentationModel(InferenceModel):
    """Base class for image segmentation models.

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

    @abstractmethod
    def predict(self, input: SegmentationInput) -> SegmentationOutput:
        """Run image segmentation.

        Args:
            input: Segmentation input with image, prompts, and options.

        Returns:
            Segmentation output with masks, scores, and optionally embeddings.
        """
