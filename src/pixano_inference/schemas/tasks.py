# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""HTTP-layer Request/Response wrappers for each task.

These combine ``BaseRequest`` with the task-specific ``Input`` types from
``pixano_inference.models`` so that the HTTP layer can deserialise incoming
JSON into a single object that carries both the ``model`` field and the
task-specific parameters.
"""

from pydantic import ConfigDict

from pixano_inference.models.detection import DetectionInput, DetectionOutput
from pixano_inference.models.segmentation import SegmentationInput, SegmentationOutput
from pixano_inference.models.tracking import TrackingInput, TrackingOutput
from pixano_inference.models.vlm import VLMInput, VLMOutput

from .base import BaseRequest, BaseResponse


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


class SegmentationRequest(BaseRequest, SegmentationInput):
    """Request for image segmentation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> SegmentationInput:
        """Convert the request to the input."""
        return self.to_base_model(SegmentationInput)


class SegmentationResponse(BaseResponse):
    """Response for image segmentation.

    Attributes:
        data: Output of the segmentation.
    """

    data: SegmentationOutput


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class DetectionRequest(BaseRequest, DetectionInput):
    """Request for zero-shot detection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> DetectionInput:
        """Convert the request to the input."""
        return self.to_base_model(DetectionInput)


class DetectionResponse(BaseResponse):
    """Response for zero-shot detection.

    Attributes:
        data: Output of the detection.
    """

    data: DetectionOutput


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


class TrackingRequest(BaseRequest, TrackingInput):
    """Request for video mask generation / tracking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> TrackingInput:
        """Convert the request to the input."""
        return self.to_base_model(TrackingInput)


class TrackingResponse(BaseResponse):
    """Response for video mask generation / tracking.

    Attributes:
        data: Output of the tracking.
    """

    data: TrackingOutput


# ---------------------------------------------------------------------------
# VLM
# ---------------------------------------------------------------------------


class VLMRequest(BaseRequest, VLMInput):
    """Request for vision-language model generation."""

    def to_input(self) -> VLMInput:
        """Convert the request to the input."""
        return self.to_base_model(VLMInput)


class VLMResponse(BaseResponse):
    """Response for vision-language model generation.

    Attributes:
        data: Output of the generation.
    """

    data: VLMOutput
