# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""HTTP-layer request/response wrappers for inference capabilities."""

from pydantic import ConfigDict

from .base import BaseRequest, BaseResponse
from .detection import DetectionInput, DetectionOutput
from .embedding import EmbeddingInput, EmbeddingOutput
from .ner import NERInput, NEROutput
from .segmentation import SegmentationInput, SegmentationOutput
from .tracking import TrackingInput, TrackingOutput
from .vlm import VLMInput, VLMOutput


class SegmentationRequest(BaseRequest, SegmentationInput):
    """Request for segmentation inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> SegmentationInput:
        """Convert the request to the input."""
        return self.to_base_model(SegmentationInput)


class SegmentationResponse(BaseResponse):
    """Response for segmentation inference."""

    data: SegmentationOutput


class DetectionRequest(BaseRequest, DetectionInput):
    """Request for detection inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> DetectionInput:
        """Convert the request to the input."""
        return self.to_base_model(DetectionInput)


class DetectionResponse(BaseResponse):
    """Response for detection inference."""

    data: DetectionOutput


class TrackingRequest(BaseRequest, TrackingInput):
    """Request for tracking inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> TrackingInput:
        """Convert the request to the input."""
        return self.to_base_model(TrackingInput)


class TrackingResponse(BaseResponse):
    """Response for tracking inference."""

    data: TrackingOutput


class VLMRequest(BaseRequest, VLMInput):
    """Request for VLM inference."""

    def to_input(self) -> VLMInput:
        """Convert the request to the input."""
        return self.to_base_model(VLMInput)


class VLMResponse(BaseResponse):
    """Response for VLM inference."""

    data: VLMOutput


class NERRequest(BaseRequest, NERInput):
    """Request for NER inference."""

    def to_input(self) -> NERInput:
        """Convert the request to the input."""
        return self.to_base_model(NERInput)


class NERResponse(BaseResponse):
    """Response for NER inference."""

    data: NEROutput


class EmbeddingRequest(BaseRequest, EmbeddingInput):
    """Request for embedding inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_input(self) -> EmbeddingInput:
        """Convert the request to the input."""
        return self.to_base_model(EmbeddingInput)


class EmbeddingResponse(BaseResponse):
    """Response for embedding inference."""

    data: EmbeddingOutput
