# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""HTTP-layer request/response wrappers for inference capabilities.

Canonical home: :mod:`pixano_inference_client.inference`. Re-exported here so existing
``pixano_inference.schemas.inference`` imports keep working.
"""

# ruff: noqa: F401

from pixano_inference_client.inference import (
    DetectionRequest,
    DetectionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    NERRequest,
    NERResponse,
    SegmentationRequest,
    SegmentationResponse,
    TrackingRequest,
    TrackingResponse,
    VLMRequest,
    VLMResponse,
)


__all__ = [
    "SegmentationRequest",
    "SegmentationResponse",
    "DetectionRequest",
    "DetectionResponse",
    "TrackingRequest",
    "TrackingResponse",
    "VLMRequest",
    "VLMResponse",
    "NERRequest",
    "NERResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
]
