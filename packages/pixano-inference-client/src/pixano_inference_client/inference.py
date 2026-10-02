# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of :mod:`pixano_inference.schemas.inference`."""

# ruff: noqa: F401

from pixano_inference.schemas.inference import (
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
    "DetectionRequest",
    "DetectionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "NERRequest",
    "NERResponse",
    "SegmentationRequest",
    "SegmentationResponse",
    "TrackingRequest",
    "TrackingResponse",
    "VLMRequest",
    "VLMResponse",
]
