# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Lightweight client and wire schemas for the Pixano Inference /v1 API.

This package depends only on ``httpx``/``pydantic``/``numpy`` (plus an optional ``masks`` extra
for ``pycocotools``/``Pillow``) — none of the server stack (``ray[serve]``, ``fastapi``,
``uvicorn``, torch). Install it when your app only needs to *call* a running server.

The full ``pixano-inference`` package re-exports everything here, so
``from pixano_inference.client import PixanoInferenceClient`` and
``from pixano_inference_client import PixanoInferenceClient`` refer to the same classes.
"""

from .base import BaseRequest, BaseResponse, CamelModel
from .client import PixanoInferenceClient, PixanoInferenceError, SyncPixanoInferenceClient
from .detection import DetectionInput, DetectionOutput
from .embedding import EmbeddingInput, EmbeddingOutput
from .inference import (
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
from .models_info import ModelInfo
from .nd_array import NDArray, NDArrayFloat
from .ner import NEREntity, NERInput, NEROutput
from .rle import CompressedRLE, mask_to_rle, rle_to_mask
from .segmentation import SegmentationInput, SegmentationOutput
from .tracking import (
    TrackingBoxPrompt,
    TrackingInput,
    TrackingInterval,
    TrackingKeyframe,
    TrackingOutput,
    TrackingPointPrompt,
)
from .v1 import (
    DeployModelRequest,
    JobStatus,
    ModelStatusInfo,
    TrackingKeyframeV1,
    TrackingPrompts,
    TrackingRequestV1,
)
from .vlm import UsageInfo, VLMInput, VLMOutput


__all__ = [
    # Clients
    "PixanoInferenceClient",
    "SyncPixanoInferenceClient",
    "PixanoInferenceError",
    # Base wire types
    "CamelModel",
    "BaseRequest",
    "BaseResponse",
    "NDArray",
    "NDArrayFloat",
    "CompressedRLE",
    "mask_to_rle",
    "rle_to_mask",
    "ModelInfo",
    # Requests / responses
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
    # Capability I/O types
    "SegmentationInput",
    "SegmentationOutput",
    "DetectionInput",
    "DetectionOutput",
    "TrackingInput",
    "TrackingOutput",
    "TrackingPointPrompt",
    "TrackingBoxPrompt",
    "TrackingInterval",
    "TrackingKeyframe",
    "VLMInput",
    "VLMOutput",
    "UsageInfo",
    "NERInput",
    "NEROutput",
    "NEREntity",
    "EmbeddingInput",
    "EmbeddingOutput",
    # v1-specific admin/tracking/job types
    "TrackingRequestV1",
    "TrackingKeyframeV1",
    "TrackingPrompts",
    "DeployModelRequest",
    "ModelStatusInfo",
    "JobStatus",
]
