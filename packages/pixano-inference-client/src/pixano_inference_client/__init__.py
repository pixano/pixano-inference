# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of the client and wire schemas that ship in ``pixano-inference``.

The client (:mod:`pixano_inference.client`) and the wire contract (:mod:`pixano_inference.schemas`)
are part of the base install of ``pixano-inference``, which needs neither Ray nor FastAPI. This
package only re-exports them under their former names, so ``from pixano_inference_client import
PixanoInferenceClient`` keeps working; new code imports from ``pixano_inference`` directly.
"""

import warnings

from pixano_inference.client import PixanoInferenceClient, PixanoInferenceError, SyncPixanoInferenceClient
from pixano_inference.schemas import (
    BaseRequest,
    BaseResponse,
    CompressedRLE,
    DeployModelRequest,
    DetectionInput,
    DetectionOutput,
    DetectionRequest,
    DetectionResponse,
    EmbeddingInput,
    EmbeddingOutput,
    EmbeddingRequest,
    EmbeddingResponse,
    JobStatus,
    ModelInfo,
    ModelStatusInfo,
    NDArray,
    NDArrayFloat,
    NEREntity,
    NERInput,
    NEROutput,
    NERRequest,
    NERResponse,
    SegmentationInput,
    SegmentationOutput,
    SegmentationRequest,
    SegmentationResponse,
    TrackingBoxPrompt,
    TrackingInput,
    TrackingInterval,
    TrackingKeyframe,
    TrackingKeyframeV1,
    TrackingOutput,
    TrackingPointPrompt,
    TrackingPrompts,
    TrackingRequest,
    TrackingRequestV1,
    TrackingResponse,
    UsageInfo,
    VLMInput,
    VLMOutput,
    VLMRequest,
    VLMResponse,
    mask_to_rle,
    rle_to_mask,
)

# 0.1.0 exported the camelCase base model under this name; the core keeps it private.
from pixano_inference.schemas.base import _BaseModel as CamelModel


warnings.warn(
    "pixano-inference-client is deprecated: the client and the wire schemas ship in the base install of "
    "'pixano-inference'. Import from pixano_inference.client and pixano_inference.schemas instead.",
    DeprecationWarning,
    stacklevel=2,
)

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
