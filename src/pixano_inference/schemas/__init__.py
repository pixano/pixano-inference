# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""The wire contract of the /v1 API.

Everything a server, a model implementation and a client exchange is defined here: the wire value
types (``NDArray``, ``CompressedRLE``), the capability ``Input``/``Output`` types that a model's
``predict`` consumes and returns, the HTTP request/response envelopes, the admin and job types, and
the per-capability ``interface`` descriptor a model may publish.

This package depends on pydantic and numpy only, and imports nothing else from
``pixano_inference``: the model API, the client and the server all build on it.
"""

# ruff: noqa: F401

from .base import BaseRequest, BaseResponse
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
from .interface import (
    DetectionInterface,
    EmbeddingInterface,
    ImageRange,
    ModelInterface,
    NERInterface,
    SegmentationInterface,
    TrackingInterface,
    VLMInterface,
)
from .models import ModelInfo
from .nd_array import NDArray, NDArrayFloat
from .ner import NEREntity, NERInput, NEROutput
from .rle import CompressedRLE, mask_to_rle, rle_to_mask
from .segmentation import SegmentationInput, SegmentationOutput
from .tracking import (
    TrackedFrame,
    TrackedObject,
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
