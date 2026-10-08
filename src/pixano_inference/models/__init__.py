# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Public API for inference models.

This module re-exports all base classes, I/O types, the ``interface`` descriptors, and the model
registry.
ML engineers should import from here when creating custom models.
"""

# ruff: noqa: F401

from .base import InferenceModel
from .capabilities import (
    CAPABILITIES,
    HTTP_CAPABILITY_BASES,
    BinaryUpload,
    CapabilitySpec,
    capability_of,
    find_capability,
    get_capability,
    infer_http_capability,
)
from .detection import DetectionInput, DetectionInterface, DetectionModel, DetectionOutput
from .embedding import EmbeddingInput, EmbeddingInterface, EmbeddingModel, EmbeddingOutput
from .ner import NEREntity, NERInput, NERInterface, NERModel, NEROutput
from .registry import ModelClassRegistry, register_model
from .segmentation import SegmentationInput, SegmentationInterface, SegmentationModel, SegmentationOutput
from .tracking import (
    TrackedFrame,
    TrackedObject,
    TrackingInput,
    TrackingInterface,
    TrackingModel,
    TrackingOutput,
)
from .vlm import ImageRange, UsageInfo, VLMInput, VLMInterface, VLMModel, VLMOutput
