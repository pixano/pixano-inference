# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Public API for inference models.

This module re-exports all base classes, I/O types, and the model registry.
ML engineers should import from here when creating custom models.
"""

# ruff: noqa: F401

from .base import InferenceModel
from .detection import DetectionInput, DetectionModel, DetectionOutput
from .llm import LLMInput, LLMModel, LLMOutput
from .ner import NERInput, NERModel, NEROutput
from .registry import ModelClassRegistry, register_model
from .segmentation import SegmentationInput, SegmentationModel, SegmentationOutput
from .tracking import TrackingInput, TrackingModel, TrackingOutput
from .vlm import UsageInfo, VLMInput, VLMModel, VLMOutput
