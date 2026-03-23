# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Capability helpers for HTTP-exposed inference model families."""

from __future__ import annotations

from pixano_inference.models.base import InferenceModel
from pixano_inference.models.detection import DetectionModel
from pixano_inference.models.segmentation import SegmentationModel
from pixano_inference.models.tracking import TrackingModel
from pixano_inference.models.vlm import VLMModel


HTTP_CAPABILITY_BASES: tuple[type[InferenceModel], ...] = (
    SegmentationModel,
    DetectionModel,
    TrackingModel,
    VLMModel,
)


def infer_http_capability(model_cls: type[InferenceModel]) -> str:
    """Infer the HTTP capability implemented by a model class."""
    for base_cls in HTTP_CAPABILITY_BASES:
        if issubclass(model_cls, base_cls):
            capability = getattr(base_cls, "capability_name", None)
            if capability is not None:
                return capability

    supported = ", ".join(base.__name__ for base in HTTP_CAPABILITY_BASES)
    raise ValueError(
        f"Model class '{model_cls.__name__}' is not supported by the HTTP inference API. "
        f"Supported base classes: {supported}."
    )
