# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

# ruff: noqa: F401
# ruff: noqa: D104

from .base import APIRequest, BaseRequest, BaseResponse
from .models import ModelInfo
from .nd_array import NDArray, NDArrayFloat
from .rle import CompressedRLE


def __getattr__(name: str):
    """Lazy imports for task schemas to avoid circular imports with models/."""
    _task_names = {
        "DetectionRequest",
        "DetectionResponse",
        "SegmentationRequest",
        "SegmentationResponse",
        "TrackingRequest",
        "TrackingResponse",
        "VLMRequest",
        "VLMResponse",
    }
    if name in _task_names:
        from . import tasks as _tasks

        return getattr(_tasks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
