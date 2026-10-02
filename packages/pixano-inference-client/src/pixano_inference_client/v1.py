# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of :mod:`pixano_inference.schemas.v1`."""

# ruff: noqa: F401

from pixano_inference.schemas.v1 import (
    DeployModelRequest,
    JobStatus,
    ModelStatusInfo,
    TrackingKeyframeV1,
    TrackingPrompts,
    TrackingRequestV1,
)


__all__ = [
    "DeployModelRequest",
    "JobStatus",
    "ModelStatusInfo",
    "TrackingKeyframeV1",
    "TrackingPrompts",
    "TrackingRequestV1",
]
