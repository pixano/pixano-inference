# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""/v1 wire schemas shared by the server routes and the client (framework-free).

Canonical home: :mod:`pixano_inference_client.v1`. Re-exported here so existing
``pixano_inference.schemas.v1`` imports keep working.
"""

# ruff: noqa: F401

from pixano_inference_client.v1 import (
    DeployModelRequest,
    JobStatus,
    ModelStatusInfo,
    TrackingKeyframeV1,
    TrackingPrompts,
    TrackingRequestV1,
)


__all__ = [
    "TrackingPrompts",
    "TrackingKeyframeV1",
    "TrackingRequestV1",
    "DeployModelRequest",
    "ModelStatusInfo",
    "JobStatus",
]
