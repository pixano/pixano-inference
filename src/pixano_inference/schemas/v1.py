# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""/v1 wire schemas shared by the server routes and the client (framework-free).

Most capability requests/responses are the camelCase-aliased models in
:mod:`pixano_inference.schemas.inference`. This module adds the shapes that differ from the
internal model I/O — the frontend's nested ``keyframes[].prompts`` tracking request, and the
admin/job envelopes. It imports no web framework, so the client can depend on it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pixano_inference.models.tracking import (
    TrackingBoxPrompt,
    TrackingInput,
    TrackingInterval,
    TrackingKeyframe,
    TrackingPointPrompt,
)
from pixano_inference.schemas.base import BaseRequest, CamelModel
from pixano_inference.schemas.rle import CompressedRLE


class TrackingPrompts(CamelModel):
    """Prompt payload for a single keyframe (points/box XOR mask)."""

    points: list[TrackingPointPrompt] | None = None
    box: TrackingBoxPrompt | None = None
    mask: CompressedRLE | None = None


class TrackingKeyframeV1(CamelModel):
    """A keyframe with nested prompts, matching the Pixano frontend contract.

    Frame indices are 0-based **relative to the submitted media window**.
    """

    frame_index: int
    prompts: TrackingPrompts


class TrackingRequestV1(BaseRequest):
    """Video-tracking request with nested keyframe prompts.

    Media is passed by value (URL/base64/frame list/path within an allowed media root);
    dataset-reference resolution belongs in the caller (the Pixano backend).
    """

    video: list[str | Path | bytes] | str | Path | bytes
    objects_ids: list[int]
    frame_indexes: list[int]
    propagate: bool | None = None
    interval: TrackingInterval | None = None
    keyframes: list[TrackingKeyframeV1] | None = None

    def to_input(self) -> TrackingInput:
        """Flatten the nested keyframes onto the internal :class:`TrackingInput`."""
        flat_keyframes = None
        if self.keyframes is not None:
            flat_keyframes = [
                TrackingKeyframe(
                    frame_index=kf.frame_index,
                    points=kf.prompts.points,
                    box=kf.prompts.box,
                    mask=kf.prompts.mask,
                )
                for kf in self.keyframes
            ]
        return TrackingInput(
            video=self.video,
            objects_ids=self.objects_ids,
            frame_indexes=self.frame_indexes,
            propagate=self.propagate,
            interval=self.interval,
            keyframes=flat_keyframes,
        )


class DeployModelRequest(CamelModel):
    """Admin request to deploy a model at runtime (mirrors a config-file ModelConfig)."""

    name: str | None = None
    model_class: str
    model_params: dict[str, Any] = {}
    deployment: dict[str, Any] = {}


class ModelStatusInfo(CamelModel):
    """Model listing entry with its live Serve status."""

    name: str
    capability: str
    model_class: str | None = None
    model_path: str | None = None
    status: str


class JobStatus(CamelModel):
    """Status envelope for an asynchronous job."""

    job_id: str
    status: Literal["running", "completed", "failed", "canceled"]
    detail: str | None = None
    data: dict[str, Any] | None = None
    metadata: dict[str, Any] = {}
    processing_time: float = 0.0
