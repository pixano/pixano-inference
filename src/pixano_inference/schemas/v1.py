# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""/v1 wire schemas shared by the server routes and the client (framework-free).

Most capability requests/responses are the camelCase-aliased models in
:mod:`pixano_inference.schemas.inference`. This module adds the shapes that differ from the
internal model I/O — the frontend's nested ``keyframes[].prompts`` tracking request, and the
admin/job envelopes. It imports no web framework, so the client depends on it directly.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from .base import BaseRequest, _BaseModel
from .interface import ModelInterface
from .rle import CompressedRLE
from .tracking import (
    TrackingBoxPrompt,
    TrackingInput,
    TrackingInterval,
    TrackingKeyframe,
    TrackingOutput,
    TrackingPointPrompt,
)


class TrackingPrompts(_BaseModel):
    """Prompt payload for a single keyframe (points/box XOR mask)."""

    points: list[TrackingPointPrompt] | None = None
    box: TrackingBoxPrompt | None = None
    mask: CompressedRLE | None = None


class TrackingKeyframeV1(_BaseModel):
    """A keyframe with nested prompts, matching the Pixano frontend contract.

    Frame indices are 0-based **relative to the submitted media window**.
    """

    frame_index: int
    prompts: TrackingPrompts


class TrackingRequestV1(BaseRequest):
    """Video-tracking request with nested keyframe prompts.

    A prompted request gives ``objects_ids`` and one keyframe per object. A prompt-free request
    (tracking by detection) gives only the video, and optionally ``classes`` and ``box_threshold``.

    Media is passed by value (URL/base64/frame list/path within an allowed media root);
    dataset-reference resolution belongs in the caller (the Pixano backend).
    """

    video: list[str | Path | bytes] | str | Path | bytes
    objects_ids: list[int] = []
    frame_indexes: list[int] = []
    propagate: bool | None = None
    interval: TrackingInterval | None = None
    keyframes: list[TrackingKeyframeV1] | None = None
    classes: list[str] | str | None = None
    box_threshold: float | None = None

    @model_validator(mode="after")
    def _check_input(self) -> TrackingRequestV1:
        # The consistency rules live on TrackingInput. Building it here rejects a bad request when
        # its body is parsed (422) rather than inside the route handler.
        self.to_input()
        return self

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
            classes=self.classes,
            box_threshold=self.box_threshold,
        )


class DeployModelRequest(_BaseModel):
    """Admin request to deploy a model at runtime (mirrors a config-file ModelConfig)."""

    name: str | None = None
    model_class: str
    model_params: dict[str, Any] = {}
    deployment: dict[str, Any] = {}


class ModelStatusInfo(_BaseModel):
    """Model listing entry with its live Serve status.

    Attributes:
        name: Deployment name, the ``model`` a request names.
        capability: Capability of the model, which fixes its route.
        model_class: Registered class name of the model.
        model_path: Checkpoint id or location, when the model has one.
        status: Live Ray Serve status of the deployment.
        interface: How the model is called, as the model declares it (see
            :mod:`pixano_inference.schemas.interface`). ``None`` when the model declares none or
            the server could not fetch it.
    """

    name: str
    capability: str
    model_class: str | None = None
    model_path: str | None = None
    status: str
    interface: ModelInterface | None = None


class JobStatus(_BaseModel):
    """Status envelope for an asynchronous job.

    Attributes:
        job_id: Identifier of the job, from the submit response.
        status: ``running`` until the job reaches a terminal state.
        detail: Why the job failed or was canceled, when it did.
        data: The result of a completed job, ``None`` otherwise. Jobs run tracking requests, so it
            is the :class:`TrackingOutput` the synchronous route would have returned.
        metadata: Metadata of the model that ran the job.
        timestamp: When the job was submitted or, once terminal, when it ended.
        processing_time: Seconds from submission to the terminal state, ``0`` while running.
    """

    job_id: str
    status: Literal["running", "completed", "failed", "canceled"]
    detail: str | None = None
    data: TrackingOutput | None = None
    metadata: dict[str, Any] = {}
    timestamp: datetime | None = None
    processing_time: float = 0.0
