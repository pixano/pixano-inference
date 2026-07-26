# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""/v1 synchronous inference routes (one per capability)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

from pixano_inference.schemas.inference import (
    DetectionRequest,
    DetectionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    NERRequest,
    NERResponse,
    SegmentationRequest,
    SegmentationResponse,
    TrackingResponse,
    VLMRequest,
    VLMResponse,
)

from .helpers import build_binary_request_from_request, run_inference
from .schemas import TrackingRequestV1


if TYPE_CHECKING:
    from pixano_inference.ray.app import DeploymentManager


def build_inference_router(deployment_manager: DeploymentManager) -> APIRouter:
    """Build the `/inference` router bound to *deployment_manager*."""
    router = APIRouter(prefix="/inference", tags=["inference"])

    @router.post("/segmentation", response_model=SegmentationResponse)
    async def segmentation(request: SegmentationRequest) -> Any:
        return await run_inference(deployment_manager, request.model, request.to_input(), "segmentation")

    @router.post("/segmentation/binary", response_model=SegmentationResponse)
    async def segmentation_binary(request: Request) -> Any:
        parsed = await build_binary_request_from_request(
            request, SegmentationRequest, file_field="image", payload_key="image"
        )
        return await run_inference(deployment_manager, parsed.model, parsed.to_input(), "segmentation")

    @router.post("/detection", response_model=DetectionResponse)
    async def detection(request: DetectionRequest) -> Any:
        return await run_inference(deployment_manager, request.model, request.to_input(), "detection")

    @router.post("/detection/binary", response_model=DetectionResponse)
    async def detection_binary(request: Request) -> Any:
        parsed = await build_binary_request_from_request(
            request, DetectionRequest, file_field="image", payload_key="image"
        )
        return await run_inference(deployment_manager, parsed.model, parsed.to_input(), "detection")

    @router.post("/vlm", response_model=VLMResponse)
    async def vlm(request: VLMRequest) -> Any:
        return await run_inference(deployment_manager, request.model, request.to_input(), "vlm")

    @router.post("/vlm/binary", response_model=VLMResponse)
    async def vlm_binary(request: Request) -> Any:
        parsed = await build_binary_request_from_request(
            request, VLMRequest, file_field="images", payload_key="images"
        )
        return await run_inference(deployment_manager, parsed.model, parsed.to_input(), "vlm")

    @router.post("/ner", response_model=NERResponse)
    async def ner(request: NERRequest) -> Any:
        return await run_inference(deployment_manager, request.model, request.to_input(), "ner")

    @router.post("/embedding", response_model=EmbeddingResponse)
    async def embedding(request: EmbeddingRequest) -> Any:
        return await run_inference(deployment_manager, request.model, request.to_input(), "embedding")

    @router.post("/embedding/binary", response_model=EmbeddingResponse)
    async def embedding_binary(request: Request) -> Any:
        parsed = await build_binary_request_from_request(
            request, EmbeddingRequest, file_field="image", payload_key="image"
        )
        return await run_inference(deployment_manager, parsed.model, parsed.to_input(), "embedding")

    @router.post("/tracking", response_model=TrackingResponse)
    async def tracking(request: TrackingRequestV1) -> Any:
        return await run_inference(deployment_manager, request.model, request.to_input(), "tracking")

    @router.post("/tracking/binary", response_model=TrackingResponse)
    async def tracking_binary(request: Request) -> Any:
        parsed = await build_binary_request_from_request(
            request, TrackingRequestV1, file_field="frames", payload_key="video"
        )
        return await run_inference(deployment_manager, parsed.model, parsed.to_input(), "tracking")

    return router
