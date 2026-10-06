# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""/v1 synchronous inference routes, registered from the capability table.

Each capability gets ``POST /inference/<name>`` (JSON body) and, when it accepts raw uploads,
``POST /inference/<name>/binary`` (multipart). The request and response models, the upload fields
and the capability name all come from :data:`pixano_inference.models.capabilities.CAPABILITIES`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

from pixano_inference.models.capabilities import CAPABILITIES, CapabilitySpec

from .helpers import build_capability_binary_request, run_inference


if TYPE_CHECKING:
    from pixano_inference.ray.app import DeploymentManager


_Endpoint = Callable[[Any], Awaitable[Any]]


def _json_endpoint(deployment_manager: DeploymentManager, spec: CapabilitySpec) -> _Endpoint:
    """Build the JSON route handler of *spec*."""

    async def endpoint(request: Any) -> Any:
        return await run_inference(deployment_manager, request.model, request.to_input(), spec.name)

    # FastAPI reads the body model from the annotation. It is set here, as a class rather than a
    # name, because the request type is only known per capability.
    endpoint.__annotations__ = {"request": spec.request_type, "return": Any}
    return endpoint


def _binary_endpoint(deployment_manager: DeploymentManager, spec: CapabilitySpec) -> _Endpoint:
    """Build the multipart route handler of *spec*."""

    async def endpoint(request: Any) -> Any:
        parsed = await build_capability_binary_request(request, spec)
        return await run_inference(deployment_manager, parsed.model, parsed.to_input(), spec.name)

    endpoint.__annotations__ = {"request": Request, "return": Any}
    return endpoint


def build_inference_router(deployment_manager: DeploymentManager) -> APIRouter:
    """Build the `/inference` router bound to *deployment_manager*."""
    router = APIRouter(prefix="/inference", tags=["inference"])

    # The route name is part of the published OpenAPI document (operationId and summary).
    for spec in CAPABILITIES:
        router.add_api_route(
            f"/{spec.name}",
            _json_endpoint(deployment_manager, spec),
            methods=["POST"],
            response_model=spec.response_type,
            name=spec.name,
        )
        if spec.binary is not None:
            router.add_api_route(
                f"/{spec.name}/binary",
                _binary_endpoint(deployment_manager, spec),
                methods=["POST"],
                response_model=spec.response_type,
                name=f"{spec.name}_binary",
            )

    return router
