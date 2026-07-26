# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""End-to-end test: the real client against a live server (Ray Serve behind the app)."""

from __future__ import annotations

import sys

import pytest


pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray")
serve = pytest.importorskip("ray.serve")

import httpx  # noqa: E402
import ray.cloudpickle as _cloudpickle  # noqa: E402

from pixano_inference.client import PixanoInferenceClient  # noqa: E402
from pixano_inference.models import DetectionModel, DetectionOutput, register_model  # noqa: E402
from pixano_inference.ray.app import create_ray_serve_app  # noqa: E402


@register_model("ClientE2EStubDetector")
class ClientE2EStubDetector(DetectionModel):
    """Numpy-only detector for the client end-to-end test."""

    def load_model(self) -> None:
        """Mark loaded."""
        self._loaded = True

    def predict(self, input):  # noqa: ANN001, D102
        return DetectionOutput(boxes=[[5, 6, 7, 8]], scores=[0.77], classes=["gadget"], masks=None)


_cloudpickle.register_pickle_by_value(sys.modules[__name__])


async def test_client_against_live_app():
    from pixano_inference.ray.config import ModelDeploymentConfig, RayServeConfig, ResourceConfig
    from pixano_inference.schemas import DetectionRequest

    config = RayServeConfig(
        num_gpus=0,
        models=[
            ModelDeploymentConfig(
                name="client-e2e-det",
                capability="detection",
                model_class="ClientE2EStubDetector",
                resources=ResourceConfig(num_gpus=0.0, num_cpus=1.0),
            )
        ],
    )
    app, _ = create_ray_serve_app(config)

    # Run the app's lifespan (ray.init + serve.start + deploy) and drive the real client
    # over an in-process ASGI transport.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with PixanoInferenceClient(url="http://testserver", transport=transport) as client:
            assert (await client.health())["status"] == "healthy"
            assert (await client.ready())["ready"] is True

            result = await client.detection(
                DetectionRequest(model="client-e2e-det", image="https://example.com/x.jpg", classes=["gadget"])
            )
            assert result.data.classes == ["gadget"]
            assert result.data.boxes == [[5, 6, 7, 8]]

            models = await client.list_models()
            assert any(m.name == "client-e2e-det" and m.status == "RUNNING" for m in models)
