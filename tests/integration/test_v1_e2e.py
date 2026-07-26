# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""End-to-end /v1 test: real Ray Serve behind the FastAPI app via the lifespan."""

from __future__ import annotations

import sys

import pytest


pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray")
serve = pytest.importorskip("ray.serve")

import ray.cloudpickle as _cloudpickle  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from pixano_inference.models import DetectionModel, DetectionOutput, register_model  # noqa: E402
from pixano_inference.ray.app import create_ray_serve_app  # noqa: E402


@register_model("E2EStubDetector")
class E2EStubDetector(DetectionModel):
    """Numpy-only detector deployed through the real /v1 app."""

    def load_model(self) -> None:
        """Mark loaded."""
        self._loaded = True

    def predict(self, input):  # noqa: ANN001, D102
        return DetectionOutput(boxes=[[10, 20, 30, 40]], scores=[0.87], classes=["thing"], masks=None)


_cloudpickle.register_pickle_by_value(sys.modules[__name__])


def test_v1_detection_end_to_end():
    from pixano_inference.ray.config import ModelDeploymentConfig, RayServeConfig, ResourceConfig

    config = RayServeConfig(
        num_gpus=0,
        strict_startup=True,
        models=[
            ModelDeploymentConfig(
                name="e2e-det",
                capability="detection",
                model_class="E2EStubDetector",
                resources=ResourceConfig(num_gpus=0.0, num_cpus=1.0),
            )
        ],
    )
    app, _ = create_ray_serve_app(config)
    # The context manager runs the lifespan: ray.init + serve.start + deploy the model.
    with TestClient(app) as client:
        assert client.get("/v1/ready").status_code == 200
        resp = client.post(
            "/v1/inference/detection",
            json={"model": "e2e-det", "image": "https://example.com/x.jpg", "classes": ["thing"]},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["data"]["classes"] == ["thing"]
        assert body["data"]["boxes"] == [[10, 20, 30, 40]]
        # Model listing reflects the running model.
        models = client.get("/v1/models").json()
        assert any(m["name"] == "e2e-det" and m["status"] == "RUNNING" for m in models)
