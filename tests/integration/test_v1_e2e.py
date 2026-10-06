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

from pixano_inference.models import (  # noqa: E402
    DetectionModel,
    DetectionOutput,
    TrackedFrame,
    TrackedObject,
    TrackingModel,
    TrackingOutput,
    register_model,
)
from pixano_inference.ray.app import create_ray_serve_app  # noqa: E402


@register_model("E2EStubDetector")
class E2EStubDetector(DetectionModel):
    """Numpy-only detector deployed through the real /v1 app."""

    def load_model(self) -> None:
        """Mark loaded."""
        self._loaded = True

    def predict(self, input):  # noqa: ANN001, D102
        return DetectionOutput(boxes=[[10, 20, 30, 40]], scores=[0.87], classes=["thing"], masks=None)


@register_model("E2EStubTracker")
class E2EStubTracker(TrackingModel):
    """Tracking by detection without a framework: two boxes drifting right, one frame at a time."""

    def load_model(self) -> None:
        """Mark loaded."""
        self._loaded = True

    def predict(self, input):  # noqa: ANN001, D102
        frames = []
        for index in range(len(input.video)):
            objects = [
                TrackedObject(track_id=1, box=[10 + index, 20, 30 + index, 40], score=0.9, class_name="thing"),
            ]
            if index > 0:  # a second track is born on the second frame
                objects.append(TrackedObject(track_id=2, box=[50 + index, 60, 70 + index, 80], score=0.6))
            frames.append(TrackedFrame(frame_index=index, objects=objects))
        return TrackingOutput(frames=frames)


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


def test_v1_prompt_free_tracking_end_to_end():
    from pixano_inference.ray.config import ModelDeploymentConfig, RayServeConfig, ResourceConfig

    config = RayServeConfig(
        num_gpus=0,
        strict_startup=True,
        models=[
            ModelDeploymentConfig(
                name="e2e-mot",
                capability="tracking",
                model_class="E2EStubTracker",
                resources=ResourceConfig(num_gpus=0.0, num_cpus=1.0),
            )
        ],
    )
    app, _ = create_ray_serve_app(config)
    with TestClient(app) as client:
        assert client.get("/v1/ready").status_code == 200
        # No object ID, no keyframe: the model creates the tracks.
        resp = client.post(
            "/v1/inference/tracking", json={"model": "e2e-mot", "video": ["f0.png", "f1.png", "f2.png"]}
        )
        assert resp.status_code == 200, resp.text
        frames = resp.json()["data"]["frames"]
        assert [frame["frameIndex"] for frame in frames] == [0, 1, 2]
        assert [[tracked["trackId"] for tracked in frame["objects"]] for frame in frames] == [[1], [1, 2], [1, 2]]
        assert frames[2]["objects"][0] == {
            "trackId": 1,
            "box": [12.0, 20.0, 32.0, 40.0],
            "score": 0.9,
            "class": "thing",
            "mask": None,
        }
