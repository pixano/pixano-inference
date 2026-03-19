# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import ray
from fastapi.testclient import TestClient

from pixano_inference.models.detection import DetectionOutput
from pixano_inference.models.segmentation import SegmentationOutput
from pixano_inference.models.tracking import TrackingOutput
from pixano_inference.models.vlm import UsageInfo, VLMOutput
from pixano_inference.ray import app as ray_app_module
from pixano_inference.ray.app import create_ray_serve_app
from pixano_inference.ray.config import RayServeConfig
from pixano_inference.schemas import ModelInfo
from pixano_inference.schemas.nd_array import NDArrayFloat
from pixano_inference.schemas.rle import CompressedRLE


class FakeRemoteMethod:
    def __init__(self, result):
        self._result = result
        self.last_input = None

    def remote(self, input_data):
        self.last_input = input_data
        return self._result


class FakeHandle:
    def __init__(self, result):
        self.predict = FakeRemoteMethod(result)


@pytest.fixture
def ray_app_client():
    config = RayServeConfig(num_gpus=0)
    app, _ = create_ray_serve_app(config)
    return TestClient(app)


class TestManagementRoutesRemoved:
    def test_instantiate_model_not_found(self, ray_app_client: TestClient):
        response = ray_app_client.post("/providers/instantiate")
        assert response.status_code in (404, 405)

    def test_deploy_model_not_found(self, ray_app_client: TestClient):
        response = ray_app_client.post("/models/deploy")
        assert response.status_code in (404, 405)

    def test_delete_model_not_found(self, ray_app_client: TestClient):
        response = ray_app_client.delete("/providers/model/test-model")
        assert response.status_code in (404, 405)


class TestServiceRoutes:
    def test_health(self, ray_app_client: TestClient):
        response = ray_app_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_list_models(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        manager = ray_app_client.app.state.deployment_manager
        monkeypatch.setattr(
            manager,
            "list_models",
            lambda: [ModelInfo(name="sam2-image", capability="segmentation", model_class="Sam2ImageModel")],
        )

        response = ray_app_client.get("/app/models/")
        assert response.status_code == 200
        assert response.json()[0]["capability"] == "segmentation"

    def test_settings_uses_capability_mapping(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        fake_ray = SimpleNamespace(
            is_initialized=lambda: True,
            cluster_resources=lambda: {"GPU": 4.0},
            available_resources=lambda: {"GPU": 1.5},
        )
        monkeypatch.setattr(ray_app_module, "ray", fake_ray)

        manager = ray_app_client.app.state.deployment_manager
        monkeypatch.setattr(
            manager,
            "list_models",
            lambda: [
                ModelInfo(name="sam2-image", capability="segmentation", model_class="Sam2ImageModel"),
                ModelInfo(name="sam2-video", capability="tracking", model_class="Sam2VideoModel"),
            ],
        )

        response = ray_app_client.get("/app/settings/")
        assert response.status_code == 200

        payload = response.json()
        assert payload["num_gpus"] == 4
        assert payload["gpus_used"] == 2.5
        assert payload["models_to_capability"] == {
            "sam2-image": "segmentation",
            "sam2-video": "tracking",
        }


class TestInferenceRoutes:
    @staticmethod
    def _install_manager_stubs(
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        *,
        handle: FakeHandle | None,
        capability: str,
        metadata: dict | None = None,
    ) -> None:
        manager = client.app.state.deployment_manager
        monkeypatch.setattr(manager, "get_handle", lambda name: handle)
        monkeypatch.setattr(manager, "get_model_capability", lambda name: capability if handle is not None else None)
        monkeypatch.setattr(manager, "get_model_metadata", lambda name: metadata or {"capability": capability})
        monkeypatch.setattr(ray, "get", lambda value: value)

    def test_segmentation_route(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        result = SegmentationOutput(
            masks=[[CompressedRLE.from_mask(np.array([[1, 0], [0, 1]], dtype=np.uint8))]],
            scores=NDArrayFloat(values=[0.95], shape=[1, 1]),
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(
            ray_app_client,
            monkeypatch,
            handle=handle,
            capability="segmentation",
            metadata={"model_name": "sam2-image", "capability": "segmentation"},
        )

        response = ray_app_client.post(
            "/inference/segmentation/",
            json={"model": "sam2-image", "image": "https://example.com/image.jpg"},
        )

        assert response.status_code == 200
        assert response.json()["metadata"]["capability"] == "segmentation"
        assert handle.predict.last_input.image == "https://example.com/image.jpg"

    def test_tracking_route(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        result = TrackingOutput(
            objects_ids=[1],
            frame_indexes=[0],
            masks=[CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))],
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="tracking")

        response = ray_app_client.post(
            "/inference/tracking/",
            json={
                "model": "sam2-video",
                "video": ["frame-0001.png"],
                "objects_ids": [1],
                "frame_indexes": [0],
            },
        )

        assert response.status_code == 200
        assert response.json()["metadata"]["capability"] == "tracking"

    def test_vlm_route(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        result = VLMOutput(
            generated_text="hello world",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            generation_config={"temperature": 0.0},
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="vlm")

        response = ray_app_client.post(
            "/inference/vlm/",
            json={
                "model": "qwen-vl",
                "prompt": "describe this image",
                "images": ["https://example.com/image.jpg"],
                "max_new_tokens": 32,
            },
        )

        assert response.status_code == 200
        assert response.json()["data"]["generated_text"] == "hello world"

    def test_detection_route(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        result = DetectionOutput(
            boxes=[[1, 2, 3, 4]],
            scores=[0.9],
            classes=["truck"],
            masks=None,
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="detection")

        response = ray_app_client.post(
            "/inference/detection/",
            json={"model": "grounding-dino", "image": "https://example.com/image.jpg", "classes": ["truck"]},
        )

        assert response.status_code == 200
        assert response.json()["data"]["classes"] == ["truck"]

    def test_capability_mismatch_returns_400(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        handle = FakeHandle(
            DetectionOutput(
                boxes=[[1, 2, 3, 4]],
                scores=[0.9],
                classes=["truck"],
                masks=None,
            )
        )
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="detection")

        response = ray_app_client.post(
            "/inference/segmentation/",
            json={"model": "grounding-dino", "image": "https://example.com/image.jpg"},
        )

        assert response.status_code == 400
        assert "does not support 'segmentation'" in response.json()["detail"]

    def test_missing_model_returns_404(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=None, capability="segmentation")

        response = ray_app_client.post(
            "/inference/segmentation/",
            json={"model": "missing-model", "image": "https://example.com/image.jpg"},
        )

        assert response.status_code == 404
