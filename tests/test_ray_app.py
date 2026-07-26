# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pixano_inference.models.tracking import TrackingOutput
from pixano_inference.ray import app as ray_app_module
from pixano_inference.ray.app import DeploymentManager, create_ray_serve_app
from pixano_inference.ray.config import ModelDeploymentConfig, RayServeConfig
from pixano_inference.schemas import ModelInfo
from pixano_inference.schemas.rle import CompressedRLE
from tests.fakes import FakeHandle


@pytest.fixture
def ray_app_client():
    config = RayServeConfig(num_gpus=0)
    app, _ = create_ray_serve_app(config)
    return TestClient(app)


def _make_tracking_config(name: str = "sam2-video"):
    return ModelDeploymentConfig(name=name, capability="tracking", model_class="Sam2VideoModel")


class TestJobManager:
    """Async unit tests for the in-process JobManager (over Serve handles)."""

    async def test_job_completes(self):
        result = TrackingOutput(
            objects_ids=[1],
            frame_indexes=[0],
            masks=[CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))],
        )
        manager = DeploymentManager(RayServeConfig(num_gpus=0))
        manager._configs["sam2-video"] = _make_tracking_config()
        manager._handles["sam2-video"] = FakeHandle(result)

        job_id = manager.submit_tracking_job("sam2-video", input_data=object())
        assert manager.get_tracking_job(job_id).status == "running"

        await asyncio.sleep(0)  # let the background task run to completion
        job = manager.get_tracking_job(job_id)
        assert job.status == "completed"
        # JobManager stores the camelCase-serialized result.
        assert job.result["frameIndexes"] == [0]
        assert job.processing_time >= 0.0

    async def test_job_records_failure(self):
        manager = DeploymentManager(RayServeConfig(num_gpus=0))
        manager._configs["sam2-video"] = _make_tracking_config()
        manager._handles["sam2-video"] = FakeHandle(None, error=RuntimeError("boom"))

        job_id = manager.submit_tracking_job("sam2-video", input_data=object())
        await asyncio.sleep(0)
        job = manager.get_tracking_job(job_id)
        assert job.status == "failed"
        assert "boom" in job.detail

    async def test_job_store_evicts_over_cap(self):
        from pixano_inference import jobs as jobs_module

        manager = DeploymentManager(RayServeConfig(num_gpus=0))
        manager._configs["sam2-video"] = _make_tracking_config()
        manager._handles["sam2-video"] = FakeHandle(
            TrackingOutput(objects_ids=[1], frame_indexes=[0], masks=[]),
        )
        for _ in range(jobs_module.DEFAULT_MAX_JOBS + 5):
            jid = manager.submit_tracking_job("sam2-video", input_data=object())
            manager.jobs._finalize(jid, status="completed", result={})
        manager.jobs.evict_now()
        assert manager.jobs.count <= jobs_module.DEFAULT_MAX_JOBS


class TestServiceRoutes:
    def test_health(self, ray_app_client: TestClient):
        response = ray_app_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_v1_health(self, ray_app_client: TestClient):
        assert ray_app_client.get("/v1/health").json()["status"] == "healthy"

    def test_ready_with_no_models_is_ready(self, ray_app_client: TestClient):
        response = ray_app_client.get("/v1/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["models_loaded"] == 0

    def test_ready_returns_503_when_model_not_running(
        self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        manager = ray_app_client.app.state.deployment_manager
        monkeypatch.setattr(manager, "model_statuses", lambda: {"sam2-image": "DEPLOYING"})
        response = ray_app_client.get("/v1/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False

    def test_list_models(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        manager = ray_app_client.app.state.deployment_manager
        monkeypatch.setattr(
            manager,
            "list_models",
            lambda: [ModelInfo(name="sam2-image", capability="segmentation", model_class="Sam2ImageModel")],
        )
        monkeypatch.setattr(manager, "model_statuses", lambda: {"sam2-image": "RUNNING"})

        response = ray_app_client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert body[0]["capability"] == "segmentation"
        assert body[0]["status"] == "RUNNING"

    def test_info(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        fake_ray = SimpleNamespace(
            is_initialized=lambda: True,
            cluster_resources=lambda: {"GPU": 4.0},
            available_resources=lambda: {"GPU": 1.5},
            nodes=lambda: [{"Alive": True}, {"Alive": True}],
        )
        monkeypatch.setattr(ray_app_module, "ray", fake_ray)
        manager = ray_app_client.app.state.deployment_manager
        monkeypatch.setattr(
            manager,
            "list_models",
            lambda: [ModelInfo(name="sam2-image", capability="segmentation", model_class="Sam2ImageModel")],
        )
        monkeypatch.setattr(manager, "model_statuses", lambda: {"sam2-image": "RUNNING"})

        response = ray_app_client.get("/v1/info")
        assert response.status_code == 200
        body = response.json()
        assert body["numGpus"] == 4
        assert body["gpusUsed"] == 2.5
        assert body["numNodes"] == 2
        assert body["modelsStatus"] == {"sam2-image": "RUNNING"}
