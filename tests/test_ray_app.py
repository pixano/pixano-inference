# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

from __future__ import annotations

import asyncio
import json
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
from pixano_inference.ray.app import DeploymentManager, create_ray_serve_app
from pixano_inference.ray.config import ModelDeploymentConfig, RayServeConfig
from pixano_inference.schemas import ModelInfo
from pixano_inference.schemas.nd_array import NDArrayFloat
from pixano_inference.schemas.rle import CompressedRLE


class FakeResponse:
    """Awaitable stand-in for a Serve DeploymentResponse.

    Resolves immediately by default; ``pending=True`` never resolves until cancelled (used
    to test the async job cancel path without a completion race).
    """

    def __init__(self, result, *, error: Exception | None = None, pending: bool = False):
        self._result = result
        self._error = error
        self._pending = pending
        self.cancelled = False

    def __await__(self):
        import asyncio

        async def _resolve():
            if self._pending:
                await asyncio.Event().wait()  # never set; cancelled via task.cancel()
            if self._error is not None:
                raise self._error
            return self._result

        return _resolve().__await__()

    def cancel(self):
        self.cancelled = True


class FakeRemoteMethod:
    def __init__(self, result, *, error: Exception | None = None, pending: bool = False):
        self._result = result
        self._error = error
        self._pending = pending
        self.last_input = None
        self.last_response: FakeResponse | None = None

    def remote(self, input_data):
        self.last_input = input_data
        self.last_response = FakeResponse(self._result, error=self._error, pending=self._pending)
        return self.last_response


class FakeHandle:
    def __init__(self, result, *, error: Exception | None = None, pending: bool = False):
        self.predict = FakeRemoteMethod(result, error=error, pending=pending)


def _mask_to_numeric_rle(mask: np.ndarray) -> list[int]:
    flat_mask = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current = 0
    run_length = 0

    for value in flat_mask:
        if value == current:
            run_length += 1
            continue
        counts.append(run_length)
        run_length = 1
        current = int(value)

    counts.append(run_length)
    return counts


@pytest.fixture
def ray_app_client():
    config = RayServeConfig(num_gpus=0)
    app, _ = create_ray_serve_app(config)
    return TestClient(app)


def _make_tracking_config(name: str = "sam2-video"):
    return ModelDeploymentConfig(name=name, capability="tracking", model_class="Sam2VideoModel")


class TestJobManager:
    """Async unit tests for the in-process tracking-job mechanism (over Serve handles)."""

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
        assert job.result["frame_indexes"] == [0]
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
        from pixano_inference.ray import app as app_module

        manager = DeploymentManager(RayServeConfig(num_gpus=0))
        manager._configs["sam2-video"] = _make_tracking_config()
        manager._handles["sam2-video"] = FakeHandle(
            TrackingOutput(objects_ids=[1], frame_indexes=[0], masks=[]),
        )
        # Submit more than the cap; finalized jobs beyond the cap are evicted.
        for _ in range(app_module._MAX_JOBS + 5):
            jid = manager.submit_tracking_job("sam2-video", input_data=object())
            manager._finalize_job(jid, status="completed", result={})
        manager._evict_terminal_jobs()
        assert len(manager._tracking_jobs) <= app_module._MAX_JOBS


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

    def test_ready_with_no_models_is_ready(self, ray_app_client: TestClient):
        # No models configured -> readiness is trivially satisfied.
        response = ray_app_client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["models_loaded"] == 0

    def test_ready_returns_503_when_model_not_running(
        self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        manager = ray_app_client.app.state.deployment_manager
        monkeypatch.setattr(manager, "model_statuses", lambda: {"sam2-image": "DEPLOYING"})
        response = ray_app_client.get("/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False

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

    def test_segmentation_route(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        result = SegmentationOutput(
            masks=[[CompressedRLE.from_mask(np.array([[1, 0], [0, 1]], dtype=np.uint8))]],
            scores=NDArrayFloat(values=[0.95], shape=[1, 1]),
            mask_logits=NDArrayFloat(values=[0.1, 0.2, 0.3, 0.4], shape=[1, 2, 2]),
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
            json={
                "model": "sam2-image",
                "image": "https://example.com/image.jpg",
                "mask_input": {"values": [0.1, 0.2, 0.3, 0.4], "shape": [1, 2, 2]},
                "return_logits": True,
            },
        )

        assert response.status_code == 200
        assert response.json()["metadata"]["capability"] == "segmentation"
        assert response.json()["data"]["mask_logits"] == {
            "values": [0.1, 0.2, 0.3, 0.4],
            "shape": [1, 2, 2],
        }
        assert handle.predict.last_input.image == "https://example.com/image.jpg"
        assert handle.predict.last_input.mask_input.values == [0.1, 0.2, 0.3, 0.4]
        assert handle.predict.last_input.return_logits is True

    def test_segmentation_binary_route(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        result = SegmentationOutput(
            masks=[[CompressedRLE.from_mask(np.array([[1, 0], [0, 1]], dtype=np.uint8))]],
            scores=NDArrayFloat(values=[0.95], shape=[1, 1]),
            mask_logits=NDArrayFloat(values=[0.1, 0.2, 0.3, 0.4], shape=[1, 2, 2]),
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="segmentation")

        response = ray_app_client.post(
            "/inference/segmentation/binary",
            files=[
                (
                    "metadata",
                    (
                        "metadata.json",
                        json.dumps(
                            {
                                "model": "sam2-image",
                                "mask_input": {"values": [0.1, 0.2, 0.3, 0.4], "shape": [1, 2, 2]},
                                "return_logits": True,
                            }
                        ),
                        "application/json",
                    ),
                ),
                ("image", ("image.png", b"binary-image", "image/png")),
            ],
        )

        assert response.status_code == 200
        assert handle.predict.last_input.image == b"binary-image"
        assert handle.predict.last_input.mask_input.values == [0.1, 0.2, 0.3, 0.4]
        assert handle.predict.last_input.return_logits is True

    def test_segmentation_binary_route_accepts_legacy_text_metadata(
        self,
        ray_app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        result = SegmentationOutput(
            masks=[[CompressedRLE.from_mask(np.array([[1, 0], [0, 1]], dtype=np.uint8))]],
            scores=NDArrayFloat(values=[0.95], shape=[1, 1]),
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="segmentation")

        response = ray_app_client.post(
            "/inference/segmentation/binary",
            data={"metadata": json.dumps({"model": "sam2-image"})},
            files={"image": ("image.png", b"binary-image", "image/png")},
        )

        assert response.status_code == 200
        assert handle.predict.last_input.image == b"binary-image"

    def test_segmentation_binary_route_accepts_large_metadata_file_part(
        self,
        ray_app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        result = SegmentationOutput(
            masks=[[CompressedRLE.from_mask(np.array([[1, 0], [0, 1]], dtype=np.uint8))]],
            scores=NDArrayFloat(values=[0.95], shape=[1, 1]),
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="segmentation")
        large_metadata = json.dumps(
            {
                "model": "sam2-image",
                "image_embedding": {"values": [1.0], "shape": [1]},
                "high_resolution_features": [
                    {
                        "values": [0.5] * 400000,
                        "shape": [400000],
                    }
                ],
            }
        )

        response = ray_app_client.post(
            "/inference/segmentation/binary",
            files=[
                ("metadata", ("metadata.json", large_metadata, "application/json")),
                ("image", ("image.png", b"binary-image", "image/png")),
            ],
        )

        assert response.status_code == 200
        assert handle.predict.last_input.high_resolution_features[0].shape == [400000]

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
                "propagate": False,
                "objects_ids": [1],
                "frame_indexes": [0],
            },
        )

        assert response.status_code == 200
        assert response.json()["metadata"]["capability"] == "tracking"
        assert handle.predict.last_input.propagate is False
        assert handle.predict.last_input.video == ["frame-0001.png"]

    def test_tracking_route_parses_interval_keyframes(
        self,
        ray_app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        result = TrackingOutput(
            objects_ids=[1],
            frame_indexes=[1, 2],
            masks=[
                CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8)),
                CompressedRLE.from_mask(np.array([[1, 0], [0, 0]], dtype=np.uint8)),
            ],
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="tracking")
        keyframe_mask = CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))

        response = ray_app_client.post(
            "/inference/tracking/",
            json={
                "model": "sam2-video",
                "video": ["frame-0001.png", "frame-0002.png", "frame-0003.png"],
                "objects_ids": [1],
                "frame_indexes": [1],
                "propagate": True,
                "interval": {"start_frame": 1, "end_frame": 2, "direction": "forward"},
                "keyframes": [{"frame_index": 1, "mask": keyframe_mask.model_dump(mode="json")}],
            },
        )

        assert response.status_code == 200
        assert handle.predict.last_input.interval.start_frame == 1
        assert handle.predict.last_input.interval.end_frame == 2
        assert handle.predict.last_input.keyframes[0].mask.size == keyframe_mask.size

    def test_tracking_job_route_accepts_numeric_rle_keyframe_masks(
        self,
        ray_app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        result = TrackingOutput(
            objects_ids=[1],
            frame_indexes=[1, 2],
            masks=[
                CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8)),
                CompressedRLE.from_mask(np.array([[1, 0], [0, 0]], dtype=np.uint8)),
            ],
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="tracking")

        keyframe_mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        numeric_rle = _mask_to_numeric_rle(keyframe_mask)

        response = ray_app_client.post(
            "/inference/tracking/jobs/",
            json={
                "model": "sam2-video",
                "video": ["frame-0001.png", "frame-0002.png", "frame-0003.png"],
                "objects_ids": [1],
                "frame_indexes": [1],
                "propagate": True,
                "interval": {"start_frame": 1, "end_frame": 2, "direction": "forward"},
                "keyframes": [{"frame_index": 1, "mask": {"size": [2, 2], "counts": numeric_rle}}],
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "running"
        assert handle.predict.last_input.keyframes[0].mask.size == [2, 2]
        np.testing.assert_array_equal(handle.predict.last_input.keyframes[0].mask.to_mask(), keyframe_mask)

    def test_tracking_binary_route(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        result = TrackingOutput(
            objects_ids=[1],
            frame_indexes=[0],
            masks=[CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))],
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="tracking")

        response = ray_app_client.post(
            "/inference/tracking/binary",
            files=[
                (
                    "metadata",
                    (
                        "metadata.json",
                        json.dumps(
                            {
                                "model": "sam2-video",
                                "objects_ids": [1],
                                "propagate": False,
                                "frame_indexes": [0],
                            }
                        ),
                        "application/json",
                    ),
                ),
                ("frames", ("frame-0001.png", b"frame-0", "image/png")),
            ],
        )

        assert response.status_code == 200
        assert handle.predict.last_input.video == [b"frame-0"]
        assert handle.predict.last_input.propagate is False

    def test_tracking_binary_route_accepts_legacy_text_metadata(
        self,
        ray_app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        result = TrackingOutput(
            objects_ids=[1],
            frame_indexes=[0],
            masks=[CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))],
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="tracking")

        response = ray_app_client.post(
            "/inference/tracking/binary",
            data={
                "metadata": json.dumps(
                    {
                        "model": "sam2-video",
                        "objects_ids": [1],
                        "frame_indexes": [0],
                    }
                )
            },
            files=[("frames", ("frame-0001.png", b"frame-0", "image/png"))],
        )

        assert response.status_code == 200
        assert handle.predict.last_input.video == [b"frame-0"]

    def test_tracking_job_submit_and_status(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        result = TrackingOutput(
            objects_ids=[1],
            frame_indexes=[0],
            masks=[CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))],
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="tracking")
        manager = ray_app_client.app.state.deployment_manager

        submit_response = ray_app_client.post(
            "/inference/tracking/jobs/",
            json={
                "model": "sam2-video",
                "video": ["frame-0001.png"],
                "propagate": False,
                "objects_ids": [1],
                "frame_indexes": [0],
            },
        )

        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]
        assert submit_response.json()["status"] == "running"
        assert handle.predict.last_input.video == ["frame-0001.png"]

        # Unknown job -> 404.
        assert ray_app_client.get("/inference/tracking/jobs/does-not-exist").status_code == 404

        # Drive the job to completion deterministically, then poll via HTTP.
        manager._finalize_job(job_id, status="completed", result=result.model_dump())
        completed = ray_app_client.get(f"/inference/tracking/jobs/{job_id}")
        assert completed.status_code == 200
        assert completed.json()["status"] == "completed"
        assert completed.json()["data"]["frame_indexes"] == [0]

    def test_tracking_job_binary_route_accepts_uploaded_frames(
        self,
        ray_app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        result = TrackingOutput(
            objects_ids=[1],
            frame_indexes=[0],
            masks=[CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))],
        )
        handle = FakeHandle(result)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="tracking")

        response = ray_app_client.post(
            "/inference/tracking/jobs/binary",
            files=[
                (
                    "metadata",
                    (
                        "metadata.json",
                        json.dumps(
                            {
                                "model": "sam2-video",
                                "objects_ids": [1],
                                "propagate": False,
                                "frame_indexes": [0],
                            }
                        ),
                        "application/json",
                    ),
                ),
                ("frames", ("frame-0001.png", b"frame-0", "image/png")),
            ],
        )

        assert response.status_code == 200
        assert response.json()["status"] == "running"
        assert handle.predict.last_input.video == [b"frame-0"]

    def test_tracking_job_cancel_route_marks_job_canceled(
        self,
        ray_app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ):
        result = TrackingOutput(
            objects_ids=[1],
            frame_indexes=[0],
            masks=[CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))],
        )
        # Pending response so the job stays 'running' until we cancel it.
        handle = FakeHandle(result, pending=True)
        self._install_manager_stubs(ray_app_client, monkeypatch, handle=handle, capability="tracking")
        manager = ray_app_client.app.state.deployment_manager

        submit_response = ray_app_client.post(
            "/inference/tracking/jobs/",
            json={
                "model": "sam2-video",
                "video": ["frame-0001.png"],
                "propagate": False,
                "objects_ids": [1],
                "frame_indexes": [0],
            },
        )

        job_id = submit_response.json()["job_id"]
        cancel_response = ray_app_client.delete(f"/inference/tracking/jobs/{job_id}")
        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "canceled"
        # The Serve response was cancelled.
        assert manager._tracking_jobs[job_id].response.cancelled is True

        polled_response = ray_app_client.get(f"/inference/tracking/jobs/{job_id}")
        assert polled_response.status_code == 200
        assert polled_response.json()["status"] == "canceled"

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
