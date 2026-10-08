# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

from __future__ import annotations

import asyncio
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pixano_inference.models import DetectionInterface, DetectionModel, DetectionOutput, register_model
from pixano_inference.models.tracking import TrackedFrame, TrackedObject, TrackingOutput
from pixano_inference.ray import app as ray_app_module
from pixano_inference.ray.app import DeploymentManager, create_ray_serve_app
from pixano_inference.ray.config import ModelDeploymentConfig, RayServeConfig
from pixano_inference.schemas import ModelInfo, TrackingInterface
from pixano_inference.schemas.rle import CompressedRLE
from tests.fakes import FakeHandle


@pytest.fixture
def ray_app_client():
    config = RayServeConfig(num_gpus=0)
    app, _ = create_ray_serve_app(config)
    return TestClient(app)


def _make_tracking_config(name: str = "sam2-video"):
    return ModelDeploymentConfig(name=name, capability="tracking", model_class="Sam2VideoModel")


@register_model("DescribedStubDetector")
class DescribedStubDetector(DetectionModel):
    """A stub that describes itself; the manager must publish what the replica returns."""

    def load_model(self) -> None:
        pass

    def predict(self, input):  # noqa: ANN001
        return DetectionOutput(boxes=[], scores=[], classes=[])

    @property
    def interface(self) -> DetectionInterface:
        return DetectionInterface(classes="closed", class_names=["thing"], outputs=["box", "score", "class"])


class _FakeInterfaceMethod:
    """``handle.get_interface.remote().result(timeout_s=...)`` as the manager calls it."""

    def __init__(self, value=None, *, error: Exception | None = None, delay_s: float = 0.0) -> None:
        self._value = value
        self._error = error
        self._delay_s = delay_s
        self.timeouts: list[float] = []

    def remote(self):
        return self

    def result(self, timeout_s: float):
        self.timeouts.append(timeout_s)
        time.sleep(self._delay_s)
        if self._error is not None:
            raise self._error
        return self._value


class TestModelInterface:
    """The interface is fetched once per deployment, cached, and never fails a deploy."""

    @staticmethod
    def _deploying_manager(monkeypatch: pytest.MonkeyPatch, get_interface: _FakeInterfaceMethod) -> DeploymentManager:
        manager = DeploymentManager(RayServeConfig(num_gpus=0))
        monkeypatch.setattr(manager, "_preflight_resource_check", lambda config: None)
        monkeypatch.setattr(manager, "_run_serve_app", lambda app, name, timeout_s: None)
        monkeypatch.setattr(manager, "_best_effort_delete", lambda name: None)
        monkeypatch.setattr(ray_app_module, "_has_live_ray_node", lambda: True)
        monkeypatch.setattr(
            ray_app_module.serve, "get_app_handle", lambda name: SimpleNamespace(get_interface=get_interface)
        )
        return manager

    @staticmethod
    def _config(name: str = "described") -> ModelDeploymentConfig:
        return ModelDeploymentConfig(name=name, capability="detection", model_class="DescribedStubDetector")

    def test_deploy_fetches_the_interface_once_and_undeploy_forgets_it(self, monkeypatch: pytest.MonkeyPatch):
        declared = DescribedStubDetector(self._config()).interface
        get_interface = _FakeInterfaceMethod(declared)
        manager = self._deploying_manager(monkeypatch, get_interface)

        manager.deploy_model(self._config())

        assert manager.get_model_interface("described") == declared
        assert manager.get_model_interface("described") == declared  # served from the cache
        assert get_interface.timeouts == [ray_app_module._INTERFACE_TIMEOUT_S]  # one bounded call

        manager.undeploy_model("described")
        assert manager.get_model_interface("described") is None

    def test_a_model_without_interface_lists_with_none(self, monkeypatch: pytest.MonkeyPatch):
        manager = self._deploying_manager(monkeypatch, _FakeInterfaceMethod(None))

        manager.deploy_model(self._config())

        assert manager.get_model_interface("described") is None
        assert "described" in manager._configs

    def test_a_fetch_failure_warns_and_never_fails_the_deploy(self, monkeypatch: pytest.MonkeyPatch, caplog):
        manager = self._deploying_manager(monkeypatch, _FakeInterfaceMethod(error=TimeoutError("replica busy")))

        with caplog.at_level("WARNING", logger=ray_app_module.__name__):
            manager.deploy_model(self._config())

        assert "described" in manager._configs
        assert manager.get_model_interface("described") is None
        assert "Could not fetch the interface of 'described'" in caplog.text

    def test_a_malformed_interface_is_dropped_not_served(self, monkeypatch: pytest.MonkeyPatch, caplog):
        manager = self._deploying_manager(monkeypatch, _FakeInterfaceMethod({"capability": "detection"}))

        with caplog.at_level("WARNING", logger=ray_app_module.__name__):
            manager.deploy_model(self._config())

        assert manager.get_model_interface("described") is None
        assert "Could not fetch the interface" in caplog.text

    def test_shutdown_during_the_fetch_still_aborts_startup(self, monkeypatch: pytest.MonkeyPatch):
        """Ctrl-C while the replica is slow to answer must not be swallowed as a fetch failure."""
        manager = self._deploying_manager(monkeypatch, _FakeInterfaceMethod(None, delay_s=2.0))
        manager._should_abort = lambda: True

        with pytest.raises(ray_app_module.StartupAborted):
            manager.deploy_model(self._config())

    def test_unknown_model_has_no_interface(self):
        assert DeploymentManager(RayServeConfig(num_gpus=0)).get_model_interface("nope") is None


class TestJobManager:
    """Async unit tests for the in-process JobManager (over Serve handles)."""

    async def test_job_completes(self):
        mask = CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))
        result = TrackingOutput(frames=[TrackedFrame(frame_index=0, objects=[TrackedObject(track_id=1, mask=mask)])])
        manager = DeploymentManager(RayServeConfig(num_gpus=0))
        manager._configs["sam2-video"] = _make_tracking_config()
        manager._handles["sam2-video"] = FakeHandle(result)

        job_id = manager.submit_tracking_job("sam2-video", input_data=object())
        assert manager.get_tracking_job(job_id).status == "running"

        await asyncio.sleep(0)  # let the background task run to completion
        job = manager.get_tracking_job(job_id)
        assert job.status == "completed"
        # JobManager stores the camelCase-serialized result.
        assert job.result["frames"][0]["frameIndex"] == 0
        assert job.result["frames"][0]["objects"][0]["trackId"] == 1
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
            TrackingOutput(frames=[]),
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
        assert body[0]["interface"] is None  # nothing declared (or a model package predating 0.7.1)

    def test_list_models_publishes_the_declared_interface(
        self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        manager = ray_app_client.app.state.deployment_manager
        declared = {
            "sam2-video": TrackingInterface(prompts=["points", "box", "mask"], interval=True, outputs=["mask"]),
            "bytetrack": None,
        }
        monkeypatch.setattr(
            manager,
            "list_models",
            lambda: [ModelInfo(name=name, capability="tracking", model_class="X") for name in declared],
        )
        monkeypatch.setattr(manager, "model_statuses", lambda: {name: "RUNNING" for name in declared})
        monkeypatch.setattr(manager, "get_model_interface", declared.get)

        body = ray_app_client.get("/v1/models").json()

        assert body[0]["interface"] == {
            "capability": "tracking",
            "prompts": ["points", "box", "mask"],
            "promptFree": False,
            "classes": "none",
            "classNames": None,
            "thresholds": [],
            "interval": True,
            "outputs": ["mask"],
        }
        assert body[1]["interface"] is None

    def test_deploy_route_returns_the_interface(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        manager = ray_app_client.app.state.deployment_manager
        declared = DescribedStubDetector(
            ModelDeploymentConfig(name="described", capability="detection", model_class="DescribedStubDetector")
        ).interface

        def _deploy(config):
            manager._configs[config.name] = config
            manager._interface_cache[config.name] = declared

        monkeypatch.setattr(manager, "deploy_model", _deploy)
        monkeypatch.setattr(manager, "model_statuses", lambda: {"described": "RUNNING"})

        response = ray_app_client.post("/v1/models", json={"name": "described", "modelClass": "DescribedStubDetector"})

        assert response.status_code == 201, response.text
        assert response.json()["interface"] == {
            "capability": "detection",
            "classes": "closed",
            "classNames": ["thing"],
            "thresholds": [],
            "outputs": ["box", "score", "class"],
        }

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


class TestStartupWatchdog:
    """Startup must stay interruptible and bounded (see _run_with_node_watchdog)."""

    def test_returns_value_and_propagates_errors(self):
        assert ray_app_module._run_with_node_watchdog(lambda: 42, timeout_s=5, what="x") == 42

        def _boom():
            raise KeyError("nope")

        with pytest.raises(KeyError):
            ray_app_module._run_with_node_watchdog(_boom, timeout_s=5, what="x")

    def test_dead_node_aborts_with_log_pointer(self, monkeypatch: pytest.MonkeyPatch):
        """A raylet that dies mid-call must surface the log, not block forever."""
        seen = {"polls": 0}

        def _liveness():
            seen["polls"] += 1
            return seen["polls"] < 2  # alive once, then dead

        monkeypatch.setattr(ray_app_module, "_has_live_ray_node", _liveness)
        monkeypatch.setattr(ray_app_module, "_ray_session_log_hint", lambda: "/tmp/ray/x/raylet.err")

        with pytest.raises(RuntimeError, match=r"raylet\.err") as excinfo:
            ray_app_module._run_with_node_watchdog(lambda: time.sleep(30), timeout_s=30, what="Ray Serve startup")
        assert "the Ray node died" in str(excinfo.value)

    def test_times_out(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ray_app_module, "_has_live_ray_node", lambda: True)
        with pytest.raises(TimeoutError, match="did not complete within"):
            ray_app_module._run_with_node_watchdog(lambda: time.sleep(30), timeout_s=0.5, what="Ray Serve startup")

    def test_should_abort_cuts_the_wait_short(self, monkeypatch: pytest.MonkeyPatch):
        """Ctrl-C reaches uvicorn as a flag; the watchdog must honour it."""
        monkeypatch.setattr(ray_app_module, "_has_live_ray_node", lambda: True)
        with pytest.raises(ray_app_module.StartupAborted, match="shutdown was requested"):
            ray_app_module._run_with_node_watchdog(
                lambda: time.sleep(30), timeout_s=30, what="Ray Serve startup", should_abort=lambda: True
            )

    def test_on_poll_can_abort_early(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ray_app_module, "_has_live_ray_node", lambda: True)

        def _bad_status():
            raise RuntimeError("Serve deployment 'm' status is DEPLOY_FAILED.")

        with pytest.raises(RuntimeError, match="DEPLOY_FAILED"):
            ray_app_module._run_with_node_watchdog(
                lambda: time.sleep(30), timeout_s=30, what="Serve deployment 'm'", on_poll=_bad_status
            )

    def test_disables_rays_uv_run_hook(self):
        """Ray would otherwise relaunch workers via `uv run` in a tree stripped of pyproject.toml."""
        from ray._private import ray_constants

        original = ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
        try:
            ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = True
            ray_app_module._disable_ray_uv_run_hook()
            assert ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is False
            assert os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
        finally:
            ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = original


class TestLifespanStartup:
    """The lifespan must not block the event loop, or Ctrl-C is inert during startup."""

    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, start):
        monkeypatch.setattr(ray_app_module, "_start_ray_and_serve", start)

        async def _no_drain(config, timeout_s=None):
            return None

        monkeypatch.setattr(ray_app_module, "_drain_ray_and_serve", _no_drain)

    async def test_event_loop_stays_responsive_during_startup(self, monkeypatch: pytest.MonkeyPatch):
        self._patch(monkeypatch, lambda config, should_abort=None: time.sleep(0.6))

        ticks = 0

        async def _tick():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.02)

        app, _ = create_ray_serve_app(RayServeConfig(num_gpus=0))
        ticker = asyncio.create_task(_tick())
        async with app.router.lifespan_context(app):
            pass
        ticker.cancel()

        # Blocking the loop (the original bug) would leave this at ~1.
        assert ticks > 5, f"event loop was blocked during startup (ticks={ticks})"

    async def test_abort_flag_stops_startup_before_deploying(self, monkeypatch: pytest.MonkeyPatch):
        self._patch(monkeypatch, lambda config, should_abort=None: None)

        deployed = []
        config = RayServeConfig(num_gpus=0, models=[_make_tracking_config("m1")])
        app, manager = create_ray_serve_app(config, should_abort=lambda: True)
        monkeypatch.setattr(manager, "deploy_model", lambda cfg: deployed.append(cfg.name))

        with pytest.raises(ray_app_module.StartupAborted):
            async with app.router.lifespan_context(app):
                pass
        assert deployed == []
