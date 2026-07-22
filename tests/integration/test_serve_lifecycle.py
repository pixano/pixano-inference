# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Integration tests exercising the real Ray Serve deployment path on CPU.

These start a local Ray + Serve runtime and deploy a trivial numpy-only stub model, then
verify deploy -> RUNNING -> predict -> undeploy and the pre-flight resource check. They are
marked ``integration`` (deselect with ``-m 'not integration'``) and need no GPU.
"""

from __future__ import annotations

import sys

import pytest


pytestmark = pytest.mark.integration

ray = pytest.importorskip("ray")
serve = pytest.importorskip("ray.serve")

import ray.cloudpickle as _cloudpickle  # noqa: E402

from pixano_inference.models import DetectionModel, DetectionOutput, register_model  # noqa: E402
from pixano_inference.ray.app import DeploymentManager  # noqa: E402
from pixano_inference.ray.config import (  # noqa: E402
    AutoscalingConfig,
    ModelDeploymentConfig,
    RayServeConfig,
    ResourceConfig,
)


@register_model("IntegrationStubDetector")
class IntegrationStubDetector(DetectionModel):
    """A framework-free stub detector for integration tests (no torch, no weights)."""

    def load_model(self) -> None:
        """Mark the model loaded."""
        self._loaded = True

    def predict(self, input):  # noqa: ANN001, D102
        return DetectionOutput(boxes=[[1, 2, 3, 4]], scores=[0.9], classes=["thing"], masks=None)


# The stub class lives in this (non-importable-in-worker) test module, so serialise it by
# value into the Serve replica rather than by reference.
_cloudpickle.register_pickle_by_value(sys.modules[__name__])


@pytest.fixture(scope="module")
def serve_runtime():
    """Start a local Ray + Serve runtime (HTTP proxy disabled) for the module."""
    ray.init(namespace="pixano-inference-test", num_cpus=4, ignore_reinit_error=True, include_dashboard=False)
    serve.start(proxy_location="Disabled")
    try:
        yield
    finally:
        serve.shutdown()
        ray.shutdown()


def _stub_config(name: str = "stub-det", **overrides) -> ModelDeploymentConfig:
    params: dict = {
        "name": name,
        "capability": "detection",
        "model_class": "IntegrationStubDetector",
        "resources": ResourceConfig(num_gpus=0.0, num_cpus=1.0),
    }
    params.update(overrides)
    return ModelDeploymentConfig(**params)


def test_deploy_predict_and_undeploy(serve_runtime):
    from pixano_inference.models import DetectionInput

    manager = DeploymentManager(RayServeConfig(num_gpus=0))
    config = _stub_config()
    manager.deploy_model(config)
    try:
        # The Serve app reached RUNNING.
        assert manager.model_statuses()["stub-det"] == "RUNNING"
        assert manager.readiness()["ready"] is True

        # Inference dispatches through the Serve handle and returns the model output.
        handle = manager.get_handle("stub-det")
        result = handle.predict.remote(DetectionInput(image="ignored-by-stub", classes=["thing"])).result(timeout_s=30)
        assert result.classes == ["thing"]
        assert result.boxes == [[1, 2, 3, 4]]
    finally:
        manager.undeploy_model("stub-det")

    # The app is gone after undeploy.
    assert "stub-det" not in manager._configs
    assert "stub-det" not in serve.status().applications


def test_fixed_replica_count_when_min_equals_max(serve_runtime):
    manager = DeploymentManager(RayServeConfig(num_gpus=0))
    config = _stub_config(name="stub-fixed", autoscaling=AutoscalingConfig(min_replicas=1, max_replicas=1))
    manager.deploy_model(config)
    try:
        assert manager.model_statuses()["stub-fixed"] == "RUNNING"
    finally:
        manager.undeploy_model("stub-fixed")


def test_preflight_rejects_insufficient_gpus(serve_runtime):
    manager = DeploymentManager(RayServeConfig(num_gpus=0))
    # The local cluster has 0 GPUs; requesting one must fail fast (not hang for 600s).
    config = _stub_config(name="stub-gpu", resources=ResourceConfig(num_gpus=1.0, num_cpus=1.0))
    with pytest.raises(ValueError, match="Insufficient GPUs"):
        manager.deploy_model(config)


def test_batched_deployment_serves_predictions(serve_runtime):
    """A deployment with max_batch_size > 1 wires @serve.batch and still returns per-request."""
    from pixano_inference.models import DetectionInput

    manager = DeploymentManager(RayServeConfig(num_gpus=0))
    config = _stub_config(name="stub-batch", max_batch_size=4, batch_wait_timeout_s=0.05)
    manager.deploy_model(config)
    try:
        handle = manager.get_handle("stub-batch")
        # Fire several concurrent requests so the batch handler collects more than one.
        responses = [handle.predict.remote(DetectionInput(image=f"img-{i}", classes=["thing"])) for i in range(4)]
        results = [r.result(timeout_s=30) for r in responses]
        assert all(r.classes == ["thing"] for r in results)
    finally:
        manager.undeploy_model("stub-batch")


def test_double_deploy_raises(serve_runtime):
    manager = DeploymentManager(RayServeConfig(num_gpus=0))
    config = _stub_config(name="stub-dup")
    manager.deploy_model(config)
    try:
        with pytest.raises(ValueError, match="already deployed"):
            manager.deploy_model(_stub_config(name="stub-dup"))
    finally:
        manager.undeploy_model("stub-dup")
