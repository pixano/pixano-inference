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


def test_deploy_installed_plugin_model(serve_runtime):
    """An entry-point plugin (installed package, no torch) deploys and predicts."""
    import base64
    import io

    pytest.importorskip("pixano_numpy_detector")  # importing it also registers the model
    from PIL import Image

    from pixano_inference.models import DetectionInput
    from pixano_inference.plugins import ensure_models_loaded

    ensure_models_loaded(force=True)

    manager = DeploymentManager(RayServeConfig(num_gpus=0))
    manager.deploy_model(
        ModelDeploymentConfig(
            name="np-det",
            capability="detection",
            model_class="NumpyDetector",
            resources=ResourceConfig(num_gpus=0.0, num_cpus=1.0),
        )
    )
    try:
        # White image with a red square from (10, 10) to (40, 40).
        image = Image.new("RGB", (64, 64), (255, 255, 255))
        for x in range(10, 41):
            for y in range(10, 41):
                image.putpixel((x, y), (255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

        handle = manager.get_handle("np-det")
        result = handle.predict.remote(DetectionInput(image=data_uri, classes=None)).result(timeout_s=30)
        assert result.classes == ["object"]
        x1, y1, x2, y2 = result.boxes[0]
        assert 8 <= x1 <= 12 and 8 <= y1 <= 12 and 38 <= x2 <= 42 and 38 <= y2 <= 42
    finally:
        manager.undeploy_model("np-det")


def test_clip_embedding_image_and_text_share_space(serve_runtime):
    """The bundled MobileCLIP2 plugin deploys and embeds image + text into the same space.

    Downloads a small CLIP checkpoint on first run (open_clip HF cache). Asserts each modality
    yields a ``[1, dim]`` vector of the same dimension and that their cosine similarity is a
    finite number in ``[-1, 1]`` — i.e. image and text really land in one comparable space.
    """
    import base64
    import io

    pytest.importorskip("open_clip")
    pytest.importorskip("pixano_inference_clip")  # importing it also registers the model
    import numpy as np
    from PIL import Image

    from pixano_inference.configs.base import DeploymentConfig, ModelConfig
    from pixano_inference.models import EmbeddingInput
    from pixano_inference.plugins import ensure_models_loaded

    ensure_models_loaded(force=True)

    # Build the deployment config the production way, so the registered OpenClipParams
    # defaults (MobileCLIP2-S2 / dfndr2b) are resolved and dumped into model_params.
    deployment_config = ModelConfig(
        name="clip",
        model_class="OpenClipEmbeddingModel",
        deployment=DeploymentConfig(num_gpus=0.0, num_cpus=2.0, timeout_s=600.0),
    ).to_deployment_config()

    manager = DeploymentManager(RayServeConfig(num_gpus=0))
    manager.deploy_model(deployment_config)
    try:
        image = Image.new("RGB", (64, 64), (0, 128, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

        handle = manager.get_handle("clip")
        text_out = handle.predict.remote(EmbeddingInput(text="a blue square")).result(timeout_s=300)
        image_out = handle.predict.remote(EmbeddingInput(image=data_uri)).result(timeout_s=300)

        text_vec = text_out.embeddings.to_numpy()
        image_vec = image_out.embeddings.to_numpy()
        assert text_vec.shape == (1, text_out.dim)
        assert image_vec.shape == (1, image_out.dim)
        assert text_out.dim == image_out.dim  # one shared space for both modalities

        cosine = float(np.dot(text_vec[0], image_vec[0]))  # vectors are L2-normalized by default
        assert np.isfinite(cosine)
        assert -1.001 <= cosine <= 1.001
    finally:
        manager.undeploy_model("clip")
