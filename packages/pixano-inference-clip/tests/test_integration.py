# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""End-to-end test deploying the CLIP plugin on a real (CPU) Ray Serve runtime."""

from __future__ import annotations

import pytest
import ray
from ray import serve

from pixano_inference.ray.app import DeploymentManager, _disable_ray_uv_run_hook
from pixano_inference.ray.config import RayServeConfig


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def serve_runtime():
    """Start a local Ray + Serve runtime (HTTP proxy disabled) for the module."""
    # As the server does before ray.init: under `uv run`, Ray would otherwise relaunch workers
    # through `uv run` in a copy of this directory, where the `../..` path source cannot resolve.
    _disable_ray_uv_run_hook()
    ray.init(
        namespace="pixano-inference-clip-test",
        num_cpus=4,
        num_gpus=0,
        ignore_reinit_error=True,
        include_dashboard=False,
    )
    serve.start(proxy_location="Disabled")
    try:
        yield
    finally:
        serve.shutdown()
        ray.shutdown()


def test_clip_embedding_image_and_text_share_space(serve_runtime):
    """The bundled MobileCLIP2 plugin deploys and embeds image + text into the same space.

    Downloads a small CLIP checkpoint on first run (open_clip HF cache). Asserts each modality
    yields a ``[1, dim]`` vector of the same dimension and that their cosine similarity is a
    finite number in ``[-1, 1]`` — i.e. image and text really land in one comparable space.
    """
    import base64
    import io

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
