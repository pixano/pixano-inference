# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""What the vLLM model declares on ``/v1/models`` (static: no weights, no vllm import)."""

from __future__ import annotations

from pixano_inference_vllm import VLLMVLMModel

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models import ImageRange, VLMInterface


def test_accepts_chat_messages_only():
    model = VLLMVLMModel(ModelDeploymentConfig(name="vllm", capability="vlm", model_class="VLLMVLMModel"))

    assert model.interface == VLMInterface(prompt=["messages"], images=ImageRange(min=0, max=None))
    assert "text" not in model.interface.prompt  # a string prompt is rejected by predict()
