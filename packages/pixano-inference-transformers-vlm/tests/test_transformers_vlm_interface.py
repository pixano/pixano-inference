# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""What the Transformers VLM declares on ``/v1/models`` (static: no weights, no torch import)."""

from __future__ import annotations

from pixano_inference_transformers_vlm import TransformersVLMModel

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models import ImageRange, VLMInterface


def test_accepts_a_text_prompt_or_chat_messages_with_any_number_of_images():
    model = TransformersVLMModel(
        ModelDeploymentConfig(name="vlm", capability="vlm", model_class="TransformersVLMModel")
    )

    assert model.interface == VLMInterface(prompt=["text", "messages"], images=ImageRange(min=0, max=None))
