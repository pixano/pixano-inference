# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the Transformers VLM plugin (no weights: the transformers objects are faked)."""

from __future__ import annotations

import base64
import io

import pytest
import torch
import transformers
from PIL import Image
from pixano_inference_transformers_vlm import TransformersVLMModel, TransformersVLMParams
from pydantic import ValidationError

from pixano_inference.configs import ModelParamsRegistry
from pixano_inference.models.registry import ModelClassRegistry
from pixano_inference.models.vlm import VLMInput
from pixano_inference.plugins import load_plugin_models
from pixano_inference.ray.config import ModelDeploymentConfig


def _data_uri() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 0, 0)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class _Inputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.calls: dict = {}

    def apply_chat_template(self, messages, add_generation_prompt=False):
        self.calls["template"] = messages
        return "<chat prompt>"

    # Processors take `images` first; the prompt must reach `text`, not `images`.
    def __call__(self, images=None, text=None, **kwargs):
        self.calls["call"] = {"images": images, "text": text, **kwargs}
        return _Inputs(input_ids=torch.tensor([[1, 2, 3, 4]]))

    def decode(self, ids, **kwargs):
        self.calls["decode"] = ids.tolist()
        return "a red square"


class _FakeModel:
    device = torch.device("cpu")

    def generate(self, input_ids, generation_config):
        return torch.cat([input_ids, torch.tensor([[7, 8]])], dim=1)


def _model() -> TransformersVLMModel:
    model = TransformersVLMModel(
        ModelDeploymentConfig(
            name="vlm", capability="vlm", model_class="TransformersVLMModel", model_params={"path": "x"}
        )
    )
    model._processor = _FakeProcessor()
    model._model = _FakeModel()
    return model


def test_entry_point_registers_model_and_params():
    result = load_plugin_models()
    assert "transformers_vlm" in result["loaded"]
    assert ModelClassRegistry.has("TransformersVLMModel")
    assert ModelParamsRegistry.get("TransformersVLMModel") is TransformersVLMParams


def test_params():
    params = TransformersVLMParams(path="llava-hf/llava-1.5-7b-hf", model_type="llava", config={"dtype": "float16"})
    assert params.model_type == "llava"
    assert TransformersVLMParams(path="p").processor_config == {}
    with pytest.raises(ValidationError):
        TransformersVLMParams()


def test_predict_chat_prompt_passes_text_and_images_by_keyword():
    model = _model()
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_uri()}},
                {"type": "text", "text": "What is this?"},
            ],
        }
    ]
    out = model.predict(VLMInput(prompt=prompt, max_new_tokens=8))

    calls = model._processor.calls
    assert calls["template"][0]["content"][0] == {"type": "image"}
    assert calls["call"]["text"] == "<chat prompt>"
    assert len(calls["call"]["images"]) == 1 and isinstance(calls["call"]["images"][0], Image.Image)
    assert calls["decode"] == [7, 8]
    assert out.generated_text == "a red square"
    assert (out.usage.prompt_tokens, out.usage.completion_tokens, out.usage.total_tokens) == (4, 2, 6)


def test_string_prompt_requires_images():
    with pytest.raises(ValueError, match="Images must be provided"):
        _model().predict(VLMInput(prompt="describe", max_new_tokens=8))


def test_generic_fallback_uses_image_text_to_text(monkeypatch: pytest.MonkeyPatch):
    loaded = {}

    class _FakeAuto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded["path"] = path
            return "model"

    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", _FakeAuto)
    assert TransformersVLMModel._load_vlm_model("org/smolvlm", None, "cpu", {}) == "model"
    assert loaded["path"] == "org/smolvlm"
