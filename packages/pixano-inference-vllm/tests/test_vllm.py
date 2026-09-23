# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the vLLM plugin.

vLLM needs Linux and a GPU, so these tests substitute a fake ``vllm`` module that mirrors the
parts of its API the model uses.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
from pixano_inference_vllm import VLLMVLMModel, VLLMVLMParams
from pydantic import ValidationError

from pixano_inference.configs import ModelParamsRegistry
from pixano_inference.models.registry import ModelClassRegistry
from pixano_inference.models.vlm import VLMInput
from pixano_inference.plugins import load_plugin_models
from pixano_inference.ray.config import ModelDeploymentConfig


class _SamplingParams(msgspec.Struct):
    temperature: float = 1.0
    max_tokens: int = 16


class _FakeLLM:
    instances: list[_FakeLLM] = []

    def __init__(self, model, **kwargs):
        # vllm.LLM forwards unknown kwargs to EngineArgs, which has no `device` field.
        if "device" in kwargs:
            raise TypeError("EngineArgs.__init__() got an unexpected keyword argument 'device'")
        self.model = model
        self.kwargs = kwargs
        self.chat_calls: list[dict] = []
        _FakeLLM.instances.append(self)

    def chat(self, messages, sampling_params=None, use_tqdm=True):
        self.chat_calls.append({"messages": messages, "sampling_params": sampling_params})
        output = SimpleNamespace(text="two cats", token_ids=[5, 6, 7])
        return [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[output])]


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch):
    module: Any = types.ModuleType("vllm")
    module.LLM = _FakeLLM
    module.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    _FakeLLM.instances.clear()
    return module


def _model(**params) -> VLLMVLMModel:
    return VLLMVLMModel(
        ModelDeploymentConfig(
            name="vllm", capability="vlm", model_class="VLLMVLMModel", model_params={"path": "org/model", **params}
        )
    )


def test_entry_point_registers_model_and_params():
    result = load_plugin_models()
    assert "vllm_vlm" in result["loaded"]
    assert ModelClassRegistry.has("VLLMVLMModel")
    assert ModelParamsRegistry.get("VLLMVLMModel") is VLLMVLMParams


def test_params():
    assert VLLMVLMParams(path="p").config == {}
    with pytest.raises(ValidationError):
        VLLMVLMParams()


def test_load_model_forwards_config_without_device(fake_vllm):
    model = _model(config={"max_model_len": 4096}, processor_config={"mm_processor_kwargs": {"fps": 1}})
    model.load_model()
    llm = _FakeLLM.instances[-1]
    assert llm.model == "org/model"
    assert llm.kwargs == {"max_model_len": 4096, "mm_processor_kwargs": {"fps": 1}, "tensor_parallel_size": 1}


def test_predict(fake_vllm):
    model = _model()
    model.load_model()
    messages = [{"role": "user", "content": [{"type": "text", "text": "How many cats?"}]}]
    out = model.predict(VLMInput(prompt=messages, max_new_tokens=32, temperature=0.2))

    call = _FakeLLM.instances[-1].chat_calls[-1]
    assert call["messages"] == messages
    assert (call["sampling_params"].temperature, call["sampling_params"].max_tokens) == (0.2, 32)
    assert out.generated_text == "two cats"
    assert (out.usage.prompt_tokens, out.usage.completion_tokens, out.usage.total_tokens) == (2, 3, 5)
    assert out.generation_config == {"temperature": 0.2, "max_tokens": 32}


def test_predict_rejects_string_prompt_and_separate_images(fake_vllm):
    model = _model()
    model.load_model()
    with pytest.raises(ValueError, match="chat template"):
        model.predict(VLMInput(prompt="hi", max_new_tokens=4))
    with pytest.raises(ValueError, match="images should be passed in the prompt"):
        model.predict(VLMInput(prompt=[{"role": "user", "content": []}], images=["x"], max_new_tokens=4))


def test_discovery_does_not_import_the_framework():
    """Importing the package (what plugin discovery does at startup) must not load torch/vllm."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import pixano_inference_vllm\n"
        "print(','.join(m for m in ('torch', 'vllm') if m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"discovery imported: {result.stdout.strip()}"
