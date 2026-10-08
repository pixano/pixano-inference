# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""What the CLIP model declares on ``/v1/models``: the modalities, and the vector size once loaded."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from contextlib import nullcontext
from typing import Any

import pytest
from pixano_inference_clip import OpenClipEmbeddingModel

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models import EmbeddingInterface


def _model() -> OpenClipEmbeddingModel:
    return OpenClipEmbeddingModel(
        ModelDeploymentConfig(
            name="clip", capability="embedding", model_class="OpenClipEmbeddingModel", model_params={"path": "x"}
        )
    )


def test_embeds_images_and_text_in_one_space_with_the_size_unknown_until_loaded():
    assert _model().interface == EmbeddingInterface(modalities=["image", "text"], dim=None)


def test_the_vector_size_is_read_from_the_weights_at_load(monkeypatch: pytest.MonkeyPatch):
    """``load_model`` probes the loaded weights once, so the listing shows ``dim`` before any request.

    ``open_clip`` and ``torch`` are faked with the little the probe touches, so this runs without
    either installed; the real libraries are exercised by the integration test.
    """

    class _Tensor:
        def __init__(self, *shape: int) -> None:
            self.shape = shape

        def to(self, device):
            return self

    class _FakeClip:
        def eval(self):
            return self

        def encode_text(self, tokens: _Tensor) -> _Tensor:
            return _Tensor(tokens.shape[0], 512)

    def _not_called(*args, **kwargs):
        raise AssertionError("compile=False: torch.compile must not run")

    fake_open_clip: Any = types.ModuleType("open_clip")
    fake_open_clip.__spec__ = importlib.machinery.ModuleSpec("open_clip", loader=None)  # satisfies find_spec
    fake_open_clip.create_model_and_transforms = lambda spec, pretrained=None, device=None: (_FakeClip(), None, None)
    fake_open_clip.get_tokenizer = lambda spec: (lambda texts: _Tensor(len(texts), 8))
    fake_torch: Any = types.ModuleType("torch")
    fake_torch.inference_mode = nullcontext
    fake_torch.compile = _not_called
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("pixano_inference_clip.model.resolve_device", lambda config: "cpu")

    model = _model()
    model.load_model()

    assert model.interface == EmbeddingInterface(modalities=["image", "text"], dim=512)
    assert model.metadata["device"] == "cpu"
