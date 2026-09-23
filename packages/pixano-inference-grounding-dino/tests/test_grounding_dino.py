# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the Grounding DINO plugin (no weights: the transformers objects are faked)."""

from __future__ import annotations

import base64
import io

import pytest
import torch
from PIL import Image
from pixano_inference_grounding_dino import GroundingDINOModel, GroundingDINOParams
from pydantic import ValidationError

from pixano_inference.configs import ModelParamsRegistry
from pixano_inference.models.detection import DetectionInput
from pixano_inference.models.registry import ModelClassRegistry
from pixano_inference.plugins import load_plugin_models
from pixano_inference.ray.config import ModelDeploymentConfig


def _data_uri(size=(32, 16)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class _Inputs(dict):
    """Stand-in for a ``BatchFeature``: a dict with attribute access and ``.to()``."""

    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.calls: dict = {}

    def __call__(self, images=None, text=None, **kwargs):
        self.calls["call"] = {"images": images, "text": text, **kwargs}
        return _Inputs(input_ids=torch.tensor([[1, 2, 3]]), pixel_values=torch.zeros(1))

    # Mirrors the transformers >= 4.51 signature: no `box_threshold`, no **kwargs.
    def post_process_grounded_object_detection(
        self, outputs, input_ids=None, threshold=0.25, text_threshold=0.25, target_sizes=None, text_labels=None
    ):
        self.calls["post"] = {"threshold": threshold, "text_threshold": text_threshold, "target_sizes": target_sizes}
        return [
            {
                "boxes": torch.tensor([[1.4, 2.6, 10.0, 12.2]]),
                "scores": torch.tensor([0.75]),
                "text_labels": ["cat"],
            }
        ]


class _FakeModel:
    device = torch.device("cpu")

    def __call__(self, **inputs):
        return object()


def _model() -> GroundingDINOModel:
    model = GroundingDINOModel(
        ModelDeploymentConfig(
            name="gd", capability="detection", model_class="GroundingDINOModel", model_params={"path": "x"}
        )
    )
    model._processor = _FakeProcessor()
    model._model = _FakeModel()
    return model


def test_entry_point_registers_model_and_params():
    result = load_plugin_models()
    assert "grounding_dino" in result["loaded"]
    assert ModelClassRegistry.has("GroundingDINOModel")
    assert ModelParamsRegistry.get("GroundingDINOModel") is GroundingDINOParams


def test_params():
    assert GroundingDINOParams(path="IDEA-Research/grounding-dino-tiny").config == {}
    with pytest.raises(ValidationError):
        GroundingDINOParams()


def test_predict_maps_thresholds_and_text_labels():
    model = _model()
    out = model.predict(
        DetectionInput(image=_data_uri(), classes=["cat", "dog"], box_threshold=0.4, text_threshold=0.3)
    )
    calls = model._processor.calls
    assert calls["call"]["text"] == "cat. dog"
    assert isinstance(calls["call"]["images"], Image.Image)
    assert calls["post"]["threshold"] == 0.4
    assert calls["post"]["text_threshold"] == 0.3
    assert calls["post"]["target_sizes"] == [(16, 32)]
    assert out.boxes == [[1, 3, 10, 12]]
    assert out.scores == pytest.approx([0.75])
    assert out.classes == ["cat"]


def test_predict_requires_classes():
    with pytest.raises(ValueError, match="requires 'classes'"):
        _model().predict(DetectionInput(image=_data_uri(), classes=None))
