# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the numpy-only example detector."""

import base64
import io

import pytest
from PIL import Image
from pixano_numpy_detector import NumpyDetector, NumpyDetectorParams
from pydantic import ValidationError

from pixano_inference.configs import ModelConfig
from pixano_inference.models import DetectionInput
from pixano_inference.plugins import load_plugin_models


def _image_with_square() -> str:
    image = Image.new("RGB", (64, 64), (255, 255, 255))
    image.paste((0, 0, 0), (10, 20, 30, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_discovered_through_its_entry_point():
    assert "numpy_detector" in load_plugin_models()["loaded"]


def test_params_defaults_and_validation():
    config = ModelConfig(name="np", model_class="NumpyDetector")
    assert config.to_deployment_config().model_params == {"path": "numpy-detector", "threshold": 20}
    with pytest.raises(ValidationError):
        NumpyDetectorParams(threshold=-1)


def test_detects_the_foreground_box():
    model = NumpyDetector(ModelConfig(name="np", model_class="NumpyDetector").to_deployment_config())
    model.load_model()
    out = model.predict(DetectionInput(image=_image_with_square()))
    assert out.boxes == [[10, 20, 29, 39]]
    assert out.classes == ["object"]
