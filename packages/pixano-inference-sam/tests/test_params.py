# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the SAM2 plugin's registration and typed params."""

import pytest
from pixano_inference_sam import Sam2ImageParams, Sam2VideoParams
from pydantic import ValidationError

from pixano_inference.configs import ModelConfig, ModelParamsRegistry
from pixano_inference.models.registry import ModelClassRegistry
from pixano_inference.plugins import load_plugin_models


def test_entry_point_registers_models_and_params():
    result = load_plugin_models()
    assert "sam2" in result["loaded"]
    assert ModelClassRegistry.has("Sam2ImageModel")
    assert ModelClassRegistry.has("Sam2VideoModel")
    assert ModelParamsRegistry.get("Sam2ImageModel") is Sam2ImageParams
    assert ModelParamsRegistry.get("Sam2VideoModel") is Sam2VideoParams


def test_image_params():
    params = Sam2ImageParams()
    assert (params.path, params.torch_dtype, params.compile) == ("facebook/sam2-hiera-base-plus", "bfloat16", True)
    params = Sam2ImageParams(path="my/model", torch_dtype="float16", compile=False)
    assert (params.path, params.torch_dtype, params.compile) == ("my/model", "float16", False)
    with pytest.raises(ValidationError):
        Sam2ImageParams(torch_dtype="int8")


def test_video_params():
    params = Sam2VideoParams()
    assert (params.path, params.vos_optimized, params.propagate) == ("facebook/sam2-hiera-large", True, True)
    params = Sam2VideoParams(path="my/video-model", vos_optimized=False, propagate=False)
    assert (params.vos_optimized, params.propagate) == (False, False)


def test_model_config_resolves_params_by_name():
    config = ModelConfig(
        name="sam2-image",
        model_class="Sam2ImageModel",
        model_params={"path": "facebook/sam2-hiera-base-plus", "torch_dtype": "float32"},
    )
    assert isinstance(config.model_params, Sam2ImageParams)
    assert config.capability == "segmentation"
    assert config.to_deployment_config().model_params == {
        "path": "facebook/sam2-hiera-base-plus",
        "torch_dtype": "float32",
        "compile": True,
    }
