# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for typed config objects."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pixano_inference.configs import (
    BaseModelParams,
    DeploymentConfig,
    GroundingDINOParams,
    ModelConfig,
    ModelParamsRegistry,
    Sam2ImageParams,
    Sam2VideoParams,
    ServerConfig,
    TransformersVLMParams,
    VLLMVLMParams,
)
from pixano_inference.impls.sam2.image import Sam2ImageModel
from pixano_inference.impls.sam2.video import Sam2VideoModel
from pixano_inference.impls.transformers.grounding_dino import GroundingDINOModel
from pixano_inference.impls.transformers.vlm import TransformersVLMModel
from pixano_inference.models import InferenceModel
from pixano_inference.ray.config import ModelDeploymentConfig


class TestModelParamsRegistry:
    """Tests for ModelParamsRegistry."""

    def test_registered_schemas(self):
        assert ModelParamsRegistry.has("Sam2ImageModel")
        assert ModelParamsRegistry.has("Sam2VideoModel")
        assert ModelParamsRegistry.has("TransformersVLMModel")
        assert ModelParamsRegistry.has("GroundingDINOModel")
        assert ModelParamsRegistry.has("VLLMVLMModel")

    def test_get_returns_correct_class(self):
        assert ModelParamsRegistry.get("Sam2ImageModel") is Sam2ImageParams
        assert ModelParamsRegistry.get("Sam2VideoModel") is Sam2VideoParams
        assert ModelParamsRegistry.get("TransformersVLMModel") is TransformersVLMParams
        assert ModelParamsRegistry.get("GroundingDINOModel") is GroundingDINOParams
        assert ModelParamsRegistry.get("VLLMVLMModel") is VLLMVLMParams

    def test_get_unknown_returns_none(self):
        assert ModelParamsRegistry.get("UnknownModel") is None

    def test_has_unknown_returns_false(self):
        assert not ModelParamsRegistry.has("UnknownModel")

    def test_list_all(self):
        all_schemas = ModelParamsRegistry.list_all()
        assert "Sam2ImageModel" in all_schemas
        assert len(all_schemas) >= 5


class TestBaseModelParams:
    def test_valid(self):
        params = BaseModelParams(path="facebook/sam2-hiera-base-plus")
        assert params.path == "facebook/sam2-hiera-base-plus"

    def test_missing_path(self):
        with pytest.raises(ValidationError):
            BaseModelParams()


class TestSam2Params:
    def test_sam2_image_defaults(self):
        params = Sam2ImageParams()
        assert params.path == "facebook/sam2-hiera-base-plus"
        assert params.torch_dtype == "bfloat16"
        assert params.compile is True

    def test_sam2_image_custom(self):
        params = Sam2ImageParams(path="my/model", torch_dtype="float16", compile=False)
        assert params.path == "my/model"
        assert params.torch_dtype == "float16"
        assert params.compile is False

    def test_sam2_image_invalid_dtype(self):
        with pytest.raises(ValidationError):
            Sam2ImageParams(torch_dtype="int8")

    def test_sam2_video_defaults(self):
        params = Sam2VideoParams()
        assert params.path == "facebook/sam2-hiera-large"
        assert params.vos_optimized is True
        assert params.propagate is True

    def test_sam2_video_custom(self):
        params = Sam2VideoParams(path="my/video-model", vos_optimized=False, propagate=False)
        assert params.vos_optimized is False
        assert params.propagate is False


class TestTransformersParams:
    def test_vlm_minimal(self):
        params = TransformersVLMParams(path="llava-hf/llava-1.5-7b-hf")
        assert params.path == "llava-hf/llava-1.5-7b-hf"
        assert params.processor_config == {}
        assert params.config == {}
        assert params.model_type is None

    def test_vlm_full(self):
        params = TransformersVLMParams(
            path="llava-hf/llava-1.5-7b-hf",
            processor_config={"use_fast": True},
            config={"torch_dtype": "float16"},
            model_type="llava",
        )
        assert params.model_type == "llava"

    def test_grounding_dino_minimal(self):
        params = GroundingDINOParams(path="IDEA-Research/grounding-dino-base")
        assert params.path == "IDEA-Research/grounding-dino-base"

    def test_vlm_missing_path(self):
        with pytest.raises(ValidationError):
            TransformersVLMParams()

    def test_grounding_dino_missing_path(self):
        with pytest.raises(ValidationError):
            GroundingDINOParams()


class TestVLLMParams:
    def test_minimal(self):
        params = VLLMVLMParams(path="my/vllm-model")
        assert params.path == "my/vllm-model"
        assert params.config == {}
        assert params.processor_config == {}

    def test_missing_path(self):
        with pytest.raises(ValidationError):
            VLLMVLMParams()


class TestModelConfig:
    def test_valid_with_typed_params(self):
        config = ModelConfig(
            name="sam2-image",
            model_class="Sam2ImageModel",
            model_params=Sam2ImageParams(),
        )
        assert config.name == "sam2-image"
        assert config.capability == "segmentation"
        assert isinstance(config.model_params, Sam2ImageParams)

    def test_valid_from_dict_auto_resolves(self):
        config = ModelConfig(
            name="sam2-image",
            model_class="Sam2ImageModel",
            model_params={"path": "facebook/sam2-hiera-base-plus", "torch_dtype": "float32"},
        )
        assert isinstance(config.model_params, Sam2ImageParams)
        assert config.model_params.torch_dtype == "float32"
        assert config.capability == "segmentation"

    def test_unknown_model_class_raises(self):
        with pytest.raises(ValidationError, match="Unknown model_class"):
            ModelConfig(name="test", model_class="UnknownModel")

    def test_invalid_model_params_key_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(
                name="test",
                model_class="Sam2ImageModel",
                model_params={"path": "my/model", "typo_field": True},
            )

    def test_type_input_derives_capability(self):
        config = ModelConfig(
            name="grounding-dino",
            model_class=GroundingDINOModel,
            model_params=GroundingDINOParams(path="IDEA-Research/grounding-dino-base"),
        )
        assert config.model_class is GroundingDINOModel
        assert config.model_class_name == "GroundingDINOModel"
        assert config.capability == "detection"

    def test_external_class_resolves_capability(self):
        from pixano_inference.models import SegmentationModel
        from pixano_inference.models.segmentation import SegmentationInput, SegmentationOutput

        class ExternalSegmentationModel(SegmentationModel):
            def load_model(self):
                pass

            def predict(self, input: SegmentationInput) -> SegmentationOutput:
                raise NotImplementedError

        config = ModelConfig(
            name="external-seg",
            model_class=ExternalSegmentationModel,
            model_params={"path": "some/model"},
        )

        assert config.capability == "segmentation"
        dc = config.to_deployment_config()
        assert dc.capability == "segmentation"
        assert dc.model_class == "ExternalSegmentationModel"

    def test_non_inference_model_class_raises(self):
        class PlainPythonClass:
            pass

        with pytest.raises(ValidationError, match="must inherit from InferenceModel"):
            ModelConfig(name="plain", model_class=PlainPythonClass)

    def test_unsupported_http_base_raises(self):
        class UnsupportedModel(InferenceModel):
            def load_model(self) -> None:
                pass

            def predict(self, input):  # pragma: no cover - test only
                return input

        with pytest.raises(ValidationError, match="not supported by the HTTP inference API"):
            ModelConfig(name="unsupported", model_class=UnsupportedModel)

    def test_to_deployment_config_with_typed_params(self):
        config = ModelConfig(
            name="sam2-image",
            model_class=Sam2ImageModel,
            model_params=Sam2ImageParams(path="facebook/sam2-hiera-base-plus", torch_dtype="float32"),
            deployment=DeploymentConfig(num_gpus=1, max_batch_size=4),
        )
        dc = config.to_deployment_config()
        assert isinstance(dc, ModelDeploymentConfig)
        assert dc.name == "sam2-image"
        assert dc.capability == "segmentation"
        assert dc.model_class == "Sam2ImageModel"
        assert dc.model_params == {
            "path": "facebook/sam2-hiera-base-plus",
            "torch_dtype": "float32",
            "compile": True,
        }
        assert dc.resources.num_gpus == 1
        assert dc.max_batch_size == 4

    def test_to_deployment_config_with_dict_params(self):
        from pixano_inference.models.segmentation import SegmentationInput, SegmentationModel, SegmentationOutput

        class CustomSegmentationModel(SegmentationModel):
            def load_model(self) -> None:
                pass

            def predict(self, input: SegmentationInput) -> SegmentationOutput:  # pragma: no cover - test only
                raise NotImplementedError

        config = ModelConfig(
            name="custom",
            model_class=CustomSegmentationModel,
            model_params={"path": "my/model", "custom": True},
        )
        dc = config.to_deployment_config()
        assert dc.capability == "segmentation"
        assert dc.model_params == {"path": "my/model", "custom": True}

    def test_deployment_defaults(self):
        config = ModelConfig(name="test", model_class="Sam2ImageModel")
        dc = config.to_deployment_config()
        assert dc.resources.num_gpus == 0.0
        assert dc.resources.num_cpus == 1.0
        assert dc.autoscaling.min_replicas == 0
        assert dc.autoscaling.max_replicas == 4
        assert dc.max_batch_size == 8


class TestDeploymentConfig:
    def test_defaults(self):
        dep = DeploymentConfig()
        assert dep.num_gpus == 0.0
        assert dep.num_cpus == 1.0
        assert dep.min_replicas == 0
        assert dep.max_replicas == 4
        assert dep.max_batch_size == 8

    def test_custom_values(self):
        dep = DeploymentConfig(
            num_gpus=2,
            min_replicas=1,
            max_replicas=8,
            max_batch_size=16,
        )
        assert dep.num_gpus == 2
        assert dep.min_replicas == 1

    def test_invalid_max_replicas(self):
        with pytest.raises(ValidationError):
            DeploymentConfig(max_replicas=0)


class TestServerConfig:
    def test_defaults(self):
        sc = ServerConfig()
        assert sc.host == "0.0.0.0"
        assert sc.port == 7463
        assert sc.models == []

    def test_to_ray_serve_config(self):
        sc = ServerConfig(
            host="localhost",
            port=8000,
            models=[
                ModelConfig(
                    name="sam2-image",
                    model_class="Sam2ImageModel",
                    model_params=Sam2ImageParams(),
                )
            ],
        )
        rsc = sc.to_ray_serve_config()
        assert rsc.host == "localhost"
        assert rsc.port == 8000
        assert len(rsc.models) == 1
        assert rsc.models[0].name == "sam2-image"
        assert rsc.models[0].capability == "segmentation"


class TestConfigLoaderIntegration:
    def test_load_sam2_python(self, tmp_path):
        config_file = tmp_path / "test_config.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "from pixano_inference.configs import DeploymentConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="sam2-image",\n'
            '        model_class="Sam2ImageModel",\n'
            '        model_params={"path": "facebook/sam2-hiera-base-plus", "torch_dtype": "float32"},\n'
            "        deployment=DeploymentConfig(num_gpus=0, min_replicas=0, max_replicas=2, max_batch_size=8),\n"
            "    ),\n"
            "]\n"
        )

        from pixano_inference.ray.config_loader import ConfigLoader

        configs = ConfigLoader(config_file).load()
        assert len(configs) == 1
        assert configs[0].name == "sam2-image"
        assert configs[0].capability == "segmentation"
        assert configs[0].model_params["torch_dtype"] == "float32"

    def test_load_python_invalid_model_class_raises(self, tmp_path):
        config_file = tmp_path / "bad_config.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="test",\n'
            '        model_class="bogus_model",\n'
            '        model_params={"path": "facebook/sam2-hiera-base-plus"},\n'
            "    ),\n"
            "]\n"
        )

        from pixano_inference.ray.config_loader import ConfigLoader

        with pytest.raises(ValidationError, match="Unknown model_class"):
            ConfigLoader(config_file).load()

    def test_load_python_invalid_model_params_raises(self, tmp_path):
        config_file = tmp_path / "bad_params.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="test",\n'
            '        model_class="Sam2ImageModel",\n'
            '        model_params={"path": "facebook/sam2-hiera-base-plus", "unknown_param": True},\n'
            "    ),\n"
            "]\n"
        )

        from pixano_inference.ray.config_loader import ConfigLoader

        with pytest.raises(ValidationError):
            ConfigLoader(config_file).load()
