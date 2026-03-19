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
from pixano_inference.ray.config import ModelDeploymentConfig
from pixano_inference.tasks import ImageTask, MultimodalImageNLPTask, NLPTask, VideoTask


class TestModelParamsRegistry:
    """Tests for ModelParamsRegistry."""

    def test_registered_schemas(self):
        """All built-in model params schemas are registered."""
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
    """Tests for BaseModelParams."""

    def test_valid(self):
        params = BaseModelParams(path="facebook/sam2-hiera-base-plus")
        assert params.path == "facebook/sam2-hiera-base-plus"

    def test_missing_path(self):
        with pytest.raises(ValidationError):
            BaseModelParams()


class TestSam2Params:
    """Tests for SAM2 params schemas."""

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
    """Tests for Transformers params schemas."""

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
    """Tests for vLLM params schemas."""

    def test_minimal(self):
        params = VLLMVLMParams(path="my/vllm-model")
        assert params.path == "my/vllm-model"
        assert params.config == {}
        assert params.processor_config == {}

    def test_missing_path(self):
        with pytest.raises(ValidationError):
            VLLMVLMParams()


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_valid_with_typed_params(self):
        config = ModelConfig(
            name="sam2-image",
            task="image_mask_generation",
            model_class="Sam2ImageModel",
            model_params=Sam2ImageParams(),
        )
        assert config.name == "sam2-image"
        assert isinstance(config.model_params, Sam2ImageParams)

    def test_valid_from_dict_auto_resolves(self):
        """model_params dict is auto-resolved to typed schema via registry."""
        config = ModelConfig(
            name="sam2-image",
            task="image_mask_generation",
            model_class="Sam2ImageModel",
            model_params={"path": "facebook/sam2-hiera-base-plus", "torch_dtype": "float32"},
        )
        assert isinstance(config.model_params, Sam2ImageParams)
        assert config.model_params.torch_dtype == "float32"

    def test_invalid_task_raises(self):
        with pytest.raises(ValidationError, match="Unknown task"):
            ModelConfig(
                name="test",
                task="invalid_task",
                model_class="Sam2ImageModel",
            )

    def test_invalid_model_params_key_raises(self):
        """Typo in model_params field name raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelConfig(
                name="test",
                task="image_mask_generation",
                model_class="Sam2ImageModel",
                model_params={"path": "my/model", "typo_field": True},
            )

    def test_unknown_model_class_falls_back_to_dict(self):
        """Unknown model_class keeps model_params as raw dict."""
        config = ModelConfig(
            name="custom",
            task="image_mask_generation",
            model_class="MyCustomModel",
            model_params={"path": "some/model", "custom_key": 42},
        )
        assert isinstance(config.model_params, dict)
        assert config.model_params["custom_key"] == 42

    def test_to_deployment_config_with_typed_params(self):
        config = ModelConfig(
            name="sam2-image",
            task="image_mask_generation",
            model_class="Sam2ImageModel",
            model_params=Sam2ImageParams(path="facebook/sam2-hiera-base-plus", torch_dtype="float32"),
            deployment=DeploymentConfig(num_gpus=1, max_batch_size=4),
        )
        dc = config.to_deployment_config()
        assert isinstance(dc, ModelDeploymentConfig)
        assert dc.name == "sam2-image"
        assert dc.task == "image_mask_generation"
        assert dc.model_class == "Sam2ImageModel"
        assert dc.model_params == {
            "path": "facebook/sam2-hiera-base-plus",
            "torch_dtype": "float32",
            "compile": True,
        }
        assert dc.resources.num_gpus == 1
        assert dc.max_batch_size == 4

    def test_to_deployment_config_with_dict_params(self):
        config = ModelConfig(
            name="custom",
            task="image_mask_generation",
            model_class="MyCustomModel",
            model_params={"path": "my/model", "custom": True},
        )
        dc = config.to_deployment_config()
        assert dc.model_params == {"path": "my/model", "custom": True}

    def test_deployment_defaults(self):
        config = ModelConfig(
            name="test",
            task="image_mask_generation",
            model_class="Sam2ImageModel",
        )
        dc = config.to_deployment_config()
        assert dc.resources.num_gpus == 0.0
        assert dc.resources.num_cpus == 1.0
        assert dc.autoscaling.min_replicas == 0
        assert dc.autoscaling.max_replicas == 4
        assert dc.max_batch_size == 8

    def test_model_module_passed_through(self):
        config = ModelConfig(
            name="custom",
            task="image_mask_generation",
            model_class="MyCustomModel",
            model_module="my_package.models",
            model_params={"path": "some/model"},
        )
        dc = config.to_deployment_config()
        assert dc.model_module == "my_package.models"

    def test_task_accepts_enum(self):
        """Task enum is converted to its string value."""
        config = ModelConfig(
            name="test",
            task=ImageTask.MASK_GENERATION,
            model_class="Sam2ImageModel",
        )
        assert config.task == "image_mask_generation"

    def test_task_accepts_all_enum_types(self):
        """All Task enum subclasses are accepted."""
        for task_enum, expected in [
            (VideoTask.MASK_GENERATION, "video_mask_generation"),
            (NLPTask.CAUSAL_LM, "causal_lm"),
            (MultimodalImageNLPTask.CAPTIONING, "image_captioning"),
        ]:
            config = ModelConfig(
                name="test",
                task=task_enum,
                model_class="MyCustomModel",
                model_params={"path": "some/model"},
            )
            assert config.task == expected

    def test_model_class_accepts_type(self):
        """A class type is converted to its __name__ string."""

        class MockModel:
            pass

        config = ModelConfig(
            name="test",
            task="image_mask_generation",
            model_class=MockModel,
            model_params={"path": "some/model"},
        )
        assert config.model_class == "MockModel"

    def test_enum_task_to_deployment_config(self):
        """to_deployment_config produces string task from enum input."""
        config = ModelConfig(
            name="test",
            task=ImageTask.MASK_GENERATION,
            model_class="Sam2ImageModel",
        )
        dc = config.to_deployment_config()
        assert dc.task == "image_mask_generation"
        assert isinstance(dc.task, str)

    def test_class_model_class_to_deployment_config(self):
        """to_deployment_config produces string model_class from type input."""

        class MockModel:
            pass

        config = ModelConfig(
            name="test",
            task="image_mask_generation",
            model_class=MockModel,
            model_params={"path": "some/model"},
        )
        dc = config.to_deployment_config()
        assert dc.model_class == "MockModel"
        assert isinstance(dc.model_class, str)

    def test_class_model_class_auto_resolves_params(self):
        """Passing a class for model_class still triggers params registry resolution."""

        class Sam2ImageModel:
            pass

        config = ModelConfig(
            name="test",
            task="image_mask_generation",
            model_class=Sam2ImageModel,
            model_params={"path": "facebook/sam2-hiera-base-plus", "torch_dtype": "float32"},
        )
        assert isinstance(config.model_params, Sam2ImageParams)
        assert config.model_params.torch_dtype == "float32"


class TestDeploymentConfig:
    """Tests for DeploymentConfig."""

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
    """Tests for ServerConfig."""

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
                    task="image_mask_generation",
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


class TestConfigLoaderIntegration:
    """Integration test: loading Python configs through the ConfigLoader."""

    def test_load_sam2_python(self, tmp_path):
        """A Python config with known model classes triggers typed validation."""
        config_file = tmp_path / "test_config.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "from pixano_inference.configs import DeploymentConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="sam2-image",\n'
            '        task="image_mask_generation",\n'
            '        model_class="Sam2ImageModel",\n'
            '        model_params={"path": "facebook/sam2-hiera-base-plus", "torch_dtype": "float32"},\n'
            "        deployment=DeploymentConfig(num_gpus=0, min_replicas=0, max_replicas=2, max_batch_size=8),\n"
            "    ),\n"
            "]\n"
        )

        from pixano_inference.ray.config_loader import ConfigLoader

        loader = ConfigLoader(config_file)
        configs = loader.load()

        assert len(configs) == 1
        assert configs[0].name == "sam2-image"
        assert configs[0].task == "image_mask_generation"
        assert configs[0].model_params["torch_dtype"] == "float32"

    def test_load_python_invalid_task_raises(self, tmp_path):
        """A Python config with an invalid task string raises."""
        config_file = tmp_path / "bad_config.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="test",\n'
            '        task="bogus_task",\n'
            '        model_class="Sam2ImageModel",\n'
            '        model_params={"path": "facebook/sam2-hiera-base-plus"},\n'
            "    ),\n"
            "]\n"
        )

        from pixano_inference.ray.config_loader import ConfigLoader

        loader = ConfigLoader(config_file)
        with pytest.raises(ValidationError, match="Unknown task"):
            loader.load()

    def test_load_python_invalid_model_params_raises(self, tmp_path):
        """A Python config with invalid model_params for a known model_class raises."""
        config_file = tmp_path / "bad_params.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="test",\n'
            '        task="image_mask_generation",\n'
            '        model_class="Sam2ImageModel",\n'
            '        model_params={"path": "facebook/sam2-hiera-base-plus", "unknown_param": True},\n'
            "    ),\n"
            "]\n"
        )

        from pixano_inference.ray.config_loader import ConfigLoader

        loader = ConfigLoader(config_file)
        with pytest.raises(ValidationError):
            loader.load()
