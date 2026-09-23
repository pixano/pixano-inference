# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for typed config objects."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field, ValidationError

from pixano_inference.configs import (
    BaseModelParams,
    DeploymentConfig,
    ModelConfig,
    ModelParamsRegistry,
    ServerConfig,
    register_model_params,
)
from pixano_inference.models import InferenceModel, register_model
from pixano_inference.models.detection import DetectionInput, DetectionModel, DetectionOutput
from pixano_inference.models.segmentation import SegmentationInput, SegmentationModel, SegmentationOutput
from pixano_inference.ray.config import ModelDeploymentConfig


# Framework-free models registered by name, standing in for a model package's classes: the
# config machinery is the same for every model, so the core tests need no model package.
@register_model_params("ConfigTestSegmenter")
class ConfigTestParams(BaseModelParams):
    """Params with defaults for every field, like a plugin that ships a default checkpoint."""

    path: str = "org/config-test-model"
    dtype: Literal["float32", "float16"] = "float32"
    compile: bool = True


@register_model("ConfigTestSegmenter")
class ConfigTestSegmenter(SegmentationModel):
    def load_model(self) -> None:
        pass

    def predict(self, input: SegmentationInput) -> SegmentationOutput:  # pragma: no cover - test only
        raise NotImplementedError


@register_model_params("ConfigTestDetector")
class ConfigTestDetectorParams(BaseModelParams):
    """Params whose ``path`` is required, like a plugin that serves any checkpoint."""

    processor_config: dict = Field(default_factory=dict)


@register_model("ConfigTestDetector")
class ConfigTestDetector(DetectionModel):
    def load_model(self) -> None:
        pass

    def predict(self, input: DetectionInput) -> DetectionOutput:  # pragma: no cover - test only
        raise NotImplementedError


class TestModelParamsRegistry:
    """Tests for ModelParamsRegistry."""

    def test_registered_schemas(self):
        assert ModelParamsRegistry.has("ConfigTestSegmenter")
        assert ModelParamsRegistry.has("ConfigTestDetector")

    def test_get_returns_correct_class(self):
        assert ModelParamsRegistry.get("ConfigTestSegmenter") is ConfigTestParams
        assert ModelParamsRegistry.get("ConfigTestDetector") is ConfigTestDetectorParams

    def test_get_unknown_returns_none(self):
        assert ModelParamsRegistry.get("UnknownModel") is None

    def test_has_unknown_returns_false(self):
        assert not ModelParamsRegistry.has("UnknownModel")

    def test_list_all(self):
        all_schemas = ModelParamsRegistry.list_all()
        assert "ConfigTestSegmenter" in all_schemas
        assert len(all_schemas) >= 2


class TestBaseModelParams:
    def test_valid(self):
        params = BaseModelParams(path="facebook/sam2-hiera-base-plus")
        assert params.path == "facebook/sam2-hiera-base-plus"

    def test_missing_path(self):
        with pytest.raises(ValidationError):
            BaseModelParams()


class TestModelConfig:
    def test_valid_with_typed_params(self):
        config = ModelConfig(
            name="seg",
            model_class="ConfigTestSegmenter",
            model_params=ConfigTestParams(),
        )
        assert config.name == "seg"
        assert config.capability == "segmentation"
        assert isinstance(config.model_params, ConfigTestParams)

    def test_valid_from_dict_auto_resolves(self):
        config = ModelConfig(
            name="seg",
            model_class="ConfigTestSegmenter",
            model_params={"path": "org/other-model", "dtype": "float16"},
        )
        assert isinstance(config.model_params, ConfigTestParams)
        assert config.model_params.dtype == "float16"
        assert config.capability == "segmentation"

    def test_unknown_model_class_raises(self):
        with pytest.raises(ValidationError, match="Unknown model_class"):
            ModelConfig(name="test", model_class="UnknownModel")

    def test_invalid_model_params_key_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(
                name="test",
                model_class="ConfigTestSegmenter",
                model_params={"path": "my/model", "typo_field": True},
            )

    def test_type_input_derives_capability(self):
        config = ModelConfig(
            name="det",
            model_class=ConfigTestDetector,
            model_params=ConfigTestDetectorParams(path="org/detector"),
        )
        assert config.model_class is ConfigTestDetector
        assert config.model_class_name == "ConfigTestDetector"
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
            name="seg",
            model_class=ConfigTestSegmenter,
            model_params=ConfigTestParams(path="org/other-model", dtype="float16"),
            deployment=DeploymentConfig(num_gpus=1, max_batch_size=4),
        )
        dc = config.to_deployment_config()
        assert isinstance(dc, ModelDeploymentConfig)
        assert dc.name == "seg"
        assert dc.capability == "segmentation"
        assert dc.model_class == "ConfigTestSegmenter"
        assert dc.model_params == {"path": "org/other-model", "dtype": "float16", "compile": True}
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
        config = ModelConfig(name="test", model_class="ConfigTestSegmenter")
        dc = config.to_deployment_config()
        assert dc.resources.num_gpus == 0.0
        assert dc.resources.num_cpus == 1.0
        assert dc.autoscaling.min_replicas == 1
        assert dc.autoscaling.max_replicas == 4
        assert dc.max_batch_size == 1
        assert dc.max_ongoing_requests == 2


class TestDeploymentConfig:
    def test_defaults(self):
        dep = DeploymentConfig()
        assert dep.num_gpus == 0.0
        assert dep.num_cpus == 1.0
        assert dep.min_replicas == 1
        assert dep.max_replicas == 4
        assert dep.max_batch_size == 1
        assert dep.max_ongoing_requests == 2

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
                    name="seg",
                    model_class="ConfigTestSegmenter",
                    model_params=ConfigTestParams(),
                )
            ],
        )
        rsc = sc.to_ray_serve_config()
        assert rsc.host == "localhost"
        assert rsc.port == 8000
        assert len(rsc.models) == 1
        assert rsc.models[0].name == "seg"
        assert rsc.models[0].capability == "segmentation"


class TestConfigLoaderIntegration:
    def test_load_python_config(self, tmp_path):
        config_file = tmp_path / "test_config.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "from pixano_inference.configs import DeploymentConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="seg",\n'
            '        model_class="ConfigTestSegmenter",\n'
            '        model_params={"path": "org/other-model", "dtype": "float16"},\n'
            "        deployment=DeploymentConfig(num_gpus=0, min_replicas=0, max_replicas=2, max_batch_size=8),\n"
            "    ),\n"
            "]\n"
        )

        from pixano_inference.ray.config_loader import ConfigLoader

        configs = ConfigLoader(config_file).load()
        assert len(configs) == 1
        assert configs[0].name == "seg"
        assert configs[0].capability == "segmentation"
        assert configs[0].model_params["dtype"] == "float16"

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
            '        model_class="ConfigTestSegmenter",\n'
            '        model_params={"path": "org/other-model", "unknown_param": True},\n'
            "    ),\n"
            "]\n"
        )

        from pixano_inference.ray.config_loader import ConfigLoader

        with pytest.raises(ValidationError):
            ConfigLoader(config_file).load()


class TestPluginParamDefaultsColdStart:
    """Plugin param defaults must resolve even when ModelConfig is the first thing constructed.

    Regression: ``_resolve_model_params`` is a before-validator and used to consult
    ``ModelParamsRegistry`` before plugins were loaded (loading only happened later, in
    ``model_post_init``) — so a plugin model referenced from a config file with no explicit
    ``model_params`` lost its defaults and the replica failed with ``KeyError('path')``.
    A fresh interpreter is the only faithful reproduction of that cold state.
    """

    def test_plugin_defaults_resolve_in_fresh_interpreter(self):
        pytest.importorskip("pixano_numpy_detector")  # framework-free example plugin (dev group)
        import subprocess
        import sys

        script = (
            "from pixano_inference.configs import ModelConfig\n"
            "c = ModelConfig(name='np', model_class='NumpyDetector')\n"
            "dep = c.to_deployment_config()\n"
            "assert dep.model_params == {'path': 'numpy-detector', 'threshold': 20}, dep.model_params\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ok" in result.stdout
