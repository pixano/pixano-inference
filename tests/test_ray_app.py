# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================


from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from pixano_inference.configs.base import ModelConfig
from pixano_inference.ray import app as ray_app_module
from pixano_inference.ray.app import create_ray_serve_app
from pixano_inference.ray.config import RayServeConfig
from pixano_inference.ray.config_loader import ConfigLoader


@pytest.fixture
def ray_app_client():
    """Create a test client from the Ray Serve FastAPI app."""
    config = RayServeConfig(num_gpus=0)
    app, _ = create_ray_serve_app(config)
    return TestClient(app)


class TestManagementRoutesRemoved:
    """Verify that model management endpoints are not exposed."""

    def test_instantiate_model_not_found(self, ray_app_client: TestClient):
        response = ray_app_client.post("/providers/instantiate")
        assert response.status_code in (404, 405)

    def test_deploy_model_not_found(self, ray_app_client: TestClient):
        response = ray_app_client.post("/models/deploy")
        assert response.status_code in (404, 405)

    def test_delete_model_not_found(self, ray_app_client: TestClient):
        response = ray_app_client.delete("/providers/model/test-model")
        assert response.status_code in (404, 405)


class TestServiceRoutesStillWork:
    """Verify that service endpoints remain functional."""

    def test_health(self, ray_app_client: TestClient):
        response = ray_app_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_list_models(self, ray_app_client: TestClient):
        response = ray_app_client.get("/app/models/")
        assert response.status_code == 200
        assert response.json() == []

    def test_settings_uses_ray_resource_api(self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch):
        fake_ray = SimpleNamespace(
            is_initialized=lambda: True,
            cluster_resources=lambda: {"GPU": 4.0},
            available_resources=lambda: {"GPU": 1.5},
        )
        monkeypatch.setattr(ray_app_module, "ray", fake_ray)

        response = ray_app_client.get("/app/settings/")
        assert response.status_code == 200

        payload = response.json()
        assert payload["num_gpus"] == 4
        assert payload["gpus_used"] == 2.5
        assert payload["gpu_to_model"] == {}

    def test_settings_without_initialized_ray_reports_no_gpu(
        self, ray_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        fake_ray = SimpleNamespace(is_initialized=lambda: False)
        monkeypatch.setattr(ray_app_module, "ray", fake_ray)

        response = ray_app_client.get("/app/settings/")
        assert response.status_code == 200

        payload = response.json()
        assert payload["num_gpus"] == 0
        assert payload["gpus_used"] == 0
        assert payload["gpu_to_model"] == {}


class TestConfigLoader:
    """Tests for ConfigLoader Python config parsing."""

    def _write_python_config(self, tmp_path: Path, model_configs: str) -> Path:
        """Write a Python config to a temp file and return its path."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "from pixano_inference.configs import DeploymentConfig\n"
            "\n"
            f"models = {model_configs}\n"
        )
        return config_file

    def test_parse_python_entry(self, tmp_path: Path):
        """Python config with ModelConfig should be parsed correctly."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "from pixano_inference.configs import DeploymentConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="sam2-image",\n'
            '        task="image_mask_generation",\n'
            '        model_class="Sam2ImageModel",\n'
            '        model_params={"path": "facebook/sam2-hiera-base-plus", "torch_dtype": "bfloat16"},\n'
            "        deployment=DeploymentConfig(num_gpus=1, min_replicas=0, max_replicas=2, max_batch_size=8),\n"
            "    ),\n"
            "]\n"
        )
        configs = ConfigLoader(config_file).load()
        assert len(configs) == 1
        cfg = configs[0]
        assert cfg.name == "sam2-image"
        assert cfg.task == "image_mask_generation"
        assert cfg.model_class == "Sam2ImageModel"
        assert cfg.model_params["path"] == "facebook/sam2-hiera-base-plus"

    def test_deployment_settings_parsed(self, tmp_path: Path):
        """Deployment settings should map to resources and autoscaling configs."""
        config_file = tmp_path / "config.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "from pixano_inference.configs import DeploymentConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="test-model",\n'
            '        task="image_mask_generation",\n'
            '        model_class="Sam2ImageModel",\n'
            '        model_params={"path": "test/model"},\n'
            "        deployment=DeploymentConfig(\n"
            "            num_gpus=2, num_cpus=4, min_replicas=1, max_replicas=8,\n"
            "            max_batch_size=16, downscale_delay_s=120.0, upscale_delay_s=10.0,\n"
            "        ),\n"
            "    ),\n"
            "]\n"
        )
        configs = ConfigLoader(config_file).load()
        cfg = configs[0]
        assert cfg.resources.num_gpus == 2
        assert cfg.resources.num_cpus == 4
        assert cfg.autoscaling.min_replicas == 1
        assert cfg.autoscaling.max_replicas == 8
        assert cfg.max_batch_size == 16
        assert cfg.autoscaling.downscale_delay_s == 120.0
        assert cfg.autoscaling.upscale_delay_s == 10.0

    def test_empty_config_returns_empty_list(self, tmp_path: Path):
        """A Python config with an empty models list should return an empty list."""
        config_file = tmp_path / "empty.py"
        config_file.write_text("models = []\n")
        configs = ConfigLoader(config_file).load()
        assert configs == []

    def test_missing_file_raises(self, tmp_path: Path):
        """A missing config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader(tmp_path / "nonexistent.py").load()

    def test_no_config_path_returns_empty(self):
        """ConfigLoader with no path should return empty list."""
        configs = ConfigLoader(None).load()
        assert configs == []


class TestPythonConfigLoader:
    """Tests for ConfigLoader Python file parsing."""

    def test_load_python_models_list(self, tmp_path: Path):
        """Happy path: load a Python config with a list of ModelConfig."""
        config_file = tmp_path / "models.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="test-model",\n'
            '        task="image_mask_generation",\n'
            '        model_class="Sam2ImageModel",\n'
            '        model_params={"path": "facebook/sam2-hiera-base-plus"},\n'
            "    ),\n"
            "]\n"
        )
        configs = ConfigLoader(config_file).load()
        assert len(configs) == 1
        assert configs[0].name == "test-model"
        assert configs[0].task == "image_mask_generation"
        assert configs[0].model_class == "Sam2ImageModel"

    def test_load_python_missing_models_raises(self, tmp_path: Path):
        """Python config without a 'models' variable should raise ValueError."""
        config_file = tmp_path / "no_models.py"
        config_file.write_text("x = 42\n")
        with pytest.raises(ValueError, match="must define a 'models' variable"):
            ConfigLoader(config_file).load()

    def test_load_python_wrong_models_type_raises(self, tmp_path: Path):
        """Python config where 'models' is not a list should raise TypeError."""
        config_file = tmp_path / "wrong_type.py"
        config_file.write_text('models = "not a list"\n')
        with pytest.raises(TypeError, match="Expected 'models' to be a list"):
            ConfigLoader(config_file).load()

    def test_load_python_wrong_item_type_raises(self, tmp_path: Path):
        """Python config where 'models' contains non-ModelConfig items should raise TypeError."""
        config_file = tmp_path / "wrong_items.py"
        config_file.write_text('models = [{"name": "x"}]\n')
        with pytest.raises(TypeError, match="Expected ModelConfig instance"):
            ConfigLoader(config_file).load()

    def test_load_python_syntax_error_raises(self, tmp_path: Path):
        """Python config with syntax errors should raise ValueError."""
        config_file = tmp_path / "broken.py"
        config_file.write_text("models = [\n")
        with pytest.raises(ValueError, match="Syntax error"):
            ConfigLoader(config_file).load()

    def test_load_python_duplicate_names_raises(self, tmp_path: Path):
        """Python config with duplicate model names should raise ValueError."""
        config_file = tmp_path / "dupes.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="same-name",\n'
            '        task="image_mask_generation",\n'
            '        model_class="Sam2ImageModel",\n'
            '        model_params={"path": "facebook/sam2-hiera-base-plus"},\n'
            "    ),\n"
            "    ModelConfig(\n"
            '        name="same-name",\n'
            '        task="video_mask_generation",\n'
            '        model_class="Sam2VideoModel",\n'
            '        model_params={"path": "facebook/sam2-hiera-large"},\n'
            "    ),\n"
            "]\n"
        )
        with pytest.raises(ValueError, match="Duplicate model name"):
            ConfigLoader(config_file).load()

    def test_load_python_with_typed_enums_and_classes(self, tmp_path: Path):
        """Python config using Task enums and model class types loads correctly."""
        config_file = tmp_path / "typed_config.py"
        config_file.write_text(
            "from pixano_inference.configs.base import ModelConfig\n"
            "from pixano_inference.tasks import ImageTask, VideoTask\n"
            "\n"
            "class FakeImageModel:\n"
            "    pass\n"
            "\n"
            "class FakeVideoModel:\n"
            "    pass\n"
            "\n"
            "models = [\n"
            "    ModelConfig(\n"
            '        name="typed-image",\n'
            "        task=ImageTask.MASK_GENERATION,\n"
            "        model_class=FakeImageModel,\n"
            '        model_params={"path": "facebook/sam2-hiera-base-plus"},\n'
            "    ),\n"
            "    ModelConfig(\n"
            '        name="typed-video",\n'
            "        task=VideoTask.MASK_GENERATION,\n"
            "        model_class=FakeVideoModel,\n"
            '        model_params={"path": "facebook/sam2-hiera-large"},\n'
            "    ),\n"
            "]\n"
        )
        configs = ConfigLoader(config_file).load()
        assert len(configs) == 2
        assert configs[0].name == "typed-image"
        assert configs[0].task == "image_mask_generation"
        assert configs[0].model_class == "FakeImageModel"
        assert configs[1].name == "typed-video"
        assert configs[1].task == "video_mask_generation"
        assert configs[1].model_class == "FakeVideoModel"

    def test_unsupported_extension_raises(self, tmp_path: Path):
        """Unsupported file extension should raise ValueError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[models]\n")
        with pytest.raises(ValueError, match="Unsupported config file extension"):
            ConfigLoader(config_file).load()

    def test_yaml_extension_unsupported(self, tmp_path: Path):
        """YAML files are no longer supported and should raise ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("models: []\n")
        with pytest.raises(ValueError, match="Unsupported config file extension"):
            ConfigLoader(config_file).load()
