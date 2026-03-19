# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Configuration loader for Ray Serve deployments."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

from .config import ModelDeploymentConfig


logger = logging.getLogger(__name__)

_PYTHON_EXTENSIONS = {".py"}


class ConfigLoader:
    """Loader for model deployment configurations.

    Python config files define a ``models`` list of ``ModelConfig``:

    .. code-block:: python

        from pixano_inference.configs import ModelConfig, Sam2ImageParams

        models = [
            ModelConfig(
                name="sam2-image",
                model_class="Sam2ImageModel",
                model_params=Sam2ImageParams(path="facebook/sam2-hiera-base-plus"),
            ),
        ]
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize the config loader.

        Args:
            config_path: Path to the configuration file (.py).
        """
        self._config_path = Path(config_path) if config_path is not None else None

    def load(self) -> list[ModelDeploymentConfig]:
        """Load model deployment configurations from the config file.

        Dispatches to the appropriate loader based on file extension.

        Returns:
            List of model deployment configurations.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the config is invalid or has an unsupported extension.
        """
        if self._config_path is None:
            return []

        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")

        suffix = self._config_path.suffix.lower()
        if suffix not in _PYTHON_EXTENSIONS:
            raise ValueError(
                f"Unsupported config file extension '{suffix}'. Only Python (.py) config files are supported."
            )
        return self._load_python()

    def _load_python(self) -> list[ModelDeploymentConfig]:
        """Load model deployment configurations from a Python file.

        The Python file must define a ``models`` variable that is a list of
        ``ModelConfig`` instances.

        Returns:
            List of model deployment configurations.

        Raises:
            ValueError: If the file has syntax errors, is missing ``models``,
                or contains duplicate model names.
            TypeError: If ``models`` is not a list or contains non-ModelConfig items.
        """
        from pixano_inference.configs.base import ModelConfig

        module = self._import_python_config(self._config_path)

        if not hasattr(module, "models"):
            raise ValueError(
                f"Python config '{self._config_path}' must define a 'models' variable. "
                f"Example: models = [ModelConfig(...)]"
            )

        models = module.models
        if not isinstance(models, list):
            raise TypeError(f"Expected 'models' to be a list, got {type(models).__name__} in '{self._config_path}'")

        configs: list[ModelDeploymentConfig] = []
        seen_names: set[str] = set()
        for i, item in enumerate(models):
            if not isinstance(item, ModelConfig):
                raise TypeError(
                    f"Expected ModelConfig instance at models[{i}], got {type(item).__name__} in '{self._config_path}'"
                )
            config = item.to_deployment_config()
            if config.name in seen_names:
                raise ValueError(
                    f"Duplicate model name '{config.name}'. "
                    f"Set an explicit 'name' on at least one of the conflicting models."
                )
            seen_names.add(config.name)
            configs.append(config)

        logger.info(f"Loaded {len(configs)} model configurations from {self._config_path}")
        return configs

    @staticmethod
    def _import_python_config(path: Path) -> ModuleType:
        """Import a Python file as a module.

        Args:
            path: Path to the Python file.

        Returns:
            The imported module.

        Raises:
            ValueError: If the file has syntax errors or cannot be loaded.
        """
        module_name = f"_pixano_config_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load Python config from '{path}'")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in Python config '{path}': {e}") from e
        finally:
            sys.modules.pop(module_name, None)

        return module
