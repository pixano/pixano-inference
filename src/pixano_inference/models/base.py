# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel


if TYPE_CHECKING:
    from pixano_inference.ray.config import ModelDeploymentConfig


logger = logging.getLogger(__name__)


class InferenceModel(ABC):
    """Abstract base class for all inference models deployed on Ray Serve.

    Subclass this to implement custom inference models that can be deployed
    on Ray Serve.

    Example:
        ```python
        from pixano_inference.models import InferenceModel, register_model

        @register_model("my_model")
        class MyModel(InferenceModel):
            def load_model(self) -> None:
                self._model = ...  # Load your model

            def predict(self, input: MyInput) -> MyOutput:
                return MyOutput(result=self._model(input))
        ```
    """

    def __init__(self, config: ModelDeploymentConfig) -> None:
        """Initialize the model with deployment config.

        Args:
            config: Model deployment configuration.
        """
        self._config = config

    @property
    def config(self) -> ModelDeploymentConfig:
        """Model deployment configuration."""
        return self._config

    @property
    def model_name(self) -> str:
        """Unique model name."""
        return self._config.name

    @property
    def task(self) -> str:
        """Task string this model handles."""
        return self._config.task

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata. Override for custom metadata."""
        return {
            "model_name": self.model_name,
            "task": self.task,
            "model_class": self._config.model_class,
        }

    @abstractmethod
    def load_model(self) -> None:
        """Load model artifacts.

        Called once in the Ray actor ``__init__``. Implement this to load
        weights, initialize processors, etc.
        """

    @abstractmethod
    def predict(self, input: BaseModel) -> BaseModel:
        """Run inference.

        Args:
            input: Task-specific Input object (subclasses narrow this type).

        Returns:
            Task-specific Output object (subclasses narrow this type).
        """

    def unload(self) -> None:
        """Free resources. Override for custom cleanup."""
