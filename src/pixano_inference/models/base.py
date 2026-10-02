# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Shared base class for deployed inference models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel


if TYPE_CHECKING:
    from pixano_inference.configs.deployment import ModelDeploymentConfig


logger = logging.getLogger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class InferenceModel(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all inference models deployed on Ray Serve.

    The class is generic in the type ``predict`` receives and the type it returns. Each capability
    base class (``SegmentationModel``, ``DetectionModel``, ...) fixes both, so a model subclasses
    the capability base and implements ``predict`` with that capability's ``Input`` and ``Output``.

    Example:
        ```python
        from pixano_inference.models import InferenceModel, register_model

        @register_model("my_model")
        class MyModel(InferenceModel[MyInput, MyOutput]):
            def load_model(self) -> None:
                self._model = ...  # Load your model

            def predict(self, input: MyInput) -> MyOutput:
                return MyOutput(result=self._model(input))
        ```
    """

    capability_name: ClassVar[str | None] = None

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
    def capability(self) -> str:
        """Capability handled by this model."""
        return self._config.capability

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata. Override for custom metadata."""
        return {
            "model_name": self.model_name,
            "capability": self.capability,
            "model_class": self._config.model_class,
        }

    @abstractmethod
    def load_model(self) -> None:
        """Load model artifacts.

        Called once in the Ray actor ``__init__``. Implement this to load
        weights, initialize processors, etc.
        """

    @abstractmethod
    def predict(self, input: InputT) -> OutputT:
        """Run inference.

        Args:
            input: The capability's Input object.

        Returns:
            The capability's Output object.
        """

    def predict_batch(self, inputs: list[InputT]) -> list[OutputT]:
        """Run inference on a batch of inputs.

        Only used when the deployment sets ``max_batch_size > 1``. The default runs
        :meth:`predict` sequentially; override it to exploit true batched execution
        (e.g. a single padded forward pass).

        Args:
            inputs: A list of task-specific Input objects.

        Returns:
            A list of task-specific Output objects, one per input, in order.
        """
        return [self.predict(inp) for inp in inputs]

    def unload(self) -> None:
        """Free resources. Override for custom cleanup."""
