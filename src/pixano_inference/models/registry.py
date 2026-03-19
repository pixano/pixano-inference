# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Model class registry for inference deployments."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .base import InferenceModel


class ModelClassRegistry:
    """Registry mapping model class names to their implementations.

    This registry allows users to register custom model classes via the
    ``@register_model`` decorator and look them up by name at deployment time.
    """

    _registry: dict[str, type[InferenceModel]] = {}

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[type[InferenceModel]], type[InferenceModel]]:
        """Decorator to register a model class.

        Args:
            name: Name to register the class under. If None, uses the class name.

        Returns:
            The decorator function.
        """

        def decorator(model_cls: type[InferenceModel]) -> type[InferenceModel]:
            reg_name = name or model_cls.__name__
            if reg_name in cls._registry:
                raise ValueError(f"Model class '{reg_name}' already registered.")
            cls._registry[reg_name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[InferenceModel]:
        """Get a registered model class by name.

        Args:
            name: Registered name of the model class.

        Returns:
            The model class.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in cls._registry:
            raise KeyError(f"Model class '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_all(cls) -> dict[str, type[InferenceModel]]:
        """List all registered model classes.

        Returns:
            Dictionary of name to model class.
        """
        return cls._registry.copy()

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a model class is registered.

        Args:
            name: Name to check.

        Returns:
            True if registered.
        """
        return name in cls._registry


def register_model(name: str | None = None) -> Callable[[type[InferenceModel]], type[InferenceModel]]:
    """Convenience decorator for registering model classes.

    Args:
        name: Name to register under. If None, uses the class name.

    Returns:
        The decorator function.
    """
    return ModelClassRegistry.register(name)
