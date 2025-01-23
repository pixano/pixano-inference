# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Registry for inference models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pixano_inference.tasks import Task, str_to_task


if TYPE_CHECKING:
    from .models.base import BaseInferenceModel
    from .providers.base import BaseProvider


REGISTERED_MODELS: dict[tuple[str, str], tuple[BaseInferenceModel, BaseProvider, Task]] = {}


def register_model(model: BaseInferenceModel, provider: BaseProvider, task: Task | str):
    """Register a model in the registry."""
    if (model.name, model.provider) in REGISTERED_MODELS:
        raise ValueError(f"Model {model.name} from provider {model.provider} is already registered.")
    task = task if isinstance(task, Task) else str_to_task(task)
    REGISTERED_MODELS[(model.name, model.provider)] = (model, provider, task)


def unregister_model(model: BaseInferenceModel):
    """Unregister a model from the registry."""
    if (model.name, model.provider) not in REGISTERED_MODELS:
        raise ValueError(f"Model {model.name} from provider {model.provider} is not registered.")
    del REGISTERED_MODELS[(model.name, model.provider)]


def get_model_from_registry(name: str, provider: str) -> BaseInferenceModel:
    """Get a model from the registry."""
    if (name, provider) not in REGISTERED_MODELS:
        raise KeyError(f"Model {name} from provider {provider} is not registered.")
    return REGISTERED_MODELS[(name, provider)][0]


def get_model_provider_from_registry(name: str, provider: str) -> BaseProvider:
    """Get the provider of a model from the registry."""
    if (name, provider) not in REGISTERED_MODELS:
        raise KeyError(f"Model {name} from provider {provider} is not registered.")
    return REGISTERED_MODELS[(name, provider)][1]


def get_model_task_from_registry(name: str, provider: str) -> Task:
    """Get the task of a model from the registry."""
    if (name, provider) not in REGISTERED_MODELS:
        raise KeyError(f"Model {name} from provider {provider} is not registered.")
    return REGISTERED_MODELS[(name, provider)][2]


def get_values_from_model_registry(name: str, provider: str) -> tuple[BaseInferenceModel, BaseProvider, Task]:
    """Get the values of a model from the registry."""
    if (name, provider) not in REGISTERED_MODELS:
        raise KeyError(f"Model {name} from provider {provider} is not registered.")
    return REGISTERED_MODELS[(name, provider)]


def list_models() -> list[tuple[tuple[str, str], tuple[BaseInferenceModel, BaseProvider, Task]]]:
    """List all models in the registry."""
    return list(REGISTERED_MODELS.items())
