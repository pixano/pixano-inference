# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Ray actor wrapper for InferenceModel subclasses."""

from __future__ import annotations

import logging
import time
from typing import Any

import ray

from pixano_inference.models.base import InferenceModel

from .config import ModelDeploymentConfig


logger = logging.getLogger(__name__)


def create_model_deployment(
    model_class: type[InferenceModel],
    config: ModelDeploymentConfig,
) -> Any:
    """Wrap an InferenceModel subclass as a Ray remote actor.

    Creates a Ray actor class with:
    - ``predict(input_data)`` method forwarding to the model
    - ``get_metadata()`` method
    - ``get_stats()`` method (request count, avg time)
    - ``unload()`` method

    The actor's ``__init__`` instantiates the model class and calls
    ``load_model()``. Ray actor options (GPU/CPU/memory) come from
    ``config.resources``.

    Args:
        model_class: An InferenceModel subclass to deploy.
        config: Deployment configuration.

    Returns:
        A Ray remote actor handle (already created and running).

    """
    ray_actor_options: dict[str, Any] = {
        "num_gpus": config.resources.num_gpus,
        "num_cpus": config.resources.num_cpus,
    }
    if config.resources.memory_mb is not None:
        ray_actor_options["memory"] = config.resources.memory_mb * 1024 * 1024

    # Capture in closure
    _model_class = model_class
    _config = config

    @ray.remote(**ray_actor_options)
    class ModelActor:
        """Ray actor wrapping an InferenceModel."""

        def __init__(self) -> None:
            self._model = _model_class(_config)
            self._model.load_model()
            self._request_count = 0
            self._total_processing_time = 0.0
            logger.info(f"Actor '{_config.name}' initialized with {_model_class.__name__}")

        def predict(self, input_data: Any) -> Any:
            """Run inference.

            Args:
                input_data: Task-specific Input object.

            Returns:
                Task-specific Output object.
            """
            start_time = time.time()
            self._request_count += 1

            result = self._model.predict(input_data)

            processing_time = time.time() - start_time
            self._total_processing_time += processing_time
            return result

        def get_metadata(self) -> dict[str, Any]:
            """Get model metadata.

            Returns:
                Model metadata dictionary.
            """
            return self._model.metadata

        def get_stats(self) -> dict[str, Any]:
            """Get deployment statistics.

            Returns:
                Statistics dictionary.
            """
            return {
                "model_name": _config.name,
                "capability": _config.capability,
                "model_class": _config.model_class,
                "request_count": self._request_count,
                "total_processing_time": self._total_processing_time,
                "avg_processing_time": (
                    self._total_processing_time / self._request_count if self._request_count > 0 else 0
                ),
            }

        def unload(self) -> None:
            """Unload the model and free resources."""
            self._model.unload()
            logger.info(f"Actor '{_config.name}' unloaded")

        def ready(self) -> bool:
            """Check if the actor is ready (model loaded).

            Returns:
                True when ready.
            """
            return True

    # Create the actor and wait for it to be ready (model loaded)
    handle = ModelActor.remote()  # type: ignore[attr-defined]
    try:
        ray.get(handle.ready.remote(), timeout=300)
    except ray.exceptions.GetTimeoutError:
        ray.kill(handle)
        raise TimeoutError(
            f"Model '{config.name}' actor did not become ready within 300 s. "
            f"Requested resources: {ray_actor_options}. "
            f"Available cluster resources: {ray.available_resources()}. "
            f"Check that the required GPUs/CPUs are available."
        )
    return handle
