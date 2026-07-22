# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Ray Serve deployment for InferenceModel subclasses.

Each model is wrapped in a :class:`ModelReplica` Serve deployment and bound into its own
Serve application. Serve supervises the replicas (restart on crash), applies autoscaling
and per-replica concurrency limits, and optionally batches requests. Model access inside a
replica is serialised on a single worker thread so a stateful model is never entered
concurrently, while Serve queues/scales across replicas.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ray import serve
from ray.serve.config import AutoscalingConfig as ServeAutoscalingConfig

from pixano_inference.models.base import InferenceModel

from .config import ModelDeploymentConfig


logger = logging.getLogger(__name__)


class ModelReplica:
    """Serve deployment wrapping a single :class:`InferenceModel` instance.

    The model class travels with the deployment (bound by value), so no driver-side
    registry state is needed in the replica. ``load_model`` runs in ``__init__``; a failure
    there fails the replica and, in turn, the deployment (surfaced by ``serve.run``).
    """

    def __init__(self, model_class: type[InferenceModel], config: ModelDeploymentConfig) -> None:
        """Instantiate and load the model, and install the media security policy."""
        # Install the media-ingestion security policy in this worker from the environment
        # the driver passed on (SSRF / path-traversal guard for URL/path resolution).
        from pixano_inference.utils.media_security import MediaPolicy, set_media_policy

        set_media_policy(MediaPolicy.from_env())

        self._config = config
        self._model = model_class(config)
        self._model.load_model()
        self._request_count = 0
        self._total_processing_time = 0.0
        self._max_batch_size = config.max_batch_size
        # Serialise model access: a stateful model must never be entered concurrently.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"predict-{config.name}")

        if self._max_batch_size > 1:
            # @serve.batch adds these setters to the bound method at runtime.
            self._batched_predict.set_max_batch_size(self._max_batch_size)  # type: ignore[attr-defined]
            self._batched_predict.set_batch_wait_timeout_s(config.batch_wait_timeout_s)  # type: ignore[attr-defined]

        logger.info("Replica for '%s' initialized with %s", config.name, model_class.__name__)

    async def predict(self, input_data: Any) -> Any:
        """Run inference for a single request (batched transparently when enabled)."""
        import asyncio

        self._request_count += 1
        start = time.time()
        try:
            if self._max_batch_size > 1:
                return await self._batched_predict(input_data)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._model.predict, input_data)
        finally:
            self._total_processing_time += time.time() - start

    @serve.batch(max_batch_size=1)
    async def _batched_predict(self, inputs: list[Any]) -> list[Any]:
        """Batched inference path (reconfigured in ``__init__`` when batching is enabled)."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._model.predict_batch, inputs)

    async def get_metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return self._model.metadata

    async def get_stats(self) -> dict[str, Any]:
        """Return request/timing statistics for this replica."""
        return {
            "model_name": self._config.name,
            "capability": self._config.capability,
            "model_class": self._config.model_class,
            "request_count": self._request_count,
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": (
                self._total_processing_time / self._request_count if self._request_count > 0 else 0.0
            ),
        }

    def check_health(self) -> None:
        """Serve health probe: raise if the model is not loaded."""
        if self._model is None:
            raise RuntimeError(f"Model '{self._config.name}' is not loaded.")

    def __del__(self) -> None:
        """Free model resources (GPU memory) when the replica is torn down."""
        try:
            executor = getattr(self, "_executor", None)
            if executor is not None:
                executor.shutdown(wait=False)
            model = getattr(self, "_model", None)
            if model is not None:
                model.unload()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.warning("Error unloading replica '%s': %s", getattr(self._config, "name", "?"), exc)


def build_model_app(model_class: type[InferenceModel], config: ModelDeploymentConfig) -> Any:
    """Build a Serve application binding :class:`ModelReplica` for *config*.

    Fixed replica count when ``min_replicas == max_replicas``; otherwise an autoscaling
    deployment (including scale-to-zero when ``min_replicas == 0``).

    Args:
        model_class: The resolved :class:`InferenceModel` subclass to serve.
        config: The deployment configuration.

    Returns:
        A bound Serve application (pass to ``serve.run``).
    """
    ray_actor_options: dict[str, Any] = {
        "num_gpus": config.resources.num_gpus,
        "num_cpus": config.resources.num_cpus,
    }
    if config.resources.memory_mb is not None:
        ray_actor_options["memory"] = config.resources.memory_mb * 1024 * 1024

    options: dict[str, Any] = {
        "name": config.name,
        "ray_actor_options": ray_actor_options,
        "max_ongoing_requests": config.max_ongoing_requests,
        "health_check_period_s": config.health_check_period_s,
    }

    autoscaling = config.autoscaling
    if autoscaling.min_replicas == autoscaling.max_replicas:
        options["num_replicas"] = autoscaling.min_replicas
    else:
        options["autoscaling_config"] = ServeAutoscalingConfig(
            min_replicas=autoscaling.min_replicas,
            max_replicas=autoscaling.max_replicas,
            target_ongoing_requests=autoscaling.target_num_ongoing_requests_per_replica,
            upscale_delay_s=autoscaling.upscale_delay_s,
            downscale_delay_s=autoscaling.downscale_delay_s,
        )

    deployment = serve.deployment(ModelReplica).options(**options)  # type: ignore[attr-defined]
    return deployment.bind(model_class, config)
