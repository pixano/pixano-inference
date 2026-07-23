# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""DeploymentManager and FastAPI app factory for Ray Serve."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import ray
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from ray import serve
from starlette.middleware.base import BaseHTTPMiddleware

from pixano_inference.__version__ import __version__
from pixano_inference.api.v1 import register_v1_api
from pixano_inference.api.v1.errors import register_exception_handlers
from pixano_inference.jobs import JobManager, JobRecord
from pixano_inference.models.registry import ModelClassRegistry
from pixano_inference.schemas import ModelInfo
from pixano_inference.security import make_api_key_dependency, warn_if_auth_disabled
from pixano_inference.server_settings import ServerSettings

from .config import ModelDeploymentConfig, RayServeConfig
from .deployment import build_model_app
from .utils import build_runtime_env


logger = logging.getLogger(__name__)

_DEPLOY_TIMEOUT_S = 600.0
_DEFAULT_TIMEOUTS: dict[str, float] = {
    "segmentation": 60.0,
    "detection": 60.0,
    "vlm": 300.0,
    "tracking": 600.0,
    "ner": 60.0,
}


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds a configured maximum, before reading it."""

    def __init__(self, app: Any, max_body_bytes: int) -> None:
        """Store the maximum allowed body size in bytes."""
        super().__init__(app)
        self._max = max_body_bytes

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        """Return 413 when the declared Content-Length exceeds the configured maximum."""
        if self._max > 0:
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > self._max:
                        return JSONResponse(
                            status_code=413,
                            content={"error": {"code": "payload_too_large", "message": "Request body too large."}},
                        )
                except ValueError:
                    pass
        return await call_next(request)


class DeploymentManager:
    """In-process manager for Ray Serve model deployments and async tracking jobs.

    Each model runs as its own Serve application (``serve.run(app, name=..., route_prefix=None)``).
    Handles are obtained lazily via ``serve.get_app_handle`` and inference is dispatched
    through the native async ``DeploymentResponse``. Deployment/config state is process-local;
    Serve owns replica supervision, autoscaling, and batching. Async jobs are delegated to a
    :class:`~pixano_inference.jobs.JobManager`.
    """

    def __init__(self, config: RayServeConfig) -> None:
        """Initialize the deployment manager.

        Args:
            config: Ray Serve configuration.
        """
        self._config = config
        self._handles: dict[str, Any] = {}  # model_name -> Serve DeploymentHandle (cache)
        self._configs: dict[str, ModelDeploymentConfig] = {}  # model_name -> config
        self._metadata_cache: dict[str, dict[str, Any]] = {}  # model_name -> metadata
        self.jobs = JobManager()

    @property
    def config(self) -> RayServeConfig:
        """Server configuration."""
        return self._config

    # --- Deployment lifecycle -------------------------------------------------------

    def deploy_model(self, config: ModelDeploymentConfig) -> None:
        """Deploy a model as its own Ray Serve application.

        Args:
            config: Model deployment configuration.

        Raises:
            ValueError: If the model is already deployed or resources are insufficient.
            KeyError: If the model class is not registered.
            RuntimeError: If the Serve deployment fails to become healthy.
        """
        if config.name in self._configs:
            raise ValueError(f"Model '{config.name}' is already deployed.")

        model_class = ModelClassRegistry.get(config.model_class)
        self._preflight_resource_check(config)

        app = build_model_app(model_class, config)
        try:
            self._run_serve_app(app, config.name, timeout_s=_DEPLOY_TIMEOUT_S)
        except Exception as exc:
            try:
                serve.delete(config.name)
            except Exception:
                pass
            raise RuntimeError(f"Failed to deploy model '{config.name}': {exc}") from exc

        self._configs[config.name] = config
        self._handles.pop(config.name, None)
        logger.info(
            "Deployed model '%s' (class=%s, capability=%s)", config.name, config.model_class, config.capability
        )

    def undeploy_model(self, name: str) -> None:
        """Undeploy a model: delete its Serve app (freeing GPU via replica cleanup).

        Args:
            name: Model name.

        Raises:
            ValueError: If the model is not deployed.
        """
        if name not in self._configs:
            raise ValueError(f"Model '{name}' is not deployed.")

        try:
            serve.delete(name)
        except Exception as exc:
            logger.warning("Error deleting Serve app for '%s': %s", name, exc)

        self._configs.pop(name, None)
        self._handles.pop(name, None)
        self._metadata_cache.pop(name, None)
        self.jobs.cancel_for_model(name)
        logger.info("Undeployed model '%s'", name)

    def _preflight_resource_check(self, config: ModelDeploymentConfig) -> None:
        """Fail fast when the cluster cannot satisfy the requested resources.

        Only checks the immediately-scheduled replicas (``min_replicas``); scale-to-zero
        deployments (``min_replicas == 0``) schedule nothing up front.
        """
        needed_replicas = config.autoscaling.min_replicas
        if needed_replicas <= 0 or not ray.is_initialized():
            return
        available = ray.available_resources()
        need_gpu = config.resources.num_gpus * needed_replicas
        need_cpu = config.resources.num_cpus * needed_replicas
        avail_gpu = float(available.get("GPU", 0.0))
        avail_cpu = float(available.get("CPU", 0.0))
        if need_gpu > avail_gpu + 1e-6:
            raise ValueError(
                f"Insufficient GPUs to deploy '{config.name}': need {need_gpu:g}, available {avail_gpu:g}."
            )
        if need_cpu > avail_cpu + 1e-6:
            raise ValueError(
                f"Insufficient CPUs to deploy '{config.name}': need {need_cpu:g}, available {avail_cpu:g}."
            )

    def _run_serve_app(self, app: Any, name: str, timeout_s: float) -> None:
        """Run a Serve app and wait until it is RUNNING, failing fast on error/timeout."""
        serve.run(app, name=name, route_prefix=None, blocking=False)
        start = time.time()
        while time.time() - start < timeout_s:
            status = self._app_status(name)
            if status == "RUNNING":
                return
            if status in {"DEPLOY_FAILED", "UNHEALTHY"}:
                raise RuntimeError(f"Serve deployment '{name}' status is {status}.")
            time.sleep(0.5)
        raise TimeoutError(f"Serve deployment '{name}' did not become RUNNING within {timeout_s:g}s.")

    def _app_status(self, name: str) -> str:
        """Return the Serve application status string for *name* (or NOT_STARTED)."""
        try:
            app = serve.status().applications.get(name)
        except Exception:
            return "NOT_STARTED"
        if app is None:
            return "NOT_STARTED"
        status = app.status
        return status.value if hasattr(status, "value") else str(status)

    # --- Handles & metadata ---------------------------------------------------------

    def get_handle(self, name: str) -> Any | None:
        """Get a Serve deployment handle by model name (cached), or None if not deployed."""
        if name in self._handles:
            return self._handles[name]
        if name not in self._configs:
            return None
        handle = serve.get_app_handle(name)
        self._handles[name] = handle
        return handle

    def get_model_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a deployed model."""
        if name in self._metadata_cache:
            return self._metadata_cache[name]
        config = self._configs.get(name)
        if config is None:
            return {}
        metadata = {
            "model_name": config.name,
            "capability": config.capability,
            "model_class": config.model_class,
        }
        self._metadata_cache[name] = metadata
        return metadata

    def list_models(self) -> list[ModelInfo]:
        """List all deployed models."""
        return [
            ModelInfo(
                name=config.name,
                capability=config.capability,
                model_path=config.model_params.get("path")
                if isinstance(config.model_params.get("path"), str)
                else None,
                model_class=config.model_class,
            )
            for config in self._configs.values()
        ]

    def get_model_capability(self, name: str) -> str | None:
        """Get the deployed capability for a model."""
        config = self._configs.get(name)
        return config.capability if config is not None else None

    def get_timeout(self, name: str, capability: str) -> float:
        """Resolve the inference timeout for a model, honoring a per-model override."""
        config = self._configs.get(name)
        if config is not None and config.timeout_s is not None:
            return config.timeout_s
        return _DEFAULT_TIMEOUTS.get(capability, 120.0)

    # --- Readiness ------------------------------------------------------------------

    def model_statuses(self) -> dict[str, str]:
        """Return {model_name: Serve status string} for all configured models."""
        try:
            apps = serve.status().applications
        except Exception:
            apps = {}
        result: dict[str, str] = {}
        for name in self._configs:
            app = apps.get(name)
            if app is None:
                result[name] = "NOT_STARTED"
            else:
                status = app.status
                result[name] = status.value if hasattr(status, "value") else str(status)
        return result

    def readiness(self) -> dict[str, Any]:
        """Report readiness: every configured model must be RUNNING."""
        statuses = self.model_statuses()
        running = sum(1 for s in statuses.values() if s == "RUNNING")
        ready = all(s == "RUNNING" for s in statuses.values())
        return {
            "ready": ready,
            "models": statuses,
            "models_loaded": running,
            "version": __version__,
        }

    # --- Async jobs (delegated to JobManager) ---------------------------------------

    def submit_tracking_job(self, model_name: str, input_data: Any) -> str:
        """Submit a tracking request as an asynchronous job over the Serve handle."""
        handle = self.get_handle(model_name)
        if handle is None:
            raise ValueError(f"Model '{model_name}' is not deployed.")
        response = handle.predict.remote(input_data)
        return self.jobs.submit(response, model_name=model_name, metadata=self.get_model_metadata(model_name))

    def get_tracking_job(self, job_id: str) -> JobRecord | None:
        """Return the current state of a tracking job."""
        return self.jobs.get(job_id)

    def cancel_tracking_job(self, job_id: str) -> JobRecord | None:
        """Cancel a tracking job on a best-effort basis."""
        return self.jobs.cancel(job_id)

    # --- Cluster info ---------------------------------------------------------------

    def get_gpu_info(self) -> dict[str, Any]:
        """Get GPU resource information from Ray."""
        if not ray.is_initialized():
            return {"num_gpus": 0, "available_gpus": 0.0, "gpus_used": 0.0}
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        total_gpus = float(cluster_resources.get("GPU", 0.0))
        available_gpus = float(available_resources.get("GPU", 0.0))
        used_gpus = max(0.0, total_gpus - available_gpus)
        return {
            "num_gpus": int(total_gpus),
            "available_gpus": available_gpus,
            "gpus_used": used_gpus,
        }

    def num_nodes(self) -> int:
        """Number of alive Ray nodes, or 1 when Ray is not initialized."""
        if not ray.is_initialized():
            return 1
        try:
            return sum(1 for node in ray.nodes() if node.get("Alive"))
        except Exception:
            return 1


def _start_ray_and_serve(config: RayServeConfig) -> None:
    """Initialize Ray (if needed) and start Serve with its HTTP proxy disabled."""
    if not ray.is_initialized():
        init_kwargs: dict[str, Any] = {"namespace": config.ray_namespace}
        runtime_env = build_runtime_env(
            pip_packages=config.pip_packages,
            working_dir=config.working_dir,
            auto_detect=False,
        )
        if runtime_env:
            init_kwargs["runtime_env"] = runtime_env
        if config.num_cpus is not None:
            init_kwargs["num_cpus"] = config.num_cpus
        if config.num_gpus is not None:
            init_kwargs["num_gpus"] = config.num_gpus
        if config.ray_address is not None:
            init_kwargs["address"] = config.ray_address
        ray.init(**init_kwargs)
        logger.info("Ray initialized (namespace=%s)", config.ray_namespace)
    # We serve the API from our own uvicorn ingress, so disable Serve's HTTP proxy.
    serve.start(proxy_location="Disabled")


def create_ray_serve_app(
    config: RayServeConfig | None = None,
) -> tuple[FastAPI, DeploymentManager]:
    """Create the FastAPI application and DeploymentManager for Ray Serve.

    The returned app carries a lifespan that starts Ray + Serve and deploys startup models
    on startup, and drains Serve + Ray on shutdown. The lifespan runs when the app is served
    (or under ``with TestClient(app)``), not on bare construction.

    Args:
        config: Ray Serve configuration. If None, uses defaults.

    Returns:
        Tuple of (FastAPI app, DeploymentManager).
    """
    if config is None:
        config = RayServeConfig()

    # Register built-in backends and installed entry-point plugins.
    from pixano_inference.plugins import ensure_models_loaded

    ensure_models_loaded()

    server_settings = ServerSettings()
    warn_if_auth_disabled(server_settings, config.host)

    deployment_manager = DeploymentManager(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _start_ray_and_serve(config)
        failures: list[str] = []
        for model_config in config.models:
            try:
                deployment_manager.deploy_model(model_config)
                logger.info("Startup model '%s' deployed", model_config.name)
            except Exception as exc:
                logger.error("Failed to deploy startup model '%s': %s", model_config.name, exc)
                failures.append(model_config.name)
        if failures and config.strict_startup:
            raise RuntimeError(
                f"Strict startup: {len(failures)} model(s) failed to deploy: {failures}. "
                "Pass --no-strict-startup to start anyway."
            )
        try:
            yield
        finally:
            try:
                serve.shutdown()
            except Exception as exc:
                logger.warning("serve.shutdown error: %s", exc)
            try:
                if ray.is_initialized():
                    ray.shutdown()
            except Exception as exc:
                logger.warning("ray.shutdown error: %s", exc)

    app = FastAPI(
        title="Pixano Inference (Ray)",
        description="Pixano Inference API powered by Ray Serve",
        version=__version__,
        lifespan=lifespan,
    )

    # Body-size limit (before any body is read).
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=server_settings.max_request_body_bytes)

    # CORS, only when explicitly configured.
    if server_settings.cors_allow_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=server_settings.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Consistent {"error": {...}} envelope; unhandled errors never leak their text.
    register_exception_handlers(app)

    auth_dependency = make_api_key_dependency(server_settings)

    # Mount the versioned API (/v1) plus the unversioned /health alias.
    register_v1_api(app, deployment_manager, auth_dependency=auth_dependency)

    # Store references in app state for access by routes
    app.state.config = config
    app.state.server_settings = server_settings
    app.state.deployment_manager = deployment_manager

    return app, deployment_manager
