# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""DeploymentManager and FastAPI app factory for Ray Serve."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import ray
from anyio import move_on_after, to_thread
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
_WATCHDOG_POLL_S = 0.25
_CLEANUP_TIMEOUT_S = 30.0
# Startup drains have no in-flight requests to protect, so they get a short leash.
_ABORT_DRAIN_TIMEOUT_S = 5.0
_DEFAULT_TIMEOUTS: dict[str, float] = {
    "segmentation": 60.0,
    "detection": 60.0,
    "vlm": 300.0,
    "tracking": 600.0,
    "ner": 60.0,
    "embedding": 60.0,
}


class StartupAborted(RuntimeError):
    """Raised when startup is cut short because shutdown was requested (Ctrl-C / SIGTERM)."""


def _ray_session_log_hint() -> str:
    """Best-effort path to the raylet log that explains why a Ray node died."""
    try:
        from ray._private import worker as _ray_worker

        node = _ray_worker._global_node
        if node is not None:
            return str(Path(node.get_session_dir_path()) / "logs" / "raylet.err")
    except Exception:  # pragma: no cover - defensive against Ray internals moving
        logger.debug("Could not resolve the Ray session directory", exc_info=True)
    return "/tmp/ray/session_latest/logs/raylet.err"


def _has_live_ray_node() -> bool:
    """Whether at least one raylet is still alive.

    A transient GCS error counts as "alive" so that only a definite node death aborts a startup;
    the timeout stays the backstop.
    """
    try:
        return any(node.get("Alive") for node in ray.nodes())
    except Exception:
        return True


def _run_with_node_watchdog(
    fn: Callable[[], Any],
    *,
    timeout_s: float,
    what: str,
    on_poll: Callable[[], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> Any:
    """Run a blocking Ray call, aborting if the Ray node dies or the timeout elapses.

    Ray's Serve calls wait on actors with no timeout of their own -- ``serve.start`` does a bare
    ``ray.get`` on the controller, and ``serve.run`` waits with ``timeout_s=-1`` whatever its
    ``blocking`` argument says -- so a raylet that dies mid-startup leaves them blocked forever.
    The call runs on a daemon thread: if it never returns we abandon it rather than let it block
    interpreter exit.

    Args:
        fn: The blocking call to run.
        timeout_s: Maximum time to wait before giving up.
        what: Human-readable description of the call, used in error messages.
        on_poll: Optional check run once per poll interval; raise from it to abort early.
        should_abort: Optional predicate polled alongside the call; when it returns True the
            wait is cut short with :class:`StartupAborted`. Used to honour Ctrl-C, which uvicorn
            turns into a flag rather than an exception.

    Returns:
        Whatever ``fn`` returned.

    Raises:
        StartupAborted: If ``should_abort`` reports that shutdown was requested.
        RuntimeError: If the Ray node dies while the call is in flight.
        TimeoutError: If the call does not finish within ``timeout_s``.
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def _target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_target, name="pixano-ray-watchdog", daemon=True).start()

    deadline = time.monotonic() + timeout_s
    seen_live_node = False
    while not done.wait(_WATCHDOG_POLL_S):
        if should_abort is not None and should_abort():
            raise StartupAborted(f"{what} aborted: shutdown was requested.")
        if _has_live_ray_node():
            seen_live_node = True
        elif seen_live_node:
            raise RuntimeError(
                f"{what} aborted: the Ray node died (the raylet exited), so this call could never "
                f"complete. See {_ray_session_log_hint()} for the reason."
            )
        if on_poll is not None:
            on_poll()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{what} did not complete within {timeout_s:g}s.")

    if "error" in box:
        raise box["error"]
    return box.get("value")


def _call_bounded(fn: Callable[[], Any], timeout_s: float, what: str) -> None:
    """Run a teardown call on a daemon thread, giving up after ``timeout_s``.

    Teardown talks to a cluster that may already be dead. anyio's worker threads are *not*
    daemonic, so abandoning a wedged ``to_thread.run_sync`` would keep the interpreter alive at
    exit -- this uses a thread we can genuinely walk away from. Errors are logged, never raised:
    the caller is already on its way out.
    """
    done = threading.Event()
    failure: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - logged, never propagated during teardown
            failure["exc"] = exc
        finally:
            done.set()

    threading.Thread(target=_target, name="pixano-teardown", daemon=True).start()
    if not done.wait(timeout_s):
        logger.warning("%s did not finish within %gs; abandoning it", what, timeout_s)
    elif "exc" in failure:
        logger.warning("%s error: %s", what, failure["exc"])


def _disable_ray_uv_run_hook() -> None:
    """Stop Ray from relaunching its workers through ``uv run``.

    Ray >= 2.53 detects a ``uv run`` parent process and rewrites the runtime env to
    ``py_executable="uv run"`` with ``working_dir`` set to the current directory -- but our
    runtime env excludes ``pyproject.toml``/``uv.lock``/``.venv`` from the uploaded copy (see
    ``.utils._DEFAULT_EXCLUDES``), so ``uv`` finds no project to resolve there and workers hang.
    With the hook off, workers inherit the already-active interpreter.
    """
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    try:
        from ray._private import ray_constants

        # The env var above is read into this constant at ``import ray`` time, which has already
        # happened by the time we get here, so the attribute is what actually takes effect.
        ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    except Exception:  # pragma: no cover - defensive against Ray internals moving
        logger.debug("Could not disable Ray's uv-run runtime env hook", exc_info=True)


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

    def __init__(self, config: RayServeConfig, should_abort: Callable[[], bool] | None = None) -> None:
        """Initialize the deployment manager.

        Args:
            config: Ray Serve configuration.
            should_abort: Optional predicate polled while waiting on Ray; when it returns True
                the wait is cut short (see :func:`_run_with_node_watchdog`).
        """
        self._config = config
        self._should_abort = should_abort
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
            StartupAborted: If shutdown was requested while waiting for the deployment.
            RuntimeError: If the Serve deployment fails to become healthy.
        """
        if config.name in self._configs:
            raise ValueError(f"Model '{config.name}' is already deployed.")

        model_class = ModelClassRegistry.get(config.model_class)
        self._preflight_resource_check(config)

        app = build_model_app(model_class, config)
        try:
            self._run_serve_app(app, config.name, timeout_s=self._deploy_timeout_s())
        except StartupAborted:
            # Shutting down: the caller drains Serve and Ray wholesale, so skip the per-app
            # cleanup rather than spend the teardown budget on it.
            raise
        except Exception as exc:
            self._best_effort_delete(config.name)
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

        self._best_effort_delete(name)

        self._configs.pop(name, None)
        self._handles.pop(name, None)
        self._metadata_cache.pop(name, None)
        self.jobs.cancel_for_model(name)
        logger.info("Undeployed model '%s'", name)

    def _deploy_timeout_s(self) -> float:
        """Deployment timeout from config, falling back to the module default."""
        return getattr(self._config, "deploy_timeout_s", None) or _DEPLOY_TIMEOUT_S

    def _best_effort_delete(self, name: str) -> None:
        """Delete a Serve app, bounded so a broken cluster cannot wedge teardown."""
        try:
            _run_with_node_watchdog(
                lambda: serve.delete(name),
                timeout_s=_CLEANUP_TIMEOUT_S,
                what=f"Deleting Serve app '{name}'",
            )
        except Exception as exc:
            logger.warning("Error deleting Serve app for '%s': %s", name, exc)

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
        """Run a Serve app and wait until it is RUNNING, failing fast on error/timeout.

        ``serve.run`` waits for the app internally with ``timeout_s=-1`` regardless of its
        ``blocking`` argument, so it is driven from a watchdog thread instead of called inline --
        otherwise neither ``timeout_s`` nor the status checks below would ever be reached.
        """

        def _fail_on_bad_status() -> None:
            status = self._app_status(name)
            if status in {"DEPLOY_FAILED", "UNHEALTHY"}:
                raise RuntimeError(f"Serve deployment '{name}' status is {status}.")

        try:
            _run_with_node_watchdog(
                lambda: serve.run(app, name=name, route_prefix=None),
                timeout_s=timeout_s,
                what=f"Serve deployment '{name}'",
                on_poll=_fail_on_bad_status,
                should_abort=self._should_abort,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"Serve deployment '{name}' did not become RUNNING within {timeout_s:g}s "
                f"(last status: {self._app_status(name)})."
            ) from exc

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
        # serve.status() auto-starts a local Ray cluster when none is connected; a status
        # query must never do that.
        apps: dict[str, Any] = {}
        if ray.is_initialized():
            try:
                apps = serve.status().applications
            except Exception:
                pass
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


def _start_ray_and_serve(config: RayServeConfig, should_abort: Callable[[], bool] | None = None) -> None:
    """Initialize Ray (if needed) and start Serve with its HTTP proxy disabled.

    Blocking: call it off the event loop (the lifespan uses ``to_thread.run_sync``).
    """
    if not ray.is_initialized():
        _disable_ray_uv_run_hook()
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
    # serve.start() blocks on the controller actor with no timeout of its own: if the Ray node
    # cannot schedule that actor it would wait forever, so bound it and report the raylet log.
    _run_with_node_watchdog(
        lambda: serve.start(proxy_location="Disabled"),
        timeout_s=config.serve_start_timeout_s,
        what="Ray Serve startup",
        should_abort=should_abort,
    )


async def _drain_ray_and_serve(config: RayServeConfig, timeout_s: float | None = None) -> None:
    """Drain Serve and Ray, bounded and shielded so a broken cluster cannot wedge exit.

    Both calls talk to a cluster that may already be dead; ``shield=True`` gives them a chance to
    run even when the lifespan is being cancelled, and the deadline guarantees we still exit.

    Args:
        config: Ray Serve configuration.
        timeout_s: Budget for each of the two teardown calls. Defaults to the configured
            graceful-shutdown window; startup failures pass a much shorter one, since no request
            has been served yet and there is nothing to drain gracefully.
    """
    timeout_s = config.graceful_shutdown_s if timeout_s is None else timeout_s

    def _drain() -> None:
        _call_bounded(serve.shutdown, timeout_s, "serve.shutdown")
        if ray.is_initialized():
            _call_bounded(ray.shutdown, timeout_s, "ray.shutdown")

    # _drain always returns within roughly 2 * timeout_s, so the worker thread is never
    # abandoned; the outer scope is a backstop, shielded so teardown still runs under cancellation.
    with move_on_after(2 * timeout_s + 5.0, shield=True):
        await to_thread.run_sync(_drain)


def create_ray_serve_app(
    config: RayServeConfig | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> tuple[FastAPI, DeploymentManager]:
    """Create the FastAPI application and DeploymentManager for Ray Serve.

    The returned app carries a lifespan that starts Ray + Serve and deploys startup models
    on startup, and drains Serve + Ray on shutdown. The lifespan runs when the app is served
    (or under ``with TestClient(app)``), not on bare construction.

    Args:
        config: Ray Serve configuration. If None, uses defaults.
        should_abort: Optional predicate polled during startup; when it returns True, startup is
            cut short. :class:`~pixano_inference.ray.server.InferenceServer` wires this to
            uvicorn's exit flags so Ctrl-C interrupts a long startup instead of being queued
            behind it.

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

    deployment_manager = DeploymentManager(config, should_abort=should_abort)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Ray/Serve startup and model loading block for a long time. Keep them off the event loop:
        # uvicorn replaces SIGINT with a handler that only sets a flag the loop polls, so blocking
        # the loop here makes Ctrl-C inert for the whole of startup. abandon_on_cancel lets us walk
        # away from a Ray call that never returns instead of hanging on the worker thread.
        def _aborted() -> bool:
            return should_abort is not None and should_abort()

        try:
            await to_thread.run_sync(_start_ray_and_serve, config, should_abort, abandon_on_cancel=True)
            failures: list[str] = []
            for model_config in config.models:
                if _aborted():
                    raise StartupAborted("Startup aborted: shutdown was requested.")
                try:
                    await to_thread.run_sync(deployment_manager.deploy_model, model_config, abandon_on_cancel=True)
                    logger.info("Startup model '%s' deployed", model_config.name)
                except StartupAborted:
                    raise
                except Exception as exc:
                    logger.error("Failed to deploy startup model '%s': %s", model_config.name, exc)
                    failures.append(model_config.name)
            if failures and config.strict_startup:
                raise RuntimeError(
                    f"Strict startup: {len(failures)} model(s) failed to deploy: {failures}. "
                    "Pass --no-strict-startup to start anyway."
                )
        except BaseException as exc:
            # Startup failed or was interrupted: drain whatever came up, so a half-started (or
            # already dead) cluster cannot hang the exit path through Ray's own atexit hooks.
            # Nothing has been served yet, so drain on a short leash rather than the full
            # graceful-shutdown window.
            if isinstance(exc, StartupAborted):
                logger.info("Startup interrupted before completion; shutting down")
            await _drain_ray_and_serve(config, timeout_s=_ABORT_DRAIN_TIMEOUT_S)
            raise

        try:
            yield
        finally:
            await _drain_ray_and_serve(config)

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

    # Request-id propagation (outermost) + Prometheus HTTP metrics.
    from pixano_inference.observability import install_observability_middleware

    install_observability_middleware(app)

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
