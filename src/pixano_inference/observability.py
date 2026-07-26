# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Observability: request-id propagation, structured logging, and Prometheus metrics.

Three concerns, all optional-at-runtime and cheap:

- **Request id**: :class:`RequestContextMiddleware` (a pure-ASGI middleware, so the id
  propagates via a ``ContextVar`` into endpoints and log records) tags every request with an
  ``X-Request-ID`` — echoed from the client's header or freshly generated — exposes it on
  ``request.state.request_id`` and the response header, and makes it available to logs.
- **Logging**: :func:`configure_logging` installs a ``dictConfig`` whose formatter includes the
  current request id (plain or JSON).
- **Metrics**: :class:`PrometheusMiddleware` records request count / latency / in-flight gauges;
  :func:`render_metrics` renders the exposition for a ``/metrics`` scrape endpoint. Ray Serve's
  own replica/queue metrics are exported separately by Serve.
"""

from __future__ import annotations

import json
import logging
import logging.config
import time
import uuid
from contextvars import ContextVar
from typing import Any, Awaitable, Callable


try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover - prometheus_client is a declared dependency
    _PROM_AVAILABLE = False


REQUEST_ID_HEADER = "X-Request-ID"

_request_id_ctx: ContextVar[str] = ContextVar("pixano_request_id", default="-")


def get_request_id() -> str:
    """Return the current request id, or ``"-"`` outside of a request."""
    return _request_id_ctx.get()


# --- Request id (pure-ASGI so the ContextVar reaches endpoints and log records) ----------


class RequestContextMiddleware:
    """Assign each HTTP request an id and expose it on state, logs, and the response header."""

    def __init__(self, app: Any) -> None:
        """Wrap the downstream ASGI *app*."""
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Set the request-id ContextVar and inject the ``X-Request-ID`` response header."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or {}).get(b"x-request-id")
        request_id = incoming.decode("latin-1") if incoming else uuid.uuid4().hex

        # Starlette's request.state reads from scope["state"], so this reaches route handlers.
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id
        token = _request_id_ctx.set(request_id)

        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((REQUEST_ID_HEADER.encode("latin-1"), request_id.encode("latin-1")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _request_id_ctx.reset(token)


# --- Metrics -----------------------------------------------------------------------------

if _PROM_AVAILABLE:
    # Module-level singletons: created once per process so building the app more than once
    # (as tests do) does not re-register and raise "Duplicated timeseries".
    _REQUESTS = Counter(
        "pixano_inference_requests_total",
        "Total HTTP requests handled by the ingress.",
        ["method", "path", "status"],
    )
    _LATENCY = Histogram(
        "pixano_inference_request_duration_seconds",
        "HTTP request latency in seconds.",
        ["method", "path"],
    )
    _IN_PROGRESS = Gauge(
        "pixano_inference_requests_in_progress",
        "In-flight HTTP requests.",
        ["method"],
    )


class PrometheusMiddleware:
    """Record request count, latency, and in-flight gauge per method and route template.

    The ``path`` label is the matched **route template** (e.g. ``/v1/inference/detection``), not
    the raw URL, so cardinality stays bounded; unmatched requests are labelled ``unmatched``.
    """

    def __init__(self, app: Any) -> None:
        """Wrap the downstream ASGI *app*."""
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Time the request and update the metrics, tolerating a missing prometheus_client."""
        if scope["type"] != "http" or not _PROM_AVAILABLE:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        status_holder = {"code": 500}

        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
            await send(message)

        _IN_PROGRESS.labels(method).inc()
        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            _IN_PROGRESS.labels(method).dec()
            route = scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            _REQUESTS.labels(method, path, str(status_holder["code"])).inc()
            _LATENCY.labels(method, path).observe(elapsed)


def render_metrics() -> tuple[bytes, str]:
    """Return the Prometheus exposition body and its content type for a ``/metrics`` route."""
    if not _PROM_AVAILABLE:  # pragma: no cover
        return b"", "text/plain; charset=utf-8"
    return generate_latest(), CONTENT_TYPE_LATEST


# --- Logging -----------------------------------------------------------------------------


class RequestIdFilter(logging.Filter):
    """Inject the current request id onto every log record as ``request_id``."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach ``record.request_id`` from the ContextVar."""
        record.request_id = _request_id_ctx.get()
        return True


class JsonLogFormatter(logging.Formatter):
    """Minimal structured JSON log formatter carrying the request id."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize the record to a single-line JSON object."""
        payload = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "requestId": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure root logging with a request-id-aware formatter.

    Args:
        level: Root log level name (e.g. ``"INFO"``).
        json_logs: Emit structured JSON logs instead of the plain text format.
    """
    plain_format = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
    formatter: dict[str, Any] = {"()": f"{__name__}.JsonLogFormatter"} if json_logs else {"format": plain_format}
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_id": {"()": f"{__name__}.RequestIdFilter"}},
            "formatters": {"default": formatter},
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["request_id"],
                }
            },
            "root": {"level": level.upper(), "handlers": ["default"]},
        }
    )


def install_observability_middleware(app: Any) -> None:
    """Add the metrics and request-id middleware to *app* (request-id outermost).

    ``add_middleware`` stacks last-added outermost, so adding metrics first then the request-id
    middleware makes the request id available to the metrics layer, the error handlers, and the
    response headers for every request.
    """
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(RequestContextMiddleware)


# Re-exported for callers that build their own metrics endpoint.
MetricsHandler = Callable[[], Awaitable[Any]]
