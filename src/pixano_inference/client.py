# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Client for the Pixano Inference /v1 API.

Two clients share one contract: :class:`PixanoInferenceClient` (async) and
:class:`SyncPixanoInferenceClient` (sync). Both hold a pooled httpx transport, send the
optional API key, retry transient failures with backoff, and raise
:class:`PixanoInferenceError` for every failure: the server's ``{code, message, requestId}``
envelope on an error status, ``connection_error`` when the server cannot be reached, and
``invalid_response`` when a successful response does not match its schema.
Requests serialize as camelCase JSON (``by_alias=True``); media is passed by value as a URL,
base64 data-URI, or media-root path (the server also exposes ``/binary`` multipart routes for
callers that prefer raw uploads).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from pydantic import TypeAdapter, ValidationError

from .schemas.base import BaseResponse
from .schemas.inference import (
    DetectionRequest,
    DetectionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    NERRequest,
    NERResponse,
    SegmentationRequest,
    SegmentationResponse,
    TrackingResponse,
    VLMRequest,
    VLMResponse,
)
from .schemas.v1 import DeployModelRequest, JobStatus, ModelStatusInfo, TrackingRequestV1
from .utils.url import is_url


DEFAULT_TIMEOUT = 60.0
TRACKING_TIMEOUT = 600.0
DEPLOY_TIMEOUT = 600.0
_RETRY_STATUS = frozenset({502, 503, 504})
_TERMINAL_JOB_STATES = frozenset({"completed", "failed", "canceled"})
# Set by the server on every response (echoed from the request or generated). The client only
# reads it; the name is repeated here rather than imported from the server stack.
_REQUEST_ID_HEADER = "X-Request-ID"
_VALIDATION_ERRORS_SHOWN = 5


class PixanoInferenceError(Exception):
    """Error raised by the client, carrying the server error envelope.

    Attributes:
        status_code: HTTP status of the response, ``0`` when there was none.
        code: The server's error code, or one of the client's own: ``connection_error``,
            ``timeout``, ``invalid_response`` (a successful response that does not match its
            schema, so the server and the client disagree on the contract).
        message: What went wrong.
        request_id: The server's request id, when a response carried one.
    """

    def __init__(self, status_code: int, code: str, message: Any, request_id: str | None = None) -> None:
        """Store the status code, error code, message, and optional request id."""
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        suffix = f" (requestId={request_id})" if request_id else ""
        super().__init__(f"[{status_code} {code}] {message}{suffix}")


class _ClientBase:
    """Shared configuration and request/response helpers for both clients."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        tracking_timeout: float = TRACKING_TIMEOUT,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Configure the client target, auth, timeouts, and retry policy."""
        url = url.rstrip("/")
        if not is_url(url):
            raise ValueError(f"Invalid URL, got '{url}'.")
        self.url = url
        self._api_key = api_key
        self._timeout = timeout
        self._tracking_timeout = tracking_timeout
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._extra_headers = dict(headers or {})

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(self._extra_headers)
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        if extra:
            headers.update(extra)
        return headers

    def _full_url(self, path: str) -> str:
        return f"{self.url}/{path.lstrip('/')}"

    @staticmethod
    def _json_body(request: Any) -> Any:
        return request.model_dump(mode="json", by_alias=True)

    def _backoff(self, attempt: int) -> float:
        return self._backoff_factor * (2**attempt)

    def _raise_for_error(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        code = "error"
        message: Any = response.reason_phrase
        request_id: str | None = None
        try:
            body = response.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                code = error.get("code", "error")
                message = error.get("message", message)
                request_id = error.get("requestId")
            else:
                message = body.get("detail") or body.get("message") or message
        raise PixanoInferenceError(response.status_code, code, message, request_id)

    @staticmethod
    def _parse(response: httpx.Response, response_type: Any) -> Any:
        """Validate the body of a successful response against its schema.

        Raises:
            PixanoInferenceError: With code ``invalid_response`` when the body is not JSON or does
                not match ``response_type``, so a caller handles one exception type whatever fails.
        """
        request_id = response.headers.get(_REQUEST_ID_HEADER)
        try:
            body = response.json()
        except ValueError as exc:
            message = f"The response body is not JSON: {exc}"
            raise PixanoInferenceError(response.status_code, "invalid_response", message, request_id) from exc
        try:
            return TypeAdapter(response_type).validate_python(body)
        except ValidationError as exc:
            message = _describe_validation_error(exc, response_type)
            raise PixanoInferenceError(response.status_code, "invalid_response", message, request_id) from exc


def _describe_validation_error(exc: ValidationError, response_type: Any) -> str:
    """One line naming the expected schema and the first fields that do not match it."""
    errors = exc.errors()
    shown = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<body>'}: {error['msg']}"
        for error in errors[:_VALIDATION_ERRORS_SHOWN]
    )
    hidden = len(errors) - _VALIDATION_ERRORS_SHOWN
    more = f" (+{hidden} more)" if hidden > 0 else ""
    name = getattr(response_type, "__name__", str(response_type))
    return f"The response does not match {name}: {shown}{more}"


class PixanoInferenceClient(_ClientBase):
    """Asynchronous client for the Pixano Inference /v1 API."""

    def __init__(self, url: str, *, transport: httpx.AsyncBaseTransport | None = None, **kwargs: Any) -> None:
        """Create the client and its pooled async transport.

        Args:
            url: Base server URL.
            transport: Optional httpx transport (e.g. an ASGI transport for in-process tests).
            **kwargs: Shared client options (api_key, timeout, retries, ...).
        """
        super().__init__(url, **kwargs)
        self._client = httpx.AsyncClient(timeout=self._timeout, transport=transport)

    @classmethod
    def connect(cls, url: str, *, api_key: str | None = None, **kwargs: Any) -> PixanoInferenceClient:
        """Construct a client for *url* (kept for backward compatibility)."""
        return cls(url, api_key=api_key, **kwargs)

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> PixanoInferenceClient:
        """Enter the async context manager."""
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Close the client on context exit."""
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        retry_statuses: frozenset[int] = _RETRY_STATUS,
        raise_on_error: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        url = self._full_url(path)
        headers = self._headers(kwargs.pop("headers", None))
        request_timeout = timeout or self._timeout
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, url, headers=headers, timeout=request_timeout, **kwargs)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                if attempt >= self._max_retries:
                    raise PixanoInferenceError(0, "connection_error", str(exc)) from exc
                await asyncio.sleep(self._backoff(attempt))
                attempt += 1
                continue
            if response.status_code in retry_statuses and attempt < self._max_retries:
                await asyncio.sleep(self._backoff(attempt))
                attempt += 1
                continue
            if raise_on_error:
                self._raise_for_error(response)
            return response

    async def _infer(self, path: str, request: Any, response_type: type[BaseResponse], timeout: float | None) -> Any:
        response = await self._request("POST", path, json=self._json_body(request), timeout=timeout)
        return self._parse(response, response_type)

    # --- Inference ------------------------------------------------------------------

    async def segmentation(
        self, request: SegmentationRequest, *, timeout: float | None = None
    ) -> SegmentationResponse:
        """Run image segmentation."""
        return await self._infer("/v1/inference/segmentation", request, SegmentationResponse, timeout)

    async def detection(self, request: DetectionRequest, *, timeout: float | None = None) -> DetectionResponse:
        """Run object detection."""
        return await self._infer("/v1/inference/detection", request, DetectionResponse, timeout)

    async def vlm(self, request: VLMRequest, *, timeout: float | None = None) -> VLMResponse:
        """Run vision-language generation."""
        return await self._infer("/v1/inference/vlm", request, VLMResponse, timeout)

    async def ner(self, request: NERRequest, *, timeout: float | None = None) -> NERResponse:
        """Run named entity recognition."""
        return await self._infer("/v1/inference/ner", request, NERResponse, timeout)

    async def embedding(self, request: EmbeddingRequest, *, timeout: float | None = None) -> EmbeddingResponse:
        """Compute image or text embeddings (CLIP-style shared space)."""
        return await self._infer("/v1/inference/embedding", request, EmbeddingResponse, timeout)

    async def tracking(self, request: TrackingRequestV1, *, timeout: float | None = None) -> TrackingResponse:
        """Run synchronous video tracking (short intervals)."""
        return await self._infer(
            "/v1/inference/tracking", request, TrackingResponse, timeout or self._tracking_timeout
        )

    # --- Async tracking jobs --------------------------------------------------------

    async def submit_tracking_job(self, request: TrackingRequestV1, *, timeout: float | None = None) -> JobStatus:
        """Submit a tracking request as an asynchronous job."""
        response = await self._request(
            "POST", "/v1/inference/tracking/jobs", json=self._json_body(request), timeout=timeout
        )
        return self._parse(response, JobStatus)

    async def get_job(self, job_id: str) -> JobStatus:
        """Poll the status of a job."""
        response = await self._request("GET", f"/v1/jobs/{job_id}")
        return self._parse(response, JobStatus)

    async def cancel_job(self, job_id: str) -> JobStatus:
        """Cancel a job."""
        response = await self._request("DELETE", f"/v1/jobs/{job_id}")
        return self._parse(response, JobStatus)

    async def wait_for_job(
        self, job_id: str, *, poll_interval: float = 1.0, timeout: float | None = None
    ) -> JobStatus:
        """Poll a job until it reaches a terminal state."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = await self.get_job(job_id)
            if job.status in _TERMINAL_JOB_STATES:
                return job
            if deadline is not None and time.monotonic() > deadline:
                raise PixanoInferenceError(0, "timeout", f"Job '{job_id}' did not finish within {timeout}s.")
            await asyncio.sleep(poll_interval)

    # --- Admin / service ------------------------------------------------------------

    async def list_models(self) -> list[ModelStatusInfo]:
        """List deployed models with their live status."""
        response = await self._request("GET", "/v1/models")
        return self._parse(response, list[ModelStatusInfo])

    async def deploy_model(self, request: DeployModelRequest, *, timeout: float | None = None) -> ModelStatusInfo:
        """Deploy a model at runtime."""
        response = await self._request(
            "POST", "/v1/models", json=self._json_body(request), timeout=timeout or DEPLOY_TIMEOUT
        )
        return self._parse(response, ModelStatusInfo)

    async def undeploy_model(self, name: str) -> dict[str, Any]:
        """Undeploy a model."""
        response = await self._request("DELETE", f"/v1/models/{name}")
        return response.json()

    async def info(self) -> dict[str, Any]:
        """Return server and cluster information."""
        return (await self._request("GET", "/v1/info")).json()

    async def ready(self) -> dict[str, Any]:
        """Return the readiness report (does not raise on 503)."""
        response = await self._request("GET", "/v1/ready", retry_statuses=frozenset(), raise_on_error=False)
        return response.json()

    async def health(self) -> dict[str, Any]:
        """Return the liveness report."""
        return (await self._request("GET", "/health")).json()


class SyncPixanoInferenceClient(_ClientBase):
    """Synchronous twin of :class:`PixanoInferenceClient` for callers not on an event loop."""

    def __init__(self, url: str, *, transport: httpx.BaseTransport | None = None, **kwargs: Any) -> None:
        """Create the client and its pooled sync transport."""
        super().__init__(url, **kwargs)
        self._client = httpx.Client(timeout=self._timeout, transport=transport)

    @classmethod
    def connect(cls, url: str, *, api_key: str | None = None, **kwargs: Any) -> SyncPixanoInferenceClient:
        """Construct a client for *url*."""
        return cls(url, api_key=api_key, **kwargs)

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def __enter__(self) -> SyncPixanoInferenceClient:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close the client on context exit."""
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        retry_statuses: frozenset[int] = _RETRY_STATUS,
        raise_on_error: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        url = self._full_url(path)
        headers = self._headers(kwargs.pop("headers", None))
        request_timeout = timeout or self._timeout
        attempt = 0
        while True:
            try:
                response = self._client.request(method, url, headers=headers, timeout=request_timeout, **kwargs)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                if attempt >= self._max_retries:
                    raise PixanoInferenceError(0, "connection_error", str(exc)) from exc
                time.sleep(self._backoff(attempt))
                attempt += 1
                continue
            if response.status_code in retry_statuses and attempt < self._max_retries:
                time.sleep(self._backoff(attempt))
                attempt += 1
                continue
            if raise_on_error:
                self._raise_for_error(response)
            return response

    def _infer(self, path: str, request: Any, response_type: type[BaseResponse], timeout: float | None) -> Any:
        response = self._request("POST", path, json=self._json_body(request), timeout=timeout)
        return self._parse(response, response_type)

    def segmentation(self, request: SegmentationRequest, *, timeout: float | None = None) -> SegmentationResponse:
        """Run image segmentation."""
        return self._infer("/v1/inference/segmentation", request, SegmentationResponse, timeout)

    def detection(self, request: DetectionRequest, *, timeout: float | None = None) -> DetectionResponse:
        """Run object detection."""
        return self._infer("/v1/inference/detection", request, DetectionResponse, timeout)

    def vlm(self, request: VLMRequest, *, timeout: float | None = None) -> VLMResponse:
        """Run vision-language generation."""
        return self._infer("/v1/inference/vlm", request, VLMResponse, timeout)

    def ner(self, request: NERRequest, *, timeout: float | None = None) -> NERResponse:
        """Run named entity recognition."""
        return self._infer("/v1/inference/ner", request, NERResponse, timeout)

    def embedding(self, request: EmbeddingRequest, *, timeout: float | None = None) -> EmbeddingResponse:
        """Compute image or text embeddings (CLIP-style shared space)."""
        return self._infer("/v1/inference/embedding", request, EmbeddingResponse, timeout)

    def tracking(self, request: TrackingRequestV1, *, timeout: float | None = None) -> TrackingResponse:
        """Run synchronous video tracking (short intervals)."""
        return self._infer("/v1/inference/tracking", request, TrackingResponse, timeout or self._tracking_timeout)

    def submit_tracking_job(self, request: TrackingRequestV1, *, timeout: float | None = None) -> JobStatus:
        """Submit a tracking request as an asynchronous job."""
        response = self._request("POST", "/v1/inference/tracking/jobs", json=self._json_body(request), timeout=timeout)
        return self._parse(response, JobStatus)

    def get_job(self, job_id: str) -> JobStatus:
        """Poll the status of a job."""
        return self._parse(self._request("GET", f"/v1/jobs/{job_id}"), JobStatus)

    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancel a job."""
        return self._parse(self._request("DELETE", f"/v1/jobs/{job_id}"), JobStatus)

    def wait_for_job(self, job_id: str, *, poll_interval: float = 1.0, timeout: float | None = None) -> JobStatus:
        """Poll a job until it reaches a terminal state."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = self.get_job(job_id)
            if job.status in _TERMINAL_JOB_STATES:
                return job
            if deadline is not None and time.monotonic() > deadline:
                raise PixanoInferenceError(0, "timeout", f"Job '{job_id}' did not finish within {timeout}s.")
            time.sleep(poll_interval)

    def list_models(self) -> list[ModelStatusInfo]:
        """List deployed models with their live status."""
        response = self._request("GET", "/v1/models")
        return self._parse(response, list[ModelStatusInfo])

    def deploy_model(self, request: DeployModelRequest, *, timeout: float | None = None) -> ModelStatusInfo:
        """Deploy a model at runtime."""
        response = self._request(
            "POST", "/v1/models", json=self._json_body(request), timeout=timeout or DEPLOY_TIMEOUT
        )
        return self._parse(response, ModelStatusInfo)

    def undeploy_model(self, name: str) -> dict[str, Any]:
        """Undeploy a model."""
        return self._request("DELETE", f"/v1/models/{name}").json()

    def info(self) -> dict[str, Any]:
        """Return server and cluster information."""
        return self._request("GET", "/v1/info").json()

    def ready(self) -> dict[str, Any]:
        """Return the readiness report (does not raise on 503)."""
        return self._request("GET", "/v1/ready", retry_statuses=frozenset(), raise_on_error=False).json()

    def health(self) -> dict[str, Any]:
        """Return the liveness report."""
        return self._request("GET", "/health").json()
