# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Client for the Pixano Inference /v1 API.

Two clients share one contract: :class:`PixanoInferenceClient` (async) and
:class:`SyncPixanoInferenceClient` (sync). Both hold a pooled httpx transport, send the
optional API key, retry transient failures with backoff, and raise
:class:`PixanoInferenceError` carrying the server's ``{code, message, requestId}`` envelope.
Requests serialize as camelCase JSON (``by_alias=True``); media is passed by value as a URL,
base64 data-URI, or media-root path (the server also exposes ``/binary`` multipart routes for
callers that prefer raw uploads).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .schemas import (
    BaseResponse,
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
from .utils import is_url


DEFAULT_TIMEOUT = 60.0
TRACKING_TIMEOUT = 600.0
DEPLOY_TIMEOUT = 600.0
_RETRY_STATUS = frozenset({502, 503, 504})
_TERMINAL_JOB_STATES = frozenset({"completed", "failed", "canceled"})


class PixanoInferenceError(Exception):
    """Error raised by the client, carrying the server error envelope."""

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
        return response_type.model_validate(response.json())

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
        return JobStatus.model_validate(response.json())

    async def get_job(self, job_id: str) -> JobStatus:
        """Poll the status of a job."""
        response = await self._request("GET", f"/v1/jobs/{job_id}")
        return JobStatus.model_validate(response.json())

    async def cancel_job(self, job_id: str) -> JobStatus:
        """Cancel a job."""
        response = await self._request("DELETE", f"/v1/jobs/{job_id}")
        return JobStatus.model_validate(response.json())

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
        return [ModelStatusInfo.model_validate(model) for model in response.json()]

    async def deploy_model(self, request: DeployModelRequest, *, timeout: float | None = None) -> ModelStatusInfo:
        """Deploy a model at runtime."""
        response = await self._request(
            "POST", "/v1/models", json=self._json_body(request), timeout=timeout or DEPLOY_TIMEOUT
        )
        return ModelStatusInfo.model_validate(response.json())

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
        return response_type.model_validate(response.json())

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
        return JobStatus.model_validate(response.json())

    def get_job(self, job_id: str) -> JobStatus:
        """Poll the status of a job."""
        return JobStatus.model_validate(self._request("GET", f"/v1/jobs/{job_id}").json())

    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancel a job."""
        return JobStatus.model_validate(self._request("DELETE", f"/v1/jobs/{job_id}").json())

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
        return [ModelStatusInfo.model_validate(model) for model in response.json()]

    def deploy_model(self, request: DeployModelRequest, *, timeout: float | None = None) -> ModelStatusInfo:
        """Deploy a model at runtime."""
        response = self._request(
            "POST", "/v1/models", json=self._json_body(request), timeout=timeout or DEPLOY_TIMEOUT
        )
        return ModelStatusInfo.model_validate(response.json())

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
