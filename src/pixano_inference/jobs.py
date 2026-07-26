# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""In-process async job manager for long-running inference (e.g. video tracking).

A job wraps a Ray Serve ``DeploymentResponse`` in an asyncio task, so status can be polled
without ever blocking the event loop on ``ray.get``. The store is bounded: terminal jobs
are evicted past a TTL and when a size cap is exceeded. State is process-local and lost on
restart (the computation lives in Serve replicas the server manages), which is appropriate
for a single-node deployment.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


logger = logging.getLogger(__name__)

TERMINAL_STATES = {"completed", "failed", "canceled"}
DEFAULT_MAX_JOBS = 500
DEFAULT_TTL_S = 3600.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    """State for a single asynchronous job."""

    model_name: str
    response: Any = None
    task: Any = None
    status: str = "running"
    detail: str | None = None
    result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)
    submitted_at_monotonic: float = field(default_factory=time.time)
    processing_time: float = 0.0


class JobManager:
    """Bounded, in-process manager of asynchronous jobs over Serve responses."""

    def __init__(self, max_jobs: int = DEFAULT_MAX_JOBS, ttl_s: float = DEFAULT_TTL_S) -> None:
        """Initialize the job manager.

        Args:
            max_jobs: Maximum number of retained jobs before terminal jobs are evicted.
            ttl_s: Time-to-live for terminal jobs before eviction.
        """
        self._jobs: dict[str, JobRecord] = {}
        self._max_jobs = max_jobs
        self._ttl_s = ttl_s

    def submit(self, response: Any, *, model_name: str, metadata: dict[str, Any] | None = None) -> str:
        """Register a job for a Serve response and start awaiting it.

        Must be called from within a running event loop (i.e. an async route handler).

        Args:
            response: A Serve ``DeploymentResponse`` (awaitable, cancellable).
            model_name: Name of the model handling the job.
            metadata: Optional metadata echoed back in job status.

        Returns:
            The generated job id.
        """
        job_id = f"job-{uuid4().hex}"
        record = JobRecord(model_name=model_name, response=response, metadata=metadata or {})
        self._jobs[job_id] = record
        record.task = asyncio.get_running_loop().create_task(self._await(job_id, response))
        self._evict()
        return job_id

    async def _await(self, job_id: str, response: Any) -> None:
        try:
            result = await response
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("Job %s failed: %s", job_id, exc)
            self._finalize(job_id, status="failed", detail=str(exc))
            return
        payload = result.model_dump(by_alias=True) if hasattr(result, "model_dump") else result
        self._finalize(job_id, status="completed", result=payload)

    def _finalize(
        self,
        job_id: str,
        *,
        status: str,
        detail: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> JobRecord | None:
        job = self._jobs.get(job_id)
        if job is None or job.status in TERMINAL_STATES:
            return job
        job.status = status
        job.detail = detail
        job.result = result
        job.timestamp = _utcnow()
        job.processing_time = max(0.0, time.time() - job.submitted_at_monotonic)
        return job

    def get(self, job_id: str) -> JobRecord | None:
        """Return the current state of a job (kept up to date by its task)."""
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> JobRecord | None:
        """Cancel a job on a best-effort basis."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status in TERMINAL_STATES:
            return job
        try:
            if job.response is not None:
                job.response.cancel()
        except Exception as exc:
            logger.warning("Failed to cancel job %s response: %s", job_id, exc)
        if job.task is not None:
            job.task.cancel()
        return self._finalize(job_id, status="canceled", detail="Job canceled.")

    def cancel_for_model(self, model_name: str, detail: str = "Model undeployed.") -> None:
        """Cancel all non-terminal jobs belonging to a model."""
        for job_id, job in list(self._jobs.items()):
            if job.model_name == model_name and job.status not in TERMINAL_STATES:
                self.cancel(job_id)
                self._finalize(job_id, status="canceled", detail=detail)

    def _evict(self) -> None:
        now = time.time()
        for job_id, job in list(self._jobs.items()):
            if job.status in TERMINAL_STATES and (now - job.submitted_at_monotonic) > self._ttl_s:
                del self._jobs[job_id]
        if len(self._jobs) <= self._max_jobs:
            return
        terminal = [(jid, job) for jid, job in self._jobs.items() if job.status in TERMINAL_STATES]
        terminal.sort(key=lambda item: item[1].submitted_at_monotonic)
        for job_id, _ in terminal[: len(self._jobs) - self._max_jobs]:
            del self._jobs[job_id]

    def evict_now(self) -> None:
        """Run one eviction pass (for a periodic background sweep)."""
        self._evict()

    def cancel_all(self) -> None:
        """Cancel every non-terminal job (used on shutdown)."""
        for job_id in list(self._jobs):
            self.cancel(job_id)

    @property
    def count(self) -> int:
        """Number of retained jobs."""
        return len(self._jobs)


def serialize_job(job_id: str, job: JobRecord) -> dict[str, Any]:
    """Serialize a job record into the camelCase API status shape."""
    return {
        "jobId": job_id,
        "status": job.status,
        "detail": job.detail,
        "data": job.result if job.status == "completed" else None,
        "metadata": job.metadata,
        "timestamp": job.timestamp,
        "processingTime": job.processing_time,
    }
