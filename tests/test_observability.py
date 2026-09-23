# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for request-id propagation, Prometheus metrics, and logging config."""

import logging

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from pixano_inference.observability import (
    REQUEST_ID_HEADER,
    RequestIdFilter,
    configure_logging,
    get_request_id,
    install_observability_middleware,
    render_metrics,
)


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()

    @application.get("/ok")
    async def ok(request: Request) -> dict:
        # The id set by the middleware must reach the endpoint via scope state and the ContextVar.
        return {"stateId": request.state.request_id, "ctxId": get_request_id()}

    @application.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)

    install_observability_middleware(application)
    return application


def test_request_id_generated_and_echoed_on_response(app):
    client = TestClient(app)
    resp = client.get("/ok")
    assert resp.status_code == 200
    rid = resp.headers[REQUEST_ID_HEADER]
    assert len(rid) == 32  # generated uuid4 hex
    # Same id is visible on request.state and via the ContextVar inside the endpoint.
    assert resp.json()["stateId"] == rid
    assert resp.json()["ctxId"] == rid


def test_request_id_taken_from_incoming_header(app):
    client = TestClient(app)
    resp = client.get("/ok", headers={"X-Request-ID": "trace-123"})
    assert resp.headers[REQUEST_ID_HEADER] == "trace-123"
    assert resp.json()["stateId"] == "trace-123"


def test_metrics_endpoint_exposes_request_counter(app):
    client = TestClient(app)
    client.get("/ok")
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    text = metrics.text
    assert "pixano_inference_requests_total" in text
    assert "pixano_inference_request_duration_seconds" in text
    # The route-template label (not the raw URL) is used to bound cardinality.
    assert 'path="/ok"' in text


def test_request_id_filter_injects_current_id():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "msg", None, None)
    RequestIdFilter().filter(record)
    assert record.request_id == get_request_id()  # "-" outside any request


def test_configure_logging_is_idempotent_and_safe():
    # Should not raise, and should leave logging usable.
    configure_logging(level="WARNING", json_logs=False)
    configure_logging(level="INFO", json_logs=True)
    logging.getLogger("pixano_inference.test").info("hello")
