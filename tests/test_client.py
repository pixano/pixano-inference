# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the /v1 async and sync clients."""

import base64

import numpy as np
import pytest
from pytest_httpx import HTTPXMock

from pixano_inference.client import (
    PixanoInferenceClient,
    PixanoInferenceError,
    SyncPixanoInferenceClient,
)
from pixano_inference.schemas import DetectionRequest, EmbeddingRequest, SegmentationRequest
from pixano_inference.schemas.nd_array import NDArrayFloat
from pixano_inference.schemas.v1 import TrackingRequestV1


URL = "http://localhost:8081"


def _ndarray_wire(values: list[float]) -> dict:
    return NDArrayFloat.from_numpy(np.array(values, dtype=np.float32)).model_dump(by_alias=True)


def _segmentation_response(id_: str = "seg-1") -> dict:
    counts_b64 = base64.b64encode(b"\x01\x02\x03").decode()
    return {
        "id": id_,
        "status": "SUCCESS",
        "timestamp": "2024-01-01T00:00:00Z",
        "processingTime": 0.5,
        "metadata": {"capability": "segmentation"},
        "data": {
            "masks": [[{"size": [100, 100], "counts": counts_b64}]],
            "scores": _ndarray_wire([0.95]),
        },
    }


# --- Construction -------------------------------------------------------------------


def test_init_and_url_normalization():
    assert PixanoInferenceClient(url=URL).url == URL
    assert PixanoInferenceClient(url=f"{URL}/").url == URL
    with pytest.raises(ValueError, match="Invalid URL, got 'wrongurl'."):
        PixanoInferenceClient(url="wrongurl")


# --- Inference ----------------------------------------------------------------------


async def test_segmentation_parses_camel_response(httpx_mock: HTTPXMock, simple_pixano_inference_client):
    httpx_mock.add_response(json=_segmentation_response("seg-1"))
    request = SegmentationRequest(model="sam2", image="https://example.com/x.jpg")
    result = await simple_pixano_inference_client.segmentation(request)
    assert result.id == "seg-1"
    assert result.status == "SUCCESS"
    # NDArray decoded from the binary wire format.
    np.testing.assert_allclose(result.data.scores.to_numpy(), np.array([0.95], dtype=np.float32))


async def test_request_targets_v1_and_sends_api_key(httpx_mock: HTTPXMock):
    httpx_mock.add_response(json=_segmentation_response())
    client = PixanoInferenceClient(url=URL, api_key="secret", max_retries=0)
    await client.segmentation(SegmentationRequest(model="sam2", image="https://example.com/x.jpg"))
    request = httpx_mock.get_request()
    assert request.url.path == "/v1/inference/segmentation"
    assert request.headers["X-API-Key"] == "secret"


async def test_detection_serializes_camelcase_body(httpx_mock: HTTPXMock, simple_pixano_inference_client):
    httpx_mock.add_response(
        json={
            "id": "d1",
            "status": "SUCCESS",
            "timestamp": "2024-01-01T00:00:00Z",
            "processingTime": 0.1,
            "metadata": {},
            "data": {"boxes": [[1, 2, 3, 4]], "scores": [0.9], "classes": ["truck"], "masks": None},
        }
    )
    await simple_pixano_inference_client.detection(
        DetectionRequest(model="gd", image="https://example.com/x.jpg", box_threshold=0.3)
    )
    body = httpx_mock.get_request().read().decode()
    assert "boxThreshold" in body  # camelCase on the wire


async def test_embedding_parses_binary_vectors(httpx_mock: HTTPXMock, simple_pixano_inference_client):
    vectors = np.arange(4, dtype=np.float32).reshape(1, 4)
    httpx_mock.add_response(
        json={
            "id": "e1",
            "status": "SUCCESS",
            "timestamp": "2024-01-01T00:00:00Z",
            "processingTime": 0.1,
            "metadata": {},
            "data": {"embeddings": _ndarray_wire([0.0, 1.0, 2.0, 3.0]), "dim": 4},
        }
    )
    result = await simple_pixano_inference_client.embedding(EmbeddingRequest(model="clip", text="a cat"))
    assert result.data.dim == 4
    np.testing.assert_allclose(result.data.embeddings.to_numpy(), vectors.reshape(-1))
    assert httpx_mock.get_request().url.path == "/v1/inference/embedding"


# --- Errors -------------------------------------------------------------------------


async def test_error_envelope_becomes_exception(httpx_mock: HTTPXMock, simple_pixano_inference_client):
    httpx_mock.add_response(
        status_code=404,
        json={"error": {"code": "not_found", "message": "Model 'x' not found", "requestId": "req-9"}},
    )
    with pytest.raises(PixanoInferenceError) as exc:
        await simple_pixano_inference_client.segmentation(
            SegmentationRequest(model="x", image="https://example.com/x.jpg")
        )
    assert exc.value.status_code == 404
    assert exc.value.code == "not_found"
    assert exc.value.request_id == "req-9"


async def test_retries_then_succeeds_on_503(httpx_mock: HTTPXMock):
    httpx_mock.add_response(status_code=503, json={"error": {"code": "unavailable", "message": "starting"}})
    httpx_mock.add_response(json=_segmentation_response("after-retry"))
    client = PixanoInferenceClient(url=URL, max_retries=2, backoff_factor=0.0)
    result = await client.segmentation(SegmentationRequest(model="sam2", image="https://example.com/x.jpg"))
    assert result.id == "after-retry"
    assert len(httpx_mock.get_requests()) == 2


# --- Jobs ---------------------------------------------------------------------------


async def test_job_lifecycle(httpx_mock: HTTPXMock, simple_pixano_inference_client):
    tracking = TrackingRequestV1(model="sam2-video", video=["f0.png"], objects_ids=[1], frame_indexes=[0])
    httpx_mock.add_response(url=f"{URL}/v1/inference/tracking/jobs", json={"jobId": "j1", "status": "running"})
    submitted = await simple_pixano_inference_client.submit_tracking_job(tracking)
    assert submitted.job_id == "j1"
    assert submitted.status == "running"

    httpx_mock.add_response(
        url=f"{URL}/v1/jobs/j1",
        json={"jobId": "j1", "status": "completed", "data": {"frameIndexes": [0]}},
    )
    done = await simple_pixano_inference_client.wait_for_job("j1", poll_interval=0.0)
    assert done.status == "completed"
    assert done.data == {"frameIndexes": [0]}


# --- Admin / service ----------------------------------------------------------------


async def test_list_models(httpx_mock: HTTPXMock, simple_pixano_inference_client):
    httpx_mock.add_response(
        url=f"{URL}/v1/models",
        json=[{"name": "sam2", "capability": "segmentation", "modelClass": "Sam2ImageModel", "status": "RUNNING"}],
    )
    models = await simple_pixano_inference_client.list_models()
    assert models[0].name == "sam2"
    assert models[0].status == "RUNNING"


async def test_ready_does_not_raise_on_503(httpx_mock: HTTPXMock, simple_pixano_inference_client):
    httpx_mock.add_response(url=f"{URL}/v1/ready", status_code=503, json={"ready": False, "models_loaded": 0})
    result = await simple_pixano_inference_client.ready()
    assert result["ready"] is False


# --- Sync twin ----------------------------------------------------------------------


def test_sync_client_segmentation(httpx_mock: HTTPXMock, sync_pixano_inference_client: SyncPixanoInferenceClient):
    httpx_mock.add_response(json=_segmentation_response("sync-1"))
    result = sync_pixano_inference_client.segmentation(
        SegmentationRequest(model="sam2", image="https://example.com/x.jpg")
    )
    assert result.id == "sync-1"
    assert httpx_mock.get_request().url.path == "/v1/inference/segmentation"
