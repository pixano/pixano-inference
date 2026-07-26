# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the /v1 API: camelCase contract, NDArray binary, NER, jobs, error envelope."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pixano_inference.models.detection import DetectionOutput
from pixano_inference.models.embedding import EmbeddingOutput
from pixano_inference.models.ner import NEREntity, NEROutput
from pixano_inference.models.segmentation import SegmentationOutput
from pixano_inference.models.tracking import TrackingOutput
from pixano_inference.models.vlm import UsageInfo, VLMOutput
from pixano_inference.ray.app import create_ray_serve_app
from pixano_inference.ray.config import RayServeConfig
from pixano_inference.schemas.nd_array import NDArrayFloat
from pixano_inference.schemas.rle import CompressedRLE
from tests.fakes import FakeHandle


@pytest.fixture
def client():
    app, _ = create_ray_serve_app(RayServeConfig(num_gpus=0))
    return TestClient(app, raise_server_exceptions=False)


def _install(client: TestClient, monkeypatch, *, handle, capability, metadata=None):
    manager = client.app.state.deployment_manager
    monkeypatch.setattr(manager, "get_handle", lambda name: handle)
    monkeypatch.setattr(manager, "get_model_capability", lambda name: capability if handle is not None else None)
    monkeypatch.setattr(manager, "get_model_metadata", lambda name: metadata or {"capability": capability})


# --- Inference: camelCase + typed responses -----------------------------------------


def test_segmentation_camel_and_binary_ndarray(client, monkeypatch):
    scores = NDArrayFloat.from_numpy(np.array([0.9], dtype=np.float32))
    logits = NDArrayFloat.from_numpy(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    result = SegmentationOutput(
        masks=[[CompressedRLE.from_mask(np.array([[1, 0], [0, 1]], dtype=np.uint8))]],
        scores=scores,
        mask_logits=logits,
    )
    handle = FakeHandle(result)
    _install(client, monkeypatch, handle=handle, capability="segmentation")

    resp = client.post(
        "/v1/inference/segmentation",
        json={"model": "sam2", "image": "https://example.com/x.jpg", "returnLogits": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Envelope is camelCase.
    assert "processingTime" in body
    # NDArray serialized as {shape, dtype, data} (not a float list).
    ml = body["data"]["maskLogits"]
    assert set(ml.keys()) == {"shape", "dtype", "data"}
    assert ml["shape"] == [2, 2]
    # camelCase request field accepted.
    assert handle.predict.last_input.return_logits is True
    # And the binary payload round-trips back to the same array.
    restored = NDArrayFloat.model_validate(ml).to_numpy()
    np.testing.assert_allclose(restored, np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))


def test_detection_route(client, monkeypatch):
    result = DetectionOutput(boxes=[[1, 2, 3, 4]], scores=[0.9], classes=["truck"], masks=None)
    _install(client, monkeypatch, handle=FakeHandle(result), capability="detection")
    resp = client.post("/v1/inference/detection", json={"model": "gd", "image": "https://example.com/x.jpg"})
    assert resp.status_code == 200
    assert resp.json()["data"]["classes"] == ["truck"]


def test_vlm_route(client, monkeypatch):
    result = VLMOutput(
        generated_text="hello",
        usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    _install(client, monkeypatch, handle=FakeHandle(result), capability="vlm")
    resp = client.post(
        "/v1/inference/vlm",
        json={"model": "qwen", "prompt": "hi", "images": ["https://example.com/x.jpg"], "maxNewTokens": 8},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["generatedText"] == "hello"
    assert body["data"]["usage"]["promptTokens"] == 1


def test_ner_route_is_servable(client, monkeypatch):
    result = NEROutput(entities=[NEREntity(text="Pixano", label="ORG", start=0, end=6, score=0.98)])
    _install(client, monkeypatch, handle=FakeHandle(result), capability="ner")
    resp = client.post("/v1/inference/ner", json={"model": "ner-model", "text": "Pixano rocks"})
    assert resp.status_code == 200
    assert resp.json()["data"]["entities"][0]["label"] == "ORG"


def test_embedding_text_and_image_share_binary_wire(client, monkeypatch):
    vectors = np.arange(8, dtype=np.float32).reshape(2, 4)
    result = EmbeddingOutput(embeddings=NDArrayFloat.from_numpy(vectors), dim=4)
    handle = FakeHandle(result)
    _install(client, monkeypatch, handle=handle, capability="embedding")

    # Text request (batch of 2).
    text_resp = client.post("/v1/inference/embedding", json={"model": "clip", "text": ["a cat", "a dog"]})
    assert text_resp.status_code == 200
    body = text_resp.json()["data"]
    assert body["dim"] == 4
    assert set(body["embeddings"].keys()) == {"shape", "dtype", "data"}
    restored = NDArrayFloat.model_validate(body["embeddings"]).to_numpy()
    np.testing.assert_allclose(restored, vectors)
    assert handle.predict.last_input.text == ["a cat", "a dog"]

    # Image request (URL).
    img_resp = client.post("/v1/inference/embedding", json={"model": "clip", "image": "https://example.com/x.jpg"})
    assert img_resp.status_code == 200


def test_embedding_requires_exactly_one_modality(client, monkeypatch):
    _install(client, monkeypatch, handle=FakeHandle(None), capability="embedding")
    # Neither image nor text -> 422 validation error (never reaches the model).
    resp = client.post("/v1/inference/embedding", json={"model": "clip"})
    assert resp.status_code == 422
    # Both -> 422.
    resp = client.post("/v1/inference/embedding", json={"model": "clip", "text": "a", "image": "https://x/y.jpg"})
    assert resp.status_code == 422


def test_tracking_nested_keyframes_map_to_flat_input(client, monkeypatch):
    result = TrackingOutput(objects_ids=[1], frame_indexes=[1], masks=[])
    handle = FakeHandle(result)
    _install(client, monkeypatch, handle=handle, capability="tracking")
    mask = CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))

    resp = client.post(
        "/v1/inference/tracking",
        json={
            "model": "sam2-video",
            "video": ["f0.png", "f1.png"],
            "objectsIds": [1],
            "frameIndexes": [0],
            "interval": {"startFrame": 0, "endFrame": 1, "direction": "forward"},
            "keyframes": [{"frameIndex": 0, "prompts": {"mask": mask.model_dump(mode="json")}}],
        },
    )
    assert resp.status_code == 200
    sent = handle.predict.last_input
    # Nested prompts flattened onto the internal TrackingKeyframe.
    assert sent.keyframes[0].frame_index == 0
    assert sent.keyframes[0].mask is not None
    assert sent.interval.start_frame == 0 and sent.interval.end_frame == 1


def test_segmentation_binary_route(client, monkeypatch):
    result = SegmentationOutput(
        masks=[[CompressedRLE.from_mask(np.array([[1, 0], [0, 1]], dtype=np.uint8))]],
        scores=NDArrayFloat.from_numpy(np.array([0.9], dtype=np.float32)),
    )
    handle = FakeHandle(result)
    _install(client, monkeypatch, handle=handle, capability="segmentation")
    import json

    resp = client.post(
        "/v1/inference/segmentation/binary",
        files=[
            ("metadata", ("metadata.json", json.dumps({"model": "sam2"}), "application/json")),
            ("image", ("image.png", b"binary-image", "image/png")),
        ],
    )
    assert resp.status_code == 200
    assert handle.predict.last_input.image == b"binary-image"


# --- Error envelope -----------------------------------------------------------------


def test_error_envelope_on_missing_model(client, monkeypatch):
    _install(client, monkeypatch, handle=None, capability="segmentation")
    resp = client.post("/v1/inference/segmentation", json={"model": "nope", "image": "https://example.com/x.jpg"})
    assert resp.status_code == 404
    body = resp.json()
    assert set(body["error"].keys()) == {"code", "message", "requestId"}
    assert body["error"]["code"] == "not_found"


def test_capability_mismatch_returns_400(client, monkeypatch):
    result = DetectionOutput(boxes=[], scores=[], classes=[], masks=None)
    _install(client, monkeypatch, handle=FakeHandle(result), capability="detection")
    resp = client.post("/v1/inference/segmentation", json={"model": "gd", "image": "https://example.com/x.jpg"})
    assert resp.status_code == 400
    assert "does not support 'segmentation'" in resp.json()["error"]["message"]


# --- Jobs ---------------------------------------------------------------------------


def test_tracking_job_submit_status_cancel(client, monkeypatch):
    result = TrackingOutput(objects_ids=[1], frame_indexes=[0], masks=[])
    handle = FakeHandle(result, pending=True)  # stays running until cancelled
    _install(client, monkeypatch, handle=handle, capability="tracking")

    submit = client.post(
        "/v1/inference/tracking/jobs",
        json={"model": "sam2-video", "video": ["f0.png"], "objectsIds": [1], "frameIndexes": [0]},
    )
    assert submit.status_code == 202
    body = submit.json()
    assert body["status"] == "running"
    job_id = body["jobId"]

    assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "running"
    assert client.get("/v1/jobs/unknown").status_code == 404

    cancel = client.delete(f"/v1/jobs/{job_id}")
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "canceled"
