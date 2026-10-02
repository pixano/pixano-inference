# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Every ``/binary`` route delivers the uploaded bytes, unchanged, to the model.

The payload is deliberately not valid UTF-8, like any real image: bytes that happen to decode as
text would be accepted by a ``str`` field and reach the model as a string.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pixano_inference.models import CAPABILITIES, get_capability
from pixano_inference.ray.app import create_ray_serve_app
from pixano_inference.ray.config import RayServeConfig
from pixano_inference.schemas import (
    DetectionOutput,
    EmbeddingOutput,
    NDArrayFloat,
    SegmentationOutput,
    TrackingOutput,
    UsageInfo,
    VLMOutput,
)
from tests.fakes import FakeHandle


RAW = b"\x89PNG\r\n\x1a\n\x00\xff\xfe\x80"

# Per capability: the JSON metadata part of the upload, and what the fake model returns.
_CASES = {
    "segmentation": (
        {"model": "m"},
        SegmentationOutput(masks=[], scores=NDArrayFloat.from_numpy(np.array([0.9], dtype=np.float32))),
    ),
    "detection": ({"model": "m"}, DetectionOutput(boxes=[], scores=[], classes=[])),
    "embedding": (
        {"model": "m"},
        EmbeddingOutput(embeddings=NDArrayFloat.from_numpy(np.zeros((1, 2), dtype=np.float32)), dim=2),
    ),
    "vlm": (
        {"model": "m", "prompt": "describe", "maxNewTokens": 8},
        VLMOutput(generated_text="ok", usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
    ),
    "tracking": (
        {"model": "m", "objectsIds": [1], "frameIndexes": [0]},
        TrackingOutput(objects_ids=[1], frame_indexes=[0], masks=[]),
    ),
}

_BINARY_CAPABILITIES = [spec.name for spec in CAPABILITIES if spec.binary is not None]


@pytest.fixture
def client():
    app, _ = create_ray_serve_app(RayServeConfig(num_gpus=0))
    return TestClient(app, raise_server_exceptions=False)


def _install(client: TestClient, monkeypatch, *, handle, capability):
    manager = client.app.state.deployment_manager
    monkeypatch.setattr(manager, "get_handle", lambda name: handle)
    monkeypatch.setattr(manager, "get_model_capability", lambda name: capability)
    monkeypatch.setattr(manager, "get_model_metadata", lambda name: {"capability": capability})


def test_every_binary_capability_has_a_case():
    assert sorted(_CASES) == sorted(_BINARY_CAPABILITIES)


@pytest.mark.parametrize("capability", _BINARY_CAPABILITIES)
def test_binary_route_delivers_raw_bytes_to_the_model(client, monkeypatch, capability):
    spec = get_capability(capability)
    metadata, result = _CASES[capability]
    handle = FakeHandle(result)
    _install(client, monkeypatch, handle=handle, capability=capability)

    single_file = spec.binary.file_field == "image"
    uploads = [RAW] if single_file else [RAW, RAW + b"\x01"]
    files = [("metadata", ("metadata.json", json.dumps(metadata), "application/json"))]
    files += [
        (spec.binary.file_field, (f"part{i}.bin", data, "application/octet-stream")) for i, data in enumerate(uploads)
    ]

    resp = client.post(f"/v1/inference/{capability}/binary", files=files)

    assert resp.status_code == 200, resp.text
    received = getattr(handle.predict.last_input, spec.binary.payload_key)
    expected = uploads[0] if single_file else uploads
    for item in [received] if single_file else received:
        assert isinstance(item, bytes), f"the model received {type(item).__name__}, not bytes"
    assert received == expected


def test_binary_route_reports_a_validation_error_as_422(client, monkeypatch):
    """An invalid upload is a 422 with a JSON body, even though the payload is not text."""
    _install(client, monkeypatch, handle=FakeHandle(None), capability="embedding")

    resp = client.post(
        "/v1/inference/embedding/binary",
        files=[
            # Both modalities at once: rejected by EmbeddingInput.
            ("metadata", ("metadata.json", json.dumps({"model": "m", "text": "a cat"}), "application/json")),
            ("image", ("image.png", RAW, "image/png")),
        ],
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"]
