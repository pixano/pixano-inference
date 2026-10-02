# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Guardrail: the client and the wire schemas work on a base install.

An application that only calls a server installs ``pixano-inference`` without the ``server`` extra:
httpx, pydantic and numpy. These tests use nothing else, so the ``light_install`` CI job runs them
in an environment where Ray, FastAPI, Pillow and pycocotools are absent.
"""

import subprocess
import sys


HEAVY_MODULES = ["ray", "ray.serve", "fastapi", "uvicorn", "starlette", "torch", "transformers"]


def test_import_pulls_no_heavy_dependency():
    # Run in a fresh interpreter so the check is unaffected by whatever the surrounding test
    # session already imported (the server tests pull ray/fastapi).
    script = (
        "import sys, pixano_inference.client, pixano_inference.schemas\n"
        f"heavy = {HEAVY_MODULES!r}\n"
        "leaked = [m for m in heavy if m in sys.modules]\n"
        "assert not leaked, 'client pulled in heavy modules: ' + repr(leaked)\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_build_request_and_parse_response_without_masks():
    """A request builds and an NDArray-binary response parses with no pycocotools/Pillow."""
    import numpy as np

    from pixano_inference.schemas import DetectionRequest, EmbeddingResponse, NDArrayFloat

    request = DetectionRequest(model="det", image="https://example.com/cat.jpg", classes=["cat"])
    body = request.model_dump(mode="json", by_alias=True)
    assert body["model"] == "det"
    assert body["boxThreshold"] == 0.5  # camelCase on the wire

    vectors = np.arange(8, dtype=np.float32).reshape(2, 4)
    response = EmbeddingResponse.model_validate(
        {
            "id": "1",
            "status": "success",
            "timestamp": "2026-01-01T00:00:00Z",
            "metadata": {},
            "data": {"embeddings": NDArrayFloat.from_numpy(vectors).model_dump(), "dim": 4},
        }
    )
    assert response.data.dim == 4
    assert response.data.embeddings.to_numpy().shape == (2, 4)
    np.testing.assert_array_equal(response.data.embeddings.to_numpy(), vectors)


def test_client_construction_and_url_validation():
    from pixano_inference.client import PixanoInferenceError, SyncPixanoInferenceClient

    client = SyncPixanoInferenceClient("http://localhost:7463/", api_key="secret")
    assert client.url == "http://localhost:7463"
    assert client._headers()["X-API-Key"] == "secret"
    client.close()

    try:
        SyncPixanoInferenceClient("not-a-url")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected a ValueError for an invalid URL.")

    # PixanoInferenceError carries the server envelope.
    err = PixanoInferenceError(503, "unavailable", "nope", "req-1")
    assert err.status_code == 503 and err.request_id == "req-1"
