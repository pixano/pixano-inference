<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# pixano-inference-client

The **lightweight client** for a [Pixano Inference](https://github.com/pixano/pixano-inference)
server. Install this when your app only needs to *call* a running server over HTTP — it pulls in
just `httpx`, `pydantic`, and `numpy`, **not** the server stack (`ray[serve]`, `fastapi`,
`uvicorn`, torch, …).

```bash
pip install pixano-inference-client
```

It ships the same client classes and wire schemas as the full `pixano-inference` package, so code
written against either works unchanged.

## Use

```python
from pixano_inference_client import SyncPixanoInferenceClient, DetectionRequest, EmbeddingRequest

client = SyncPixanoInferenceClient("http://localhost:7463", api_key="…")

det = client.detection(DetectionRequest(model="my-detector", image="https://example.com/cat.jpg"))
print(det.data.boxes, det.data.scores, det.data.classes)

emb = client.embedding(EmbeddingRequest(model="clip", text="a photo of a cat"))
print(emb.data.dim, emb.data.embeddings.to_numpy().shape)
```

An asynchronous twin, `PixanoInferenceClient`, exposes the same methods with `await`. Both accept
an optional API key, retry transient failures with backoff, and raise `PixanoInferenceError`
carrying the server's `{code, message, requestId}` envelope.

## What's included

- Clients: `PixanoInferenceClient` (async), `SyncPixanoInferenceClient` (sync), `PixanoInferenceError`.
- Request/response types for every `/v1` capability: `SegmentationRequest/Response`,
  `DetectionRequest/Response`, `VLMRequest/Response`, `NERRequest/Response`,
  `EmbeddingRequest/Response`, `TrackingRequestV1`/`TrackingResponse`.
- Wire types: `NDArray`/`NDArrayFloat` (compact base64 arrays), `CompressedRLE` (masks),
  `ModelInfo`, `ModelStatusInfo`, `JobStatus`, `DeployModelRequest`, and the capability I/O models.

## Optional: mask codecs

Decoding a segmentation response into a numpy mask (`CompressedRLE.to_mask()`), or building one
from a mask (`CompressedRLE.from_mask()`), needs `pycocotools` (and `Pillow` for image inputs).
These are **optional** — parsing a normal compressed-bytes response works without them:

```bash
pip install pixano-inference-client[masks]
```

## Relationship to `pixano-inference`

The full server package `pixano-inference` **depends on** this package and re-exports it, so
`from pixano_inference.client import PixanoInferenceClient` and
`from pixano_inference_client import PixanoInferenceClient` refer to the same classes. Server
operators install `pixano-inference`; third-party callers install just `pixano-inference-client`.
