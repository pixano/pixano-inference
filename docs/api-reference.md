# Pixano Inference HTTP API

This document describes the HTTP API exposed by the Ray Serve-based Pixano
Inference server.

**Base URL:** `http://<host>:<port>` with default `http://127.0.0.1:7463`

Start the server with a Python config file:

```bash
pixano-inference --config models.py
```

## Overview

- Models are loaded at startup from a Python `.py` config file passed to `--config`.
- All inference routes are synchronous `POST` endpoints.
- There are no runtime HTTP endpoints for deploying or undeploying models.
- Every inference request includes a `model` field that must match a deployed model name.
- Endpoint families are capability-based: segmentation, detection, tracking, and VLM.

## Service endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Basic API metadata and docs link |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness summary |
| `GET` | `/app/settings/` | Server settings and resource summary |
| `GET` | `/app/models/` | List deployed models |

### `GET /app/settings/`

Example response:

```json
{
  "app_name": "Pixano Inference",
  "app_version": "0.6.0",
  "app_description": "Pixano Inference API powered by Ray Serve",
  "num_cpus": 8,
  "num_gpus": 2,
  "num_nodes": 1,
  "gpus_used": 1.0,
  "gpu_to_model": {},
  "models": ["sam2-image"],
  "models_to_capability": {
    "sam2-image": "segmentation"
  }
}
```

### `GET /app/models/`

Returns a list of `ModelInfo` objects:

```json
[
  {
    "name": "sam2-image",
    "capability": "segmentation",
    "model_path": "facebook/sam2-hiera-base-plus",
    "model_class": "Sam2ImageModel"
  }
]
```

## Inference endpoints

| Method | Path | Request schema | Response schema | Python client helper |
|---|---|---|---|---|
| `POST` | `/inference/segmentation/` | `SegmentationRequest` | `SegmentationResponse` | `client.segmentation()` |
| `POST` | `/inference/detection/` | `DetectionRequest` | `DetectionResponse` | `client.detection()` |
| `POST` | `/inference/tracking/` | `TrackingRequest` | `TrackingResponse` | `client.tracking()` |
| `POST` | `/inference/vlm/` | `VLMRequest` | `VLMResponse` | `client.vlm()` |

If a model exists but does not support the endpoint capability, the server
returns `400`.

The request and response models are available from `pixano_inference.schemas`.

### Example: segmentation

```json
{
  "model": "sam2-image",
  "image": "data:image/png;base64,...",
  "points": [[[200, 175]]],
  "labels": [[1]]
}
```

### Example: detection

```json
{
  "model": "grounding-dino",
  "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
  "classes": ["cat", "remote control"],
  "box_threshold": 0.3,
  "text_threshold": 0.2
}
```

## Response envelope

All inference endpoints return the same top-level envelope:

```json
{
  "id": "ray-sam2-image-1739000000000",
  "status": "SUCCESS",
  "timestamp": "2026-01-01T12:00:00",
  "processing_time": 0.234,
  "metadata": {
    "model_name": "sam2-image",
    "capability": "segmentation",
    "model_class": "Sam2ImageModel"
  },
  "data": {}
}
```

| Field | Description |
|---|---|
| `id` | Server-generated request identifier |
| `status` | Inference status, typically `SUCCESS` |
| `timestamp` | Response timestamp |
| `processing_time` | End-to-end inference time in seconds |
| `metadata` | Deployment metadata for the model that handled the request |
| `data` | Capability-specific payload |

## Python client

```python
import asyncio

from pixano_inference.client import PixanoInferenceClient
from pixano_inference.schemas import SegmentationRequest


async def main() -> None:
    client = PixanoInferenceClient.connect("http://localhost:7463")
    request = SegmentationRequest(
        model="sam2-image",
        image="data:image/png;base64,...",
        points=[[[200, 175]]],
        labels=[[1]],
    )
    response = await client.segmentation(request)
    print(response.processing_time)
    print(response.data.scores.to_numpy())


asyncio.run(main())
```

## Error responses

- `400` when the model exists but does not support the requested capability.
- `404` when the requested model name is not deployed.
- `422` when the request body fails schema validation.
- `500` when inference fails inside the model deployment.

The Python client raises `fastapi.HTTPException` with the server error detail
when a request is unsuccessful.
