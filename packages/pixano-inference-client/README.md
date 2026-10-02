<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# pixano-inference-client (deprecated)

This distribution is a **deprecated alias**. The client and the wire schemas it used to hold are
part of the base install of [`pixano-inference`](https://github.com/pixano/pixano-inference),
which needs only `httpx`, `pydantic` and `numpy`: the server stack (`ray[serve]`, `fastapi`,
`uvicorn`) is behind the `server` extra, so an application that only _calls_ a server never
installs it.

```bash
pip install pixano-inference            # client and wire schemas, no server stack
pip install "pixano-inference[masks]"   # plus the mask codecs (Pillow, pycocotools)
```

## Migrating

Replace the dependency and the imports. The classes are the same objects, so nothing else changes.

| With `pixano-inference-client`                                    | With `pixano-inference`                                            |
| ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| `pip install pixano-inference-client`                             | `pip install pixano-inference`                                     |
| `pip install pixano-inference-client[masks]`                      | `pip install "pixano-inference[masks]"`                            |
| `from pixano_inference_client import SyncPixanoInferenceClient`   | `from pixano_inference.client import SyncPixanoInferenceClient`    |
| `from pixano_inference_client import PixanoInferenceError`        | `from pixano_inference.client import PixanoInferenceError`         |
| `from pixano_inference_client import DetectionRequest, NDArray …` | `from pixano_inference.schemas import DetectionRequest, NDArray …` |
| `from pixano_inference_client.rle import CompressedRLE`           | `from pixano_inference.schemas.rle import CompressedRLE`           |

```python
from pixano_inference.client import SyncPixanoInferenceClient
from pixano_inference.schemas import DetectionRequest

client = SyncPixanoInferenceClient("http://localhost:7463", api_key="…")
det = client.detection(DetectionRequest(model="my-detector", image="https://example.com/cat.jpg"))
print(det.data.boxes, det.data.scores, det.data.classes)
```

## What this version does

`pixano-inference-client` 0.2 depends on `pixano-inference >= 0.7, < 0.8` and re-exports the same
names as 0.1 (`pixano_inference_client` and its submodules), so existing imports keep working.
Importing it emits a `DeprecationWarning`; a test suite that turns warnings into errors fails on
that import until it is migrated.

Do not install `pixano-inference-client` 0.1 next to `pixano-inference` 0.7 or later: each would
bring its own copy of the wire types, and an object built from one does not validate against the
other. `pixano-inference` 0.6 keeps working with `pixano-inference-client` 0.1.
