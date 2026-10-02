<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Custom Model Deployment: YOLO Detection

This tutorial deploys a third-party model (Ultralytics YOLO) as a Pixano Inference custom
detection service, packaged as an **installable plugin** — the recommended way to extend the
server (see [../../docs/ray_serve/custom_models.md](../../docs/ray_serve/custom_models.md)).
For a framework-free (numpy-only) starting point, see [`../numpy_detector`](../numpy_detector).

> **Licence note:** Ultralytics is AGPL-3.0. Deploying it as a network service carries
> source-disclosure obligations (AGPL section 13). Review the licence before production use.

## Install

The example is a self-contained package that declares `pixano_inference.models` entry
points; installing it makes `YOLOModel` (detection) and `YOLOByteTrackModel` (multi-object
tracking, see [Tracking by detection with ByteTrack](#tracking-by-detection-with-bytetrack))
discoverable by name (and importable in every Ray Serve worker):

```bash
uv sync --project examples/yolo    # its own environment: the Pixano Inference core + ultralytics
```

## Project Structure

```
examples/yolo/
    pyproject.toml                 # package metadata + the pixano_inference.models entry points
    src/pixano_yolo/
        __init__.py
        model.py                   # YOLOModel (extends DetectionModel, @register_model)
        tracker.py                 # YOLOByteTrackModel (extends TrackingModel, @register_model)
    config.py                      # deployment config, references "YOLOModel" by name
    test_yolo.py                   # end-to-end client script
    config_tracking.py             # deployment config for "YOLOByteTrackModel"
    test_tracking.py               # end-to-end client script for tracking
```

## Step 1: Implement the Model

Create a model class that extends one of the built-in base classes. For object detection, extend `DetectionModel`.

```python
# model.py
from pixano_inference.models.detection import DetectionInput, DetectionModel, DetectionOutput
from pixano_inference.models.registry import register_model
from pixano_inference.configs import ModelDeploymentConfig


@register_model("YOLOModel")
class YOLOModel(DetectionModel):

    def __init__(self, config: ModelDeploymentConfig) -> None:
        super().__init__(config)
        self._model = None

    def load_model(self) -> None:
        """Called once when the Ray actor starts. Load weights here."""
        from ultralytics import YOLO

        path = dict(self._config.model_params).pop("path")

        device = "cpu"
        if self._config.resources.num_gpus > 0:
            import torch
            if torch.cuda.is_available():
                device = "cuda"

        self._model = YOLO(path)
        self._model.to(device)

    def predict(self, input: DetectionInput) -> DetectionOutput:
        """Run inference on a single request."""
        from pixano_inference.utils.media import convert_string_to_image

        pil_image = convert_string_to_image(input.image)
        results = self._model.predict(pil_image, conf=input.box_threshold)
        result = results[0]

        boxes, scores, class_names = [], [], []
        if result.boxes is not None and len(result.boxes):
            for box, conf, cls_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                boxes.append([int(round(c)) for c in box.tolist()])
                scores.append(float(conf))
                class_names.append(result.names[int(cls_id)])

        return DetectionOutput(boxes=boxes, scores=scores, classes=class_names)

    def unload(self) -> None:
        """Free resources when the model is removed."""
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
```

Key points:

- **`load_model()`** is called once when the Ray actor initializes. Download weights, load checkpoints, and move to device here.
- **`predict()`** receives a typed `DetectionInput` and must return a `DetectionOutput`.
- **`unload()`** is called when the model is removed. Free GPU memory and clean up.

## Step 2: Write the Deployment Config

Create a Python config file that defines a `models` list. Reference the model **by name** —
the entry point already registered it, so no import is needed:

```python
# config.py
from pixano_inference.configs import DeploymentConfig, ModelConfig

models = [
    ModelConfig(
        name="yolo26s",
        model_class="YOLOModel",
        model_params={"path": "yolo26s.pt"},  # Passed to model via config
        deployment=DeploymentConfig(
            num_gpus=1,          # GPUs per replica (set 0 for CPU-only)
            num_cpus=1,          # CPUs per replica
            min_replicas=1,      # Keep a replica warm (0 enables scale-to-zero)
            max_replicas=2,      # Max concurrent replicas
        ),
    ),
]
```

## Step 3: Start the Server

Because the package is installed, no `PYTHONPATH`/`--module-path` is needed — the model is
discovered via its entry point and referenced by name in the config:

```bash
uv run --project examples/yolo pixano-inference --config examples/yolo/config.py
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:7463 (Press CTRL+C to quit)
```

## Step 4: Run the End-to-End Test

With the server running, use the included test script:

```bash
# With a real image
uv run --project examples/yolo python examples/yolo/test_yolo.py \
    --server-url http://127.0.0.1:7463 \
    --model-name yolo26s \
    --image path/to/image.jpg

# With a synthetic test image (no --image flag)
uv run --project examples/yolo python examples/yolo/test_yolo.py \
    --server-url http://127.0.0.1:7463 \
    --model-name yolo26s
```

Example output:

```
Detection
Status: SUCCESS
Processing time: 0.072s
Detections: 5
  [0] class=person, score=0.944, box=[668, 395, 810, 881]
  [1] class=person, score=0.930, box=[48, 400, 247, 903]
  [2] class=bus, score=0.928, box=[1, 229, 806, 742]
  [3] class=person, score=0.559, box=[221, 406, 345, 862]
  [4] class=person, score=0.428, box=[0, 553, 78, 876]
```

You can also use the Python client directly:

```python
from pixano_inference.client import PixanoInferenceClient
from pixano_inference.schemas import DetectionRequest

client = PixanoInferenceClient.connect("http://localhost:7463")

request = DetectionRequest(
    model="yolo26s",
    image="https://ultralytics.com/images/bus.jpg",  # URL, file path, or base64
    box_threshold=0.3,
)
result = await client.detection(request)

for box, score, cls in zip(result.data.boxes, result.data.scores, result.data.classes):
    print(f"{cls}: {score:.3f} {box}")
```

## HTTP API

The detection endpoint is:

```
POST /inference/detection/
```

Request body:

```json
{
  "model": "yolo26s",
  "image": "https://ultralytics.com/images/bus.jpg",
  "box_threshold": 0.5
}
```

Response:

```json
{
  "id": "ray-yolo26s-1711817600",
  "status": "SUCCESS",
  "processing_time": 0.072,
  "data": {
    "boxes": [[668, 395, 810, 881], [48, 400, 247, 903]],
    "scores": [0.944, 0.930],
    "classes": ["person", "person"],
    "masks": null
  }
}
```

## Tracking by detection with ByteTrack

The same package ships a second model, `YOLOByteTrackModel`, built on the
[track mode](https://docs.ultralytics.com/modes/track) of Ultralytics: YOLO detects the objects
of each frame and [ByteTrack](https://arxiv.org/abs/2110.06864) links the detections into tracks.
It extends `TrackingModel`, and unlike a promptable tracker (SAM2) it takes **no prompt**: the
request carries only the video, and the model decides how many tracks there are and numbers them.

```python
# tracker.py (abridged)
@register_model("YOLOByteTrackModel")
class YOLOByteTrackModel(TrackingModel):

    def predict(self, input: TrackingInput) -> TrackingOutput:
        frames = []
        for index, frame in enumerate(input.video):
            image = convert_string_to_image(frame)
            # persist=False starts a fresh tracker for this request; persist=True continues it.
            result = self._model.track(image, persist=index > 0, tracker="bytetrack.yaml", verbose=False)[0]
            frames.append(
                TrackedFrame(
                    frame_index=index,
                    objects=[
                        TrackedObject(track_id=int(i), box=xyxy.tolist(), score=float(s), class_name=result.names[int(c)])
                        for i, xyxy, s, c in zip(result.boxes.id, result.boxes.xyxy, result.boxes.conf, result.boxes.cls)
                    ],
                )
            )
        return TrackingOutput(frames=frames)
```

The output is built the way the tracker produces it: one `TrackedFrame` per frame, holding one
`TrackedObject` per track alive in that frame. The full implementation in
[`src/pixano_yolo/tracker.py`](src/pixano_yolo/tracker.py) also accepts a single video file, filters by
class name, and rejects a prompted request.

Deploy it and run the test script (set `num_gpus=0` in `config_tracking.py` on a CPU-only machine):

```bash
uv run --project examples/yolo pixano-inference --config examples/yolo/config_tracking.py

# In another terminal: track the people of the sample clip and save annotated frames
uv run --project examples/yolo python examples/yolo/test_tracking.py \
    --server-url http://127.0.0.1:7463 \
    --classes person \
    --output tracking_result.png
```

Example output:

```
Tracking
Status: SUCCESS
Processing time: 0.971s
Frames: 30

Frame 0:
  track #1 class=person score=0.889 box=[309.0, 2.8, 517.6, 403.9]
  track #2 class=person score=0.849 box=[143.1, 132.2, 293.4, 408.9]

Tracks: 2
  #1 class=person frames 0-29 (30 of 30)
  #2 class=person frames 0-18 (19 of 30)
```

Track #2 ends at frame 18, when that child is hidden behind the other one: a track lives only
while the detector sees its object (ByteTrack keeps a lost track for `track_buffer` frames and
resumes it if the object comes back in time).

With the Python client, send the frames and read the result by frame or by track:

```python
from pixano_inference.client import PixanoInferenceClient
from pixano_inference.schemas import TrackingRequestV1

client = PixanoInferenceClient.connect("http://localhost:7463")

request = TrackingRequestV1(
    model="yolo-bytetrack",
    video=frames,            # frames as URLs, base64 data URIs or paths; or one video
    classes=["person"],      # optional: class names to keep
    box_threshold=None,      # optional: detector confidence floor (the tracker's default is 0.1)
)
result = await client.tracking(request)

for frame in result.data.frames:
    for tracked in frame.objects:
        print(frame.frame_index, tracked.track_id, tracked.class_name, tracked.score, tracked.box)

for track_id, track in result.data.tracks().items():   # the same result as trajectories
    print(track_id, [frame_index for frame_index, _ in track])
```

The endpoint is `POST /v1/inference/tracking`. Request body:

```json
{
  "model": "yolo-bytetrack",
  "video": ["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."],
  "classes": ["person"]
}
```

Response (`box` is `[x1, y1, x2, y2]` in pixels of the frame):

```json
{
  "status": "SUCCESS",
  "data": {
    "frames": [
      {
        "frameIndex": 0,
        "objects": [
          {"trackId": 1, "box": [309.0, 2.8, 517.6, 403.9], "score": 0.889, "class": "person", "mask": null},
          {"trackId": 2, "box": [143.1, 132.2, 293.4, 408.9], "score": 0.849, "class": "person", "mask": null}
        ]
      }
    ]
  }
}
```

Notes:

- Track IDs restart at 1 for every request; a request is one video.
- Track mode keeps the detector confidence low (0.1) on purpose: ByteTrack also associates the
  low-score boxes, which is what keeps a partly hidden object on its track. `box_threshold` raises
  that floor.
- `model_params["tracker"]` selects the Ultralytics tracker config (`bytetrack.yaml` by default).

## Available Base Classes

You can extend any of these base classes depending on your model's capability:

| Base Class | Capability | Input/Output | Use Case |
|---|---|---|---|
| `DetectionModel` | `detection` | `DetectionInput` / `DetectionOutput` | Object detection, instance segmentation |
| `SegmentationModel` | `segmentation` | `SegmentationInput` / `SegmentationOutput` | Interactive/prompt-based segmentation |
| `TrackingModel` | `tracking` | `TrackingInput` / `TrackingOutput` | Video object tracking |
| `VLMModel` | `vlm` | `VLMInput` / `VLMOutput` | Vision-language models |

All are in `pixano_inference.models`.

## Troubleshooting

**Server hangs on startup (no "Uvicorn running" message)**

The server deploys models synchronously before starting. If `num_gpus=1` but no GPU is available, the Ray actor cannot be scheduled and the server hangs. Fix: set `num_gpus=0` in `DeploymentConfig` for CPU-only machines.

**`ModuleNotFoundError: No module named 'ultralytics'`**

The plugin package pulls in ultralytics. Sync its environment (`uv sync --project examples/yolo`) and
start the server from it (`uv run --project examples/yolo pixano-inference ...`).

**`Unknown model_class 'YOLOModel'`**

The plugin package is not installed in the server's environment. Start the server from the
package's own environment (`uv run --project examples/yolo pixano-inference ...`), or
`pip install` your published package where the server runs, so its entry point is discovered.
