<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# YOLO detector and tracker

Two custom models for Pixano Inference, built on [Ultralytics YOLO](https://docs.ultralytics.com)
and shipped as one installable package. Use it as a template for your own detector or tracker.

| Model                | What it does                                                  | Endpoint                       | Source                                                     |
| -------------------- | ------------------------------------------------------------- | ------------------------------ | ---------------------------------------------------------- |
| `YOLOModel`          | Detects the objects of an image                               | `POST /v1/inference/detection` | [`src/pixano_yolo/model.py`](src/pixano_yolo/model.py)     |
| `YOLOByteTrackModel` | Detects and tracks every object of a video (YOLO + ByteTrack) | `POST /v1/inference/tracking`  | [`src/pixano_yolo/tracker.py`](src/pixano_yolo/tracker.py) |

> **Licence note:** Ultralytics is AGPL-3.0. Deploying it as a network service carries
> source-disclosure obligations (AGPL section 13). Review the licence before production use.

## Run it

All commands are run from the repository root.

```bash
# 1. Install the package in its own environment (the Pixano Inference server + ultralytics)
uv sync --project examples/yolo

# 2. Start a server with the detector...
uv run --project examples/yolo pixano-inference --config examples/yolo/config.py
#    ...or with the tracker
uv run --project examples/yolo pixano-inference --config examples/yolo/config_tracking.py

# 3. In another terminal, call it
uv run --project examples/yolo python examples/yolo/test_yolo.py --image path/to/image.jpg
uv run --project examples/yolo python examples/yolo/test_tracking.py --classes person
```

The server is ready when it prints `Uvicorn running on http://127.0.0.1:7463`. The first start
downloads the YOLO weights into the current directory.

Both configs ask for one GPU per replica; on a machine without a GPU, set `num_gpus=0` in the
config. To serve both models from one server, put the two `ModelConfig` entries in one file.

## The detector: `YOLOModel`

A detector extends `DetectionModel`: it receives a `DetectionInput` (an image and a confidence
threshold) and returns a `DetectionOutput` (boxes, scores and class names).

```python
# src/pixano_yolo/model.py (simplified)
@register_model("YOLOModel")
class YOLOModel(DetectionModel):

    def load_model(self) -> None:
        """Called once when the replica starts: load the weights here."""
        from ultralytics import YOLO

        self._model = YOLO(self._config.model_params["path"])

    def predict(self, input: DetectionInput) -> DetectionOutput:
        """Called for each request."""
        from pixano_inference.utils.media import convert_string_to_image

        image = convert_string_to_image(input.image)  # URL, base64 or path -> PIL image
        result = self._model.predict(image, conf=input.box_threshold)[0]

        return DetectionOutput(
            boxes=[[int(round(c)) for c in box] for box in result.boxes.xyxy.tolist()],  # [x1, y1, x2, y2]
            scores=result.boxes.conf.tolist(),
            classes=[result.names[int(c)] for c in result.boxes.cls.tolist()],
        )
```

`test_yolo.py` sends one image and prints the detections:

```
Status: SUCCESS
Processing time: 0.683s
Detections: 5
  [0] class=bus, score=0.923, box=[6, 230, 802, 738]
  [1] class=person, score=0.923, box=[668, 395, 809, 880]
  [2] class=person, score=0.898, box=[48, 401, 247, 903]
  [3] class=person, score=0.846, box=[221, 406, 345, 861]
  [4] class=person, score=0.833, box=[0, 552, 78, 876]
```

From Python:

```python
import base64
from pathlib import Path

from pixano_inference.client import SyncPixanoInferenceClient
from pixano_inference.schemas import DetectionRequest


def data_uri(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


client = SyncPixanoInferenceClient("http://localhost:7463")
result = client.detection(DetectionRequest(model="yolo26s", image=data_uri("bus.jpg"), box_threshold=0.3))

for box, score, name in zip(result.data.boxes, result.data.scores, result.data.classes):
    print(f"{name}: {score:.3f} {box}")
```

Over HTTP, `POST /v1/inference/detection` (`image` is an http(s) URL or a base64 data URI):

```json
{
  "model": "yolo26s",
  "image": "data:image/jpeg;base64,...",
  "boxThreshold": 0.5
}
```

```json
{
  "id": "ray-yolo26s-1d9cd9ce2b58",
  "status": "SUCCESS",
  "processingTime": 0.062,
  "data": {
    "boxes": [
      [6, 230, 802, 738],
      [668, 395, 809, 880]
    ],
    "scores": [0.923, 0.923],
    "classes": ["bus", "person"],
    "masks": null
  }
}
```

## The tracker: `YOLOByteTrackModel`

The tracker uses the [track mode](https://docs.ultralytics.com/modes/track) of Ultralytics: YOLO
detects the objects of each frame and [ByteTrack](https://arxiv.org/abs/2110.06864) links the
detections into tracks. The request carries **no prompt**, only the video: the model decides how
many tracks there are and numbers them.

A tracker extends `TrackingModel`: it receives a `TrackingInput` and returns a `TrackingOutput`
with one `TrackedFrame` per frame, each listing the `TrackedObject`s seen in that frame.

```python
# src/pixano_yolo/tracker.py (simplified)
@register_model("YOLOByteTrackModel")
class YOLOByteTrackModel(TrackingModel):

    def predict(self, input: TrackingInput) -> TrackingOutput:
        frames = []
        for index, frame in enumerate(input.video):
            image = convert_string_to_image(frame)
            # persist=False starts a fresh tracker for this request; persist=True continues it.
            result = self._model.track(image, persist=index > 0, tracker="bytetrack.yaml", verbose=False)[0]

            objects = []
            if result.boxes.id is not None:  # None when nothing is tracked in the frame
                for track_id, box, score, cls in zip(
                    result.boxes.id.tolist(),
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                ):
                    objects.append(
                        TrackedObject(track_id=int(track_id), box=box, score=score, class_name=result.names[int(cls)])
                    )
            frames.append(TrackedFrame(frame_index=index, objects=objects))

        return TrackingOutput(frames=frames)
```

The full file also accepts a single video file, filters by class name, and rejects a request that
carries prompts.

`test_tracking.py` sends the first frames of a sample clip (two children on a bed), prints the
tracks and saves a few frames with the boxes drawn (`tracking_result.png`):

```
Status: SUCCESS
Processing time: 1.524s
Frames: 30

Frame 0:
  track #1 class=person score=0.889 box=[309.0, 2.8, 517.6, 403.9]
  track #2 class=person score=0.849 box=[143.1, 132.2, 293.4, 408.9]

Tracks: 2
  #1 class=person frames 0-29 (30 of 30)
  #2 class=person frames 0-18 (19 of 30)
```

Track #2 ends at frame 18, when that child is hidden behind the other one.

From Python:

```python
import base64
from pathlib import Path

from pixano_inference.client import SyncPixanoInferenceClient
from pixano_inference.schemas import TrackingRequestV1


def data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


frames = sorted(Path("docs/assets/examples/sam2/bedroom").glob("*.jpg"))[:30]

client = SyncPixanoInferenceClient("http://localhost:7463")
result = client.tracking(
    TrackingRequestV1(
        model="yolo-bytetrack",
        video=[data_uri(frame) for frame in frames],
        classes=["person"],  # optional: the class names to keep
    )
)

# Frame by frame, as the model returns it
for frame in result.data.frames:
    for tracked in frame.objects:
        print(frame.frame_index, tracked.track_id, tracked.class_name, f"{tracked.score:.2f}", tracked.box)

# The same result, track by track
for track_id, track in result.data.tracks().items():
    print(f"track {track_id}: frames {[frame_index for frame_index, _ in track]}")
```

Over HTTP, `POST /v1/inference/tracking` (`video` is a list of frames, or a single video):

```json
{
  "model": "yolo-bytetrack",
  "video": ["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."],
  "classes": ["person"]
}
```

```json
{
  "status": "SUCCESS",
  "data": {
    "frames": [
      {
        "frameIndex": 0,
        "objects": [
          {
            "trackId": 1,
            "box": [309.0, 2.8, 517.6, 403.9],
            "score": 0.889,
            "class": "person",
            "mask": null
          },
          {
            "trackId": 2,
            "box": [143.1, 132.2, 293.4, 408.9],
            "score": 0.849,
            "class": "person",
            "mask": null
          }
        ]
      }
    ]
  }
}
```

Good to know:

- A box is `[x1, y1, x2, y2]` in pixels of the frame.
- Track IDs restart at 1 for every request: a request is one video.
- A track lasts while the detector sees its object. ByteTrack keeps a lost track for a few frames
  and resumes it if the object comes back in time; otherwise the object gets a new ID.
- Track mode keeps the detector confidence low (0.1) on purpose: ByteTrack also uses the low-score
  boxes, which keeps a partly hidden object on its track. `boxThreshold` in the request raises
  that floor.
- `model_params["tracker"]` selects the Ultralytics tracker config (`bytetrack.yaml` by default).

## How the package plugs into the server

Three pieces make a model available, and they are the same for both models.

**1. Register the class.** `@register_model("YOLOModel")` gives the model the name used in configs.

**2. Declare an entry point** in [`pyproject.toml`](pyproject.toml). At startup the server imports
every module listed in the `pixano_inference.models` group, which runs the decorators:

```toml
[project]
dependencies = ["pixano-inference[server] >= 0.7.1, < 0.8.0", "torch >= 2.3.0, < 3.0.0", "ultralytics", "lap >= 0.5.12"]

[project.entry-points."pixano_inference.models"]
yolo_detector = "pixano_yolo.model"
yolo_bytetrack = "pixano_yolo.tracker"
```

**3. Reference the model by name** in a config file. No import is needed:

```python
# config.py
from pixano_inference.configs import DeploymentConfig, ModelConfig

models = [
    ModelConfig(
        name="yolo26s",                       # the name clients use in their requests
        model_class="YOLOModel",              # the registered class
        model_params={"path": "yolo26s.pt"},  # passed to the model as self._config.model_params
        deployment=DeploymentConfig(num_gpus=1, num_cpus=1, min_replicas=1, max_replicas=2),
    ),
]
```

To write a model for another capability (segmentation, vision-language, embeddings, ...), see the
[custom models guide](../../docs/ray_serve/custom_models.md). For a starting point without any ML
framework, see [`../numpy_detector`](../numpy_detector).

## Files

```
examples/yolo/
    pyproject.toml            # dependencies and the pixano_inference.models entry points
    src/pixano_yolo/
        model.py              # YOLOModel
        tracker.py            # YOLOByteTrackModel
    config.py                 # deploys YOLOModel as "yolo26s"
    config_tracking.py        # deploys YOLOByteTrackModel as "yolo-bytetrack"
    test_yolo.py              # client script for the detector
    test_tracking.py          # client script for the tracker
    tests/                    # unit tests (no weights needed)
```

## Troubleshooting

**`Unknown model_class 'YOLOModel'`** or **`No module named 'ultralytics'`**

The server is not running from the package's environment. Start it with
`uv run --project examples/yolo pixano-inference ...`, or install your package where the server runs.

**`Model 'yolo26s' not found`**

The name in the request must be the `name` of a `ModelConfig` in the config the server was started
with: `yolo26s` in `config.py`, `yolo-bytetrack` in `config_tracking.py`.

**The tracker answers with a 500 error**

It refuses a request that names objects or carries prompts, and a class name the detector does not
know. The reason is in the server log.
