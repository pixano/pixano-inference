# Custom Model Deployment: YOLO Detection

This tutorial walks through deploying a third-party model (Ultralytics YOLO) as a custom detection service with Pixano Inference.

## Prerequisites

Install pixano-inference with the `ultralytics` extra:

```bash
uv sync --extra ultralytics
```

Or with pip:

```bash
pip install pixano-inference[ultralytics]
```

## Project Structure

```
examples/yolo/
    model.py        # Model implementation (extends DetectionModel)
    config.py       # Deployment configuration
    test_yolo.py   # End-to-end test script
```

## Step 1: Implement the Model

Create a model class that extends one of the built-in base classes. For object detection, extend `DetectionModel`.

```python
# model.py
from pixano_inference.models.detection import DetectionInput, DetectionModel, DetectionOutput
from pixano_inference.models.registry import register_model
from pixano_inference.ray.config import ModelDeploymentConfig


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

Create a Python config file that defines a `models` list. Import your model class directly and pass it to `model_class`:

```python
# config.py
from yolo.model import YOLOModel
from pixano_inference.configs import DeploymentConfig, ModelConfig

models = [
    ModelConfig(
        name="yolo26s",
        model_class=YOLOModel,
        model_params={"path": "yolo26s.pt"},  # Passed to model via config
        deployment=DeploymentConfig(
            num_gpus=1,          # GPUs per replica (set 0 for CPU-only)
            num_cpus=1,          # CPUs per replica
            min_replicas=0,      # Scale to zero when idle
            max_replicas=2,      # Max concurrent replicas
        ),
    ),
]
```

## Step 3: Start the Server

```bash
PYTHONPATH=examples:$PYTHONPATH \
  uv run pixano-inference \
    --config examples/yolo/config.py
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:7463 (Press CTRL+C to quit)
```

## Step 4: Run the End-to-End Test

With the server running, use the included test script:

```bash
# With a real image
uv run python examples/yolo/test_yolo.py \
    --server-url http://127.0.0.1:7463 \
    --model-name yolo26s \
    --image path/to/image.jpg

# With a synthetic test image (no --image flag)
uv run python examples/yolo/test_yolo.py \
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

Install the ultralytics extra: `uv sync --extra ultralytics`

**`ModuleNotFoundError: No module named 'yolo'`**

Ensure the `examples` directory is on the Python path. Use `--module-path examples` or set `PYTHONPATH=examples:$PYTHONPATH`.
