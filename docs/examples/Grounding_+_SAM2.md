<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Grounding DINO + SAM2 Video

## Define visualization utilities

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


np.random.seed(3)


def show_image(path):
    fig = plt.figure(figsize=(9, 6))
    plt.imshow(Image.open(path))


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
```

## Python config

Deploy both Grounding DINO and SAM2 Video in a single config:

```python
from pixano_inference.configs import (
    DeploymentConfig,
    GroundingDINOParams,
    ModelConfig,
    Sam2VideoParams,
)


models = [
    ModelConfig(
        name="grounding-dino",
        model_class="GroundingDINOModel",
        model_params=GroundingDINOParams(path="IDEA-Research/grounding-dino-tiny"),
        deployment=DeploymentConfig(num_gpus=1),
    ),
    ModelConfig(
        name="sam2-video",
        model_class="Sam2VideoModel",
        model_params=Sam2VideoParams(path="facebook/sam2-hiera-tiny", torch_dtype="bfloat16", propagate=True),
        deployment=DeploymentConfig(num_gpus=1),
    ),
]
```

## Start the server

```bash
pixano-inference --config models.py
```

## Connect the client

```python
from pixano_inference.client import PixanoInferenceClient


client = PixanoInferenceClient.connect(url="http://localhost:7463")
```

## Load frames and show the first one

```python
from pathlib import Path

frames = sorted([str(f) for f in Path("./docs/assets/examples/sam2/bedroom").glob("**/*") if f.is_file()])
first_frame = frames[0]
show_image(first_frame)
```

## Call Grounding DINO on the first frame

```python
import asyncio

from pixano_inference.schemas import DetectionRequest


async def run_detection():
    request = DetectionRequest(
        model="grounding-dino",
        image=first_frame,
        classes=["bed", "kid"],
        box_threshold=0.3,
        text_threshold=0.2,
    )
    response = await client.detection(request)
    return response


detection_response = asyncio.run(run_detection())
boxes = detection_response.data.boxes
scores = detection_response.data.scores
classes = detection_response.data.classes

show_image(first_frame)
for box in boxes:
    show_box(box, plt.gca())
```

## Call SAM2 Video with detected boxes

```python
from pixano_inference.schemas import TrackingRequest


async def run_video_segmentation():
    obj_ids = list(range(len(boxes)))
    frame_indexes = [0] * len(boxes)

    request = TrackingRequest(
        model="sam2-video",
        video=frames,
        frame_indexes=frame_indexes,
        objects_ids=obj_ids,
        boxes=boxes,
    )
    response = await client.tracking(request)
    return response


masks_response = asyncio.run(run_video_segmentation())
```

## Display the result

```python
vis_frame_stride = 4
plt.close("all")
for out_frame_idx in range(0, len(frames), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(frames[out_frame_idx]))
    for out_obj_id, out_mask, frame_indx in zip(
        masks_response.data.objects_ids, masks_response.data.masks, masks_response.data.frame_indexes
    ):
        if frame_indx != out_frame_idx:
            continue
        out_mask = out_mask.to_mask()
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
```
