<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# SAM2 Video Segmentation

## Define visualization utilities

```python
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(3)


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

Deploy SAM2 for video segmentation:

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig, Sam2VideoParams


models = [
    ModelConfig(
        name="sam2-video",
        task="video_mask_generation",
        model_class="Sam2VideoModel",
        model_params=Sam2VideoParams(path="facebook/sam2-hiera-tiny", torch_dtype="bfloat16", propagate=True),
        deployment=DeploymentConfig(num_gpus=1),
    )
]
```

## Start the server

```bash
pixano-inference --config models.py
```

## Connect the client and run inference

```python
import asyncio
from pathlib import Path

from pixano_inference.client import PixanoInferenceClient
from pixano_inference.schemas import TrackingRequest


async def main():
    client = PixanoInferenceClient.connect(url="http://localhost:7463")

    frames = sorted([str(f) for f in Path("./docs/assets/examples/sam2/bedroom").glob("**/*") if f.is_file()])

    request = TrackingRequest(
        model="sam2-video",
        video=frames,
        objects_ids=[0, 2],
        frame_indexes=[0, 0],
        points=[[[210, 350]], [[400, 500]]],
        labels=[[1], [1]],
    )
    response = await client.tracking(request)

    print(f"Processing time: {response.processing_time:.3f}s")
    return response


response = asyncio.run(main())
```

## Display the result

```python
import os

from PIL import Image


video_dir = Path("./docs/assets/examples/sam2/bedroom/")

frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

vis_frame_stride = 4
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask, frame_indx in zip(
        response.data.objects_ids, response.data.masks, response.data.frame_indexes
    ):
        if frame_indx != out_frame_idx:
            continue
        out_mask = out_mask.to_mask()
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
```
