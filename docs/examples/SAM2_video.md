<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# SAM2 Video

## Define the visualization utilities (from SAM2)

```python
import matplotlib.pyplot as plt
import numpy as np
import torch


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


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

```

## Instantiate the model

```python
import torch

from pixano_inference.providers.sam2 import Sam2Provider


provider = Sam2Provider()
model = provider.load_model(
    name="sam",
    task="image_mask_generation",
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    path="facebook/sam2-hiera-tiny",
)
```

## Call the model

```python
from pathlib import Path


obj_ids = [0, 2]
frame_indexes = [0, 0]
points = [[[210, 350]], [[400, 500]]]
labels = [[1], [1]]
output = model.video_mask_generation(
    sorted([f for f in Path("./docs/assets/examples/sam2/bedroom").glob("**/*") if f.is_file()]),
    objects_ids=obj_ids,
    frame_indexes=frame_indexes,
    points=points,
    labels=labels,
    propagate=True,
)
```

## Display the result

```python
import os

from PIL import Image


video_dir = Path("./docs/assets/examples/sam2/bedroom/")

# scan all the JPEG frame names in this directory
frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
```
