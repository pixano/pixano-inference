<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Grounding DINO + SAM2 Video

## Define the visualization utilities (from SAM2)

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


np.random.seed(3)


def show_image(path):
    fig = plt.figure(figsize=(9, 6))
    fig.imshow(Image.open(path)) # OpenCV imread uses BGR by default but matplotlib uses RGB so we need to change it manually for correct visualization of the image

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

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

```

## Instantiate the models

```python
from pathlib import Path

from pixano_inference.providers import Sam2Provider, TransformersProvider


sam2_provider = Sam2Provider()
sam2 = sam2_provider.load_model(
    "sam2",
    "video_mask_generation",
    torch.device("cuda") if torch.cuda.is_available() else "cpu",
    "facebook/sam2-hiera-tiny",
)
transformers_provider = TransformersProvider()
grounding_dino = transformers_provider.load_model(
    "grounding_dino",
    "image_zero_shot_detection",
    torch.device("cuda") if torch.cuda.is_available() else "cpu",
    "IDEA-Research/grounding-dino-tiny",
)
```

## Load frames and show the first one

```python
frames = sorted([f for f in Path("./docs/assets/examples/sam2/bedroom").glob("**/*") if f.is_file()])
first_frame = frames[0]
show_image(first_frame)
```

## Call Grounding DINO on the first frame

```python
from pixano_inference.pydantic import ImageZeroShotDetectionOutput
from pixano_inference.utils.media import convert_string_to_image


image_zero_shot_detection_out: ImageZeroShotDetectionOutput = grounding_dino.image_zero_shot_detection(
    image=convert_string_to_image(first_frame), classes=["bed", "kid"], box_threshold=0.3, text_threshold=0.2
)
boxes = image_zero_shot_detection_out.boxes
scores = image_zero_shot_detection_out.scores
classes = image_zero_shot_detection_out.classes

show_image(first_frame)
for box in boxes:
    show_box(box, plt.gca())
```

## Call SAM2

```python
obj_ids = list(range(len(boxes)))
frame_indexes = [0] * len(boxes)
masks_output = sam2.video_mask_generation(
    video=frames,
    frame_indexes=frame_indexes,
    objects_ids=obj_ids,
    boxes=boxes,
    propagate=True,
)
```

## Display the result

```python
vis_frame_stride = 4
plt.close("all")
for out_frame_idx in range(0, len(frames), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(frames[out_frame_idx]))
    for out_obj_id, out_mask, frame_indx in zip(masks_output.objects_ids, masks_output.masks, masks_output.frame_indexes):
        if frame_indx != out_frame_idx:
            continue
        out_mask = out_mask.to_mask()
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
```
