<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# SAM2 Image Segmentation

## Define visualization utilities

```python
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(3)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
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


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()
```

## Python config

Deploy SAM2 for image segmentation:

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig, Sam2ImageParams


models = [
    ModelConfig(
        name="sam2-image",
        task="image_mask_generation",
        model_class="Sam2ImageModel",
        model_params=Sam2ImageParams(path="facebook/sam2-hiera-tiny", torch_dtype="bfloat16"),
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
import base64

from pixano_inference.client import PixanoInferenceClient
from pixano_inference.schemas import SegmentationRequest


async def main():
    client = PixanoInferenceClient.connect(url="http://localhost:7463")

    # Encode the image as base64
    with open("./docs/assets/examples/sam2/truck.jpg", "rb") as f:
        image_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

    points = [[[300, 375]], [[500, 375], [3, 3]]]
    labels = [[1], [1, 0]]

    request = SegmentationRequest(
        model="sam2-image",
        image=image_b64,
        points=points,
        labels=labels,
        multimask_output=True,
    )
    response = await client.segmentation(request)

    print(f"Processing time: {response.processing_time:.3f}s")
    print(f"Scores: {response.data.scores.to_numpy()}")

    return response


response = asyncio.run(main())
```

## Display the result

```python
from PIL import Image

image = Image.open("./docs/assets/examples/sam2/truck.jpg")
masks = response.data.masks
scores = response.data.scores.to_numpy()

for pred_points, pred_labels, pred_masks, score in zip(points, labels, masks, scores):
    np_points = np.array(pred_points)
    np_labels = np.array(pred_labels)
    show_masks(
        image,
        np.array([mask.to_mask() for mask in pred_masks]),
        score,
        point_coords=np_points,
        input_labels=np_labels,
        borders=True,
    )
plt.axis("on")
```
