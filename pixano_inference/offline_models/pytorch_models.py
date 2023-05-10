# @Copyright: CEA-LIST/DIASI/SIALV/LVA (2023)
# @Author: CEA-LIST/DIASI/SIALV/LVA <pixano@cea.fr>
# @License: CECILL-C
#
# This software is a collaborative computer program whose purpose is to
# generate and explore labeled data for computer vision applications.
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
#
# http://www.cecill.info

from pathlib import Path

import numpy as np
import pyarrow as pa
import shortuuid
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

from pixano.core import arrow_types
from pixano.inference import OfflineModel
from pixano.transforms import (
    coco_names_80,
    coco_names_91,
    mask_to_rle,
    normalize,
    voc_names,
    xyxy_to_xywh,
)


def unmold_mask(mask: torch.Tensor, threshold: float = 0.5):
    """Convert mask from torch.Tensor to np.array, squeeze a dimension if needed, and treshold values

    Args:
        mask (torch.Tensor): Mask (1, W, H)
        threshold (float, optional): Confidence threshold. Defaults to 0.5.

    Returns:
        np.array: Mask (W, H)
    """

    # Detach and convert to NumPy
    mask = mask.cpu().numpy()

    # Squeeze dimension if needed
    if 1 in mask.shape:
        mask = mask.squeeze(mask.shape.index(1))

    # Threshold values
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    return mask


class DeepLabV3(OfflineModel):
    """PyTorch Hub DeepLabV3 Model

    Attributes:
        name (str): Model name
        id (str): Model ID
        device (str): Model GPU or CPU device
        source (str): Model source
        info (str): Additional model info
        model (torch.nn.Module): PyTorch model
        transforms (torch.nn.Module): PyTorch preprocessing transforms
    """

    def __init__(self, id: str = "", device: str = "cuda") -> None:
        """Initialize model

        Args:
            id (str, optional): Previously used ID, generate new ID if "". Defaults to "".
            device (str, optional): Model GPU or CPU device (e.g. "cuda", "cpu"). Defaults to "cuda".
        """

        super().__init__(
            name="DeepLabV3",
            id=id,
            device=device,
            source="PyTorch Hub",
            info="DeepLabV3, ResNet-50 Backbone",
        )

        # Model
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
        )
        self.model.eval()
        self.model.to(self.device)

        # Transforms
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(
        self, batch: pa.RecordBatch, view: str, media_dir: Path, threshold: float = 0.0
    ) -> list[list[arrow_types.ObjectAnnotation]]:
        """Returns model inferences for a given batch of images

        Args:
            batch (pa.RecordBatch): Input batch
            view (str): Dataset view
            media_dir (Path): Media location
            threshold (float, optional): Confidence threshold. Defaults to 0.0.

        Returns:
            list[list[arrow_types.ObjectAnnotation]]: Model inferences as lists of ObjectAnnotation
        """

        objects = []

        # PyTorch Transforms don't support image batches, so iterate manually
        for x in range(batch.num_rows):
            # Preprocess image
            img = Image.open(media_dir / batch[view][x].as_py()._uri).convert("RGB")
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(img_tensor)["out"][0]

            # Process model outputs
            sem_mask = output.argmax(0)
            labels = torch.unique(sem_mask)[1:]
            masks = sem_mask == labels[:, None, None]

            objects.append(
                [
                    arrow_types.ObjectAnnotation(
                        id=shortuuid.uuid(),
                        view_id=view,
                        mask=mask_to_rle(unmold_mask(mask)),
                        mask_source=self.id,
                        category_id=int(label),
                        category_name=voc_names(label),
                    )
                    for label, mask in zip(labels, masks)
                ]
            )
        return objects


class MaskRCNNv2(OfflineModel):
    """PyTorch Hub MaskRCNNv2 Model

    Attributes:
        name (str): Model name
        id (str): Model ID
        device (str): Model GPU or CPU device
        source (str): Model source
        info (str): Additional model info
        model (torch.nn.Module): PyTorch model
        transforms (torch.nn.Module): PyTorch preprocessing transforms
    """

    def __init__(self, id: str = "", device: str = "cuda") -> None:
        """Initialize model

        Args:
            id (str, optional): Previously used ID, generate new ID if "". Defaults to "".
            device (str, optional): Model GPU or CPU device (e.g. "cuda", "cpu"). Defaults to "cuda".
        """

        super().__init__(
            name="MaskRCNNv2",
            id=id,
            device=device,
            source="PyTorch Hub",
            info="MaskRCNN, ResNet-50-FPN v2 Backbone, COCO_V1 Weights",
        )

        # Model
        self.model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        self.model.eval()
        self.model.to(self.device)

        # Transforms
        self.transforms = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()

    def __call__(
        self, batch: pa.RecordBatch, view: str, media_dir: Path, threshold: float = 0.0
    ) -> list[list[arrow_types.ObjectAnnotation]]:
        """Returns model inferences for a given batch of images

        Args:
            batch (pa.RecordBatch): Input batch
            view (str): Dataset view
            media_dir (Path): Media location
            threshold (float, optional): Confidence threshold. Defaults to 0.0.

        Returns:
            list[list[arrow_types.ObjectAnnotation]]: Model inferences as lists of ObjectAnnotation
        """

        objects = []

        # PyTorch Transforms don't support image batches, so iterate manually
        for x in range(batch.num_rows):
            # Preprocess image
            img = Image.open(media_dir / batch[view][x].as_py()._uri).convert("RGB")
            img_tensor = self.transforms(img).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(img_tensor)[0]

            # Process model outputs
            w, h = img.size
            objects.append(
                [
                    arrow_types.ObjectAnnotation(
                        id=shortuuid.uuid(),
                        view_id=view,
                        bbox=normalize(xyxy_to_xywh(output["boxes"][i]), w, h),
                        bbox_confidence=float(output["scores"][i]),
                        bbox_source=self.id,
                        mask=mask_to_rle(unmold_mask(output["masks"][i])),
                        mask_source=self.id,
                        category_id=int(output["labels"][i]),
                        category_name=coco_names_91(output["labels"][i]),
                    )
                    for i in range(len(output["scores"]))
                    if output["scores"][i] > threshold
                ]
            )
        return objects


class YOLO(OfflineModel):
    """PyTorch Hub YOLOv5 Model

    Attributes:
        name (str): Model name
        id (str): Model ID
        device (str): Model GPU or CPU device
        source (str): Model source
        info (str): Additional model info
        model (torch.nn.Module): PyTorch model
    """

    def __init__(self, size: str = "s", id: str = "", device: str = "cuda") -> None:
        """Initialize model

        Args:
            size (str, optional): Model size ("n", "s", "m", "x"). Defaults to "s".
            id (str, optional): Previously used ID, generate new ID if "". Defaults to "".
            device (str, optional): Model GPU or CPU device (e.g. "cuda", "cpu"). Defaults to "cuda".
        """

        super().__init__(
            name=f"YOLOv5{size}",
            id=id,
            device=device,
            source="PyTorch Hub",
            info=f"YOLOv5 model, {size.upper()} backbone",
        )

        # Model
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            model=f"yolov5{size}",
            pretrained=True,
        )
        self.model.to(self.device)

    def __call__(
        self, batch: pa.RecordBatch, view: str, media_dir: Path, threshold: float = 0.0
    ) -> list[list[arrow_types.ObjectAnnotation]]:
        """Returns model inferences for a given batch of images

        Args:
            batch (pa.RecordBatch): Input batch
            view (str): Dataset view
            media_dir (Path): Media location
            threshold (float, optional): Confidence threshold. Defaults to 0.0.

        Returns:
            list[list[arrow_types.ObjectAnnotation]]: Model inferences as lists of ObjectAnnotation
        """

        # Preprocess images
        imgs = [
            Image.open(media_dir / batch[view][x].as_py()._uri).convert("RGB")
            for x in range(batch.num_rows)
        ]

        # Inference
        outputs = self.model(imgs)

        # Process model outputs
        objects = []
        for img, img_output in zip(imgs, outputs.xyxy):
            w, h = img.size
            objects.append(
                [
                    arrow_types.ObjectAnnotation(
                        id=shortuuid.uuid(),
                        view_id=view,
                        bbox=normalize(xyxy_to_xywh(pred[0:4]), w, h),
                        bbox_confidence=float(pred[4]),
                        bbox_source=self.id,
                        category_id=int(pred[5] + 1),
                        category_name=coco_names_80(pred[5] + 1),
                    )
                    for pred in img_output
                    if pred[4] > threshold
                ]
            )

        return objects
