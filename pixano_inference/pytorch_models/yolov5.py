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

import pyarrow as pa
import shortuuid
import torch
from PIL import Image
from pixano.core import arrow_types
from pixano.inference import InferenceModel
from pixano.transforms import coco_ids_80to91, coco_names_91, normalize, xyxy_to_xywh


class YOLOv5(InferenceModel):
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

    def inference_batch(
        self, batch: pa.RecordBatch, view: str, uri_prefix: str, threshold: float = 0.0
    ) -> list[list[arrow_types.ObjectAnnotation]]:
        """Inference preannotation for a batch

        Args:
            batch (pa.RecordBatch): Input batch
            view (str): Dataset view
            uri_prefix (str): URI prefix for media files
            threshold (float, optional): Confidence threshold. Defaults to 0.0.

        Returns:
            list[list[arrow_types.ObjectAnnotation]]: Model inferences as lists of ObjectAnnotation
        """

        # Preprocess image batch
        im_batch = [
            batch[view][x].as_py(uri_prefix).as_pillow() for x in range(batch.num_rows)
        ]

        # Inference
        outputs = self.model(im_batch)

        # Process model outputs
        objects = []
        for img, img_output in zip(im_batch, outputs.xyxy):
            w, h = img.size
            objects.append(
                [
                    arrow_types.ObjectAnnotation(
                        id=shortuuid.uuid(),
                        view_id=view,
                        bbox=normalize(xyxy_to_xywh(pred[0:4]), h, w),
                        bbox_confidence=float(pred[4]),
                        bbox_source=self.id,
                        category_id=coco_ids_80to91(pred[5] + 1),
                        category_name=coco_names_91(coco_ids_80to91(pred[5] + 1)),
                    )
                    for pred in img_output
                    if pred[4] > threshold
                ]
            )

        return objects
