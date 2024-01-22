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

import pyarrow as pa
import shortuuid
import torch
from pixano.core import BBox, Image
from pixano.models import InferenceModel
from pixano.utils import coco_ids_80to91, coco_names_91


class YOLOv5(InferenceModel):
    """PyTorch Hub YOLOv5 Model

    Attributes:
        name (str): Model name
        model_id (str): Model ID
        device (str): Model GPU or CPU device
        description (str): Model description
        model (torch.nn.Module): PyTorch model
    """

    def __init__(
        self,
        size: str = "s",
        model_id: str = "",
        device: str = "cuda",
    ) -> None:
        """Initialize model

        Args:
            size (str, optional): Model size ("n", "s", "m", "x"). Defaults to "s".
            model_id (str, optional): Previously used ID, generate new ID if "". Defaults to "".
            device (str, optional): Model GPU or CPU device (e.g. "cuda", "cpu"). Defaults to "cuda".
        """

        super().__init__(
            name=f"YOLOv5{size}",
            model_id=model_id,
            device=device,
            description=f"From PyTorch Hub. YOLOv5 model, {size.upper()} backbone.",
        )

        # Model
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            model=f"yolov5{size}",
            pretrained=True,
        )
        self.model.to(self.device)

    def preannotate(
        self,
        batch: pa.RecordBatch,
        views: list[str],
        uri_prefix: str,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Inference pre-annotation for a batch

        Args:
            batch (pa.RecordBatch): Input batch
            views (list[str]): Dataset views
            uri_prefix (str): URI prefix for media files
            threshold (float, optional): Confidence threshold. Defaults to 0.0.

        Returns:
            list[dict]: Processed rows
        """

        rows = []

        for view in views:
            # Preprocess image batch
            im_batch = []
            for x in range(batch.num_rows):
                im = Image.from_dict(batch[view][x].as_py())
                im.uri_prefix = uri_prefix
                im_batch.append(im.as_pillow())

            # Inference
            outputs = self.model(im_batch)

            # Process model outputs
            for x, img, img_output in zip(
                range(batch.num_rows), im_batch, outputs.xyxy
            ):
                w, h = img.size
                rows.extend(
                    [
                        {
                            "id": shortuuid.uuid(),
                            "item_id": batch["id"][x].as_py(),
                            "view_id": view,
                            "bbox": BBox.from_xyxy(
                                [coord.item() for coord in pred[0:4]],
                                confidence=pred[4].item(),
                            )
                            .normalize(h, w)
                            .to_dict(),
                            "category": coco_names_91(coco_ids_80to91(pred[5] + 1)),
                        }
                        for pred in img_output
                        if pred[4] > threshold
                    ]
                )

        return rows
