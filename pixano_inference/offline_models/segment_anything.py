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

import cv2
import pyarrow as pa
import shortuuid
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from pixano.core import arrow_types
from pixano.inference import OfflineModel
from pixano.transforms import mask_to_rle, normalize


class SAM(OfflineModel):
    """Segment Anything Model (SAM)

    Attributes:
        name (str): Model name
        id (str): Model ID
        device (str): Model GPU or CPU device (e.g. "cuda", "cpu")
        source (str): Model source
        info (str): Additional model info
        model (torch.nn.Module): SAM model
        model (torch.nn.Module): PyTorch model
    """

    def __init__(
        self, checkpoint_path: Path, size: str = "h", id: str = "", device: str = "cuda"
    ) -> None:
        """Initialize model

        Args:
            checkpoint_path (Path): Model checkpoint path.
            size (str, optional): Model size ("b", "l", "h"). Defaults to "h".
            id (str, optional): Previously used ID, generate new ID if "". Defaults to "".
            device (str, optional): Model GPU or CPU device (e.g. "cuda", "cpu"). Defaults to "cuda".
        """

        super().__init__(
            name=f"SAM_ViT_{size.upper()}",
            id=id,
            device=device,
            source="GitHub",
            info=f"Segment Anything Model (SAM), ViT-{size.upper()} Backbone",
        )

        # Model
        self.sam = sam_model_registry[f"vit_{size}"](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.model = SamAutomaticMaskGenerator(self.sam)

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

        # Iterate manually
        for x in range(batch.num_rows):
            # Preprocess image
            img = cv2.imread(str(media_dir / batch[view][x].as_py()._uri))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Inference
            with torch.no_grad():
                output = self.model.generate(img)

            # Process model outputs
            h, w = img.shape[:2]
            objects.append(
                [
                    arrow_types.ObjectAnnotation(
                        id=shortuuid.uuid(),
                        view_id=view,
                        bbox=normalize(output[i]["bbox"], w, h),
                        bbox_confidence=float(output[i]["predicted_iou"]),
                        bbox_source=self.id,
                        mask=mask_to_rle(output[i]["segmentation"]),
                        mask_source=self.id,
                        category_id=0,
                        category_name="N/A",
                    )
                    for i in range(len(output))
                    if output[i]["predicted_iou"] > threshold
                ]
            )
        return objects
