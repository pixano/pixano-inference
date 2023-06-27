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
from pixano.core import arrow_types
from pixano.inference import InferenceModel
from pixano.transforms import mask_to_rle, voc_names


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


class DeepLabV3(InferenceModel):
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

        objects = []

        # PyTorch Transforms don't support different-sized image batches, so iterate manually
        for x in range(batch.num_rows):
            # Preprocess image
            im = batch[view][x].as_py(uri_prefix).as_pillow()
            im_tensor = self.transforms(im).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(im_tensor)["out"][0]

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
