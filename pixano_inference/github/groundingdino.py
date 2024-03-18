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
from pixano.core import BBox, Image
from pixano.models import InferenceModel
from torchvision.ops import box_convert

from pixano_inference.utils import attempt_import


class GroundingDINO(InferenceModel):
    """GroundingDINO Model

    Attributes:
        name (str): Model name
        model_id (str): Model ID
        device (str): Model GPU or CPU device
        description (str): Model description
        model (torch.nn.Module): PyTorch model
        checkpoint_path (Path): Model checkpoint path
        config_path (Path): Model config path
    """

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Path,
        model_id: str = "",
        device: str = "cuda",
    ) -> None:
        """Initialize model

        Args:
            checkpoint_path (Path): Model checkpoint path (download from https://github.com/IDEA-Research/GroundingDINO)
            config_path (Path): Model config path (download from https://github.com/IDEA-Research/GroundingDINO)
            model_id (str, optional): Previously used ID, generate new ID if "". Defaults to "".
            device (str, optional): Model GPU or CPU device (e.g. "cuda", "cpu"). Defaults to "cuda".
        """

        # Import GroundingDINO
        gd_inf = attempt_import(
            "groundingdino.util.inference",
            "groundingdino@git+https://github.com/IDEA-Research/GroundingDINO",
        )

        super().__init__(
            name="GroundingDINO",
            model_id=model_id,
            device=device,
            description="Fom GitHub, GroundingDINO model.",
        )

        # Model
        self.model = gd_inf.load_model(
            config_path.as_posix(),
            checkpoint_path.as_posix(),
        )
        self.model.to(self.device)

    def preannotate(
        self,
        batch: pa.RecordBatch,
        views: list[str],
        uri_prefix: str,
        threshold: float = 0.0,
        prompt: str = "",
    ) -> list[dict]:
        """Inference pre-annotation for a batch

        Args:
            batch (pa.RecordBatch): Input batch
            views (list[str]): Dataset views
            uri_prefix (str): URI prefix for media files
            threshold (float, optional): Confidence threshold. Defaults to 0.0.
            prompt (str, optional): Annotation text prompt. Defaults to "".

        Returns:
            list[dict]: Processed rows
        """

        rows = []

        # Import GroundingDINO
        gd_inf = attempt_import(
            "groundingdino.util.inference",
            "groundingdino@git+https://github.com/IDEA-Research/GroundingDINO",
        )

        for view in views:
            # Iterate manually
            for x in range(batch.num_rows):
                # Preprocess image
                im: Image = Image.from_dict(batch[view][x].as_py())
                im.uri_prefix = uri_prefix

                _, image = gd_inf.load_image(im.path.as_posix())

                # Inference
                bbox_tensor, logit_tensor, category_list = gd_inf.predict(
                    model=self.model,
                    image=image,
                    caption=prompt,
                    box_threshold=0.35,
                    text_threshold=0.25,
                )

                # Convert bounding boxes from cyxcywh to xywh
                bbox_tensor = box_convert(
                    boxes=bbox_tensor, in_fmt="cxcywh", out_fmt="xywh"
                )
                bbox_list = [[coord.item() for coord in bbox] for bbox in bbox_tensor]

                # Process model outputs
                rows.extend(
                    [
                        {
                            "id": shortuuid.uuid(),
                            "item_id": batch["id"][x].as_py(),
                            "view_id": view,
                            "bbox": BBox.from_xywh(
                                bbox_list[i],
                                confidence=logit_tensor[i].item(),
                            ).to_dict(),
                            "category": category_list[i],
                        }
                        for i in range(len(category_list))
                        if logit_tensor[i].item() > threshold
                    ]
                )

        return rows
