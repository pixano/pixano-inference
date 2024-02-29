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

import warnings
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import shortuuid
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from pixano.core import BBox, CompressedRLE, Image
from pixano.models import InferenceModel


class MobileSAM(InferenceModel):
    """MobileSAM

    Attributes:
        name (str): Model name
        model_id (str): Model ID
        device (str): Model GPU or CPU device (e.g. "cuda", "cpu")
        description (str): Model description
        model (torch.nn.Module): MobileSAM model
        checkpoint_path (Path): Model checkpoint path
    """

    def __init__(
        self,
        checkpoint_path: Path,
        model_id: str = "",
        device: str = "cpu",
    ) -> None:
        """Initialize model

        Args:
            checkpoint_path (Path): Model checkpoint path.
            model_id (str, optional): Previously used ID, generate new ID if "". Defaults to "".
            device (str, optional): Model GPU or CPU device (e.g. "cuda", "cpu"). Defaults to "cpu".
        """

        super().__init__(
            name="Mobile_SAM",
            model_id=model_id,
            device=device,
            description="From GitHub. MobileSAM, ViT-T backbone.",
        )

        # Model
        self.model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        self.model.to(device=self.device)

        # Model path
        self.checkpoint_path = checkpoint_path

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
            # Iterate manually
            for x in range(batch.num_rows):
                # Preprocess image
                im = Image.from_dict(batch[view][x].as_py())
                im.uri_prefix = uri_prefix
                im = im.as_cv2()
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                # Inference
                with torch.no_grad():
                    generator = SamAutomaticMaskGenerator(self.model)
                    output = generator.generate(im)

                # Process model outputs
                h, w = im.shape[:2]
                rows.extend(
                    [
                        {
                            "id": shortuuid.uuid(),
                            "item_id": batch["id"][x].as_py(),
                            "view_id": view,
                            "bbox": BBox.from_xywh(
                                [coord.item() for coord in output[i]["bbox"]],
                                confidence=output[i]["predicted_iou"].item(),
                            )
                            .normalize(h, w)
                            .to_dict(),
                            "mask": CompressedRLE.from_mask(
                                output[i]["segmentation"]
                            ).to_dict(),
                        }
                        for i in range(len(output))
                        if output[i]["predicted_iou"] > threshold
                    ]
                )

        return rows

    def precompute_embeddings(
        self,
        batch: pa.RecordBatch,
        views: list[str],
        uri_prefix: str,
    ) -> list[dict]:
        """Embedding precomputing for a batch

        Args:
            batch (pa.RecordBatch): Input batch
            views (list[str]): Dataset views
            uri_prefix (str): URI prefix for media files

        Returns:
            pa.RecordBatch: Embedding rows
        """

        rows = [
            {
                "id": batch["id"][x].as_py(),
            }
            for x in range(batch.num_rows)
        ]

        for view in views:
            # Iterate manually
            for x in range(batch.num_rows):
                # Preprocess image
                im = Image.from_dict(batch[view][x].as_py())
                im.uri_prefix = uri_prefix
                im = im.as_cv2()
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                # Inference
                with torch.no_grad():
                    predictor = SamPredictor(self.model)
                    predictor.set_image(im)
                    img_embedding = predictor.get_image_embedding().cpu().numpy()

                # Process model outputs
                emb_bytes = BytesIO()
                np.save(emb_bytes, img_embedding)
                rows[x][view] = emb_bytes.getvalue()

        return rows

    def export_to_onnx(self, library_dir: Path):
        """Export Torch model to ONNX

        Args:
            library_dir (Path): Dataset library directory
        """

        # Model directory
        model_dir = library_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Put model to CPU for export
        self.model.to("cpu")

        # Export settings
        onnx_model = SamOnnxModel(self.model, return_single_mask=True)
        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }
        embed_dim = self.model.prompt_encoder.embed_dim
        embed_size = self.model.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        dummy_inputs = {
            "image_embeddings": torch.randn(
                1, embed_dim, *embed_size, dtype=torch.float
            ),
            "point_coords": torch.randint(
                low=0, high=1024, size=(1, 5, 2), dtype=torch.float
            ),
            "point_labels": torch.randint(
                low=0, high=4, size=(1, 5), dtype=torch.float
            ),
            "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
            "has_mask_input": torch.tensor([1], dtype=torch.float),
            "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
        }
        output_names = ["masks", "iou_predictions", "low_res_masks"]
        onnx_path = model_dir / self.checkpoint_path.name.replace(".pt", ".onnx")

        # Export model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            with open(onnx_path, "wb") as f:
                torch.onnx.export(
                    onnx_model,
                    tuple(dummy_inputs.values()),
                    f,
                    export_params=True,
                    verbose=False,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=list(dummy_inputs.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )
        # Quantize model
        quantize_dynamic(
            model_input=onnx_path,
            model_output=onnx_path,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )

        # Put model back to device after export
        self.model.to(self.device)
