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
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from pixano.core import BBox, CompressedRLE, Image, ObjectAnnotation
from pixano.models import InferenceModel
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel


class SAM(InferenceModel):
    """Segment Anything Model (SAM)

    Attributes:
        name (str): Model name
        id (str): Model ID
        device (str): Model GPU or CPU device (e.g. "cuda", "cpu")
        source (str): Model source
        info (str): Additional model info
        model (torch.nn.Module): SAM model
    """

    def __init__(
        self,
        checkpoint_path: Path,
        size: str = "h",
        id: str = "",
        device: str = "cuda",
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
        self.model = sam_model_registry[f"vit_{size}"](checkpoint=checkpoint_path)
        self.model.to(device=self.device)

        # Model path
        self.checkpoint_path = checkpoint_path

    def inference_batch(
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
            list[dict]: Inference rows
        """

        rows = [
            {
                "id": batch["id"][x].as_py(),
                "objects": [],
                "split": batch["split"][x].as_py(),
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
                    generator = SamAutomaticMaskGenerator(self.model)
                    output = generator.generate(im)

                # Process model outputs
                h, w = im.shape[:2]
                rows[x]["objects"].extend(
                    [
                        ObjectAnnotation(
                            id=shortuuid.uuid(),
                            view_id=view,
                            bbox=BBox.from_xywh(output[i]["bbox"]).normalize(h, w),
                            bbox_confidence=float(output[i]["predicted_iou"]),
                            bbox_source=self.id,
                            mask=CompressedRLE.from_mask(output[i]["segmentation"]),
                            mask_source=self.id,
                            category_id=0,
                            category_name="N/A",
                        )
                        for i in range(len(output))
                        if output[i]["predicted_iou"] > threshold
                    ]
                )

        return rows

    def embedding_batch(
        self, batch: pa.RecordBatch, views: list[str], uri_prefix: str
    ) -> list[np.ndarray]:
        """Embedding precomputing for a batch

        Args:
            batch (pa.RecordBatch): Input batch
            views (list[str]): Dataset views
            uri_prefix (str): URI prefix for media files

        Returns:
            list[np.ndarray]: Model embeddings as NumPy arrays
        """

        rows = [
            {
                "id": batch["id"][x].as_py(),
                "split": batch["split"][x].as_py(),
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
                rows[x][f"{view}_embedding"] = emb_bytes.getvalue()

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
        onnx_path = model_dir / self.checkpoint_path.name.replace(".pth", ".onnx")

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
