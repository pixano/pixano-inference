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
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import torch
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

from pixano.inference import OnlineModel


class SAM(OnlineModel):
    """Segment Anything Model (SAM)

    Attributes:
        name (str): Model name
        id (str): Model ID
        device (str): Model GPU or CPU device
        source (str): Model source
        info (str): Additional model info
        onnx_path (Path): ONNX Model Path
        onnx_session (onnxruntime.InferenceSession): ONNX session
        working (dict): Dictionary of current working data
        sam (torch.nn.Module): SAM model
        model (torch.nn.Module): PyTorch Predictor model
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
            checkpoint_path (Path): Model checkpoint path
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
        self.model = SamPredictor(self.sam)

        # Model path
        self.checkpoint_path = checkpoint_path

    def __call__(self, input: dict[np.ndarray]) -> np.ndarray:
        """Return model annotation based on user input

        Args:
            input (dict[np.ndarray]): User input

        Returns:
            np.ndarray: Model annotation masks
        """

        # Preprocess image and user input
        input["point_coords"] = self.model.transform.apply_coords(
            input["point_coords"], self.working["image"].shape[:2]
        ).astype(np.float32)

        # Inference
        masks, _, _ = self.onnx_session.run(None, input)

        # Process model outputs
        self.working["masks"] = masks > self.model.mask_threshold
        return self.working["masks"]

    def process_batch(
        self, batch: pa.RecordBatch, view: str, media_dir: Path
    ) -> list[np.ndarray]:
        """Precompute embeddings for a batch

        Args:
            batch (pa.RecordBatch): Input batch
            view (str): Dataset view
            media_dir (Path): Media location

        Returns:
            list[np.ndarray]: Model embeddings as NumPy arrays
        """

        embeddings = []

        # Iterate manually
        for x in range(batch.num_rows):
            # Preprocess image
            img = cv2.imread(str(media_dir / batch[view][x].as_py()._uri))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Inference
            with torch.no_grad():
                self.model.set_image(img)
                img_embedding = self.model.get_image_embedding().cpu().numpy()

            # Process model outputs
            embeddings.append(img_embedding)
        return embeddings

    def export_onnx_model(self) -> Path:
        """Export Torch model to ONNX

        Returns:
            Path: ONNX model path
        """

        # Put model to CPU for export
        self.sam.to("cpu")

        # Export settings
        onnx_model = SamOnnxModel(self.sam, return_single_mask=True)
        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }
        embed_dim = self.sam.prompt_encoder.embed_dim
        embed_size = self.sam.prompt_encoder.image_embedding_size
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
        onnx_path = self.checkpoint_path.as_posix().replace(".pth", ".onnx")

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
        self.sam.to(self.device)

        return onnx_path
