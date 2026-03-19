# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""SAM2 image segmentation model."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

from pixano_inference.models.registry import register_model
from pixano_inference.models.segmentation import SegmentationInput, SegmentationModel, SegmentationOutput
from pixano_inference.ray.config import ModelDeploymentConfig

from .._helpers import pad_points_and_labels, resolve_device, resolve_torch_dtype, validate_prompts


logger = logging.getLogger(__name__)


@register_model("Sam2ImageModel")
class Sam2ImageModel(SegmentationModel):
    """Native Ray Serve model for SAM2 image mask generation.

    ``model_params`` contract:

    - ``path`` (str, required): HuggingFace model ID or local checkpoint path.
    - ``torch_dtype`` (str, default ``"bfloat16"``): Torch dtype for autocast.
    - ``compile`` (bool, default ``True``): Whether to ``torch.compile`` the model.

    Any remaining keys are forwarded to ``build_sam2`` / ``build_sam2_hf``.
    """

    def __init__(self, config: ModelDeploymentConfig) -> None:
        """Initialize the model.

        Args:
            config: Model deployment configuration.
        """
        super().__init__(config)
        self._predictor: Any = None
        self._torch_dtype: Any = None

    def load_model(self) -> None:
        """Load the SAM2 image predictor."""
        from pixano_inference.utils.package import assert_sam2_installed

        assert_sam2_installed()

        import torch
        from sam2.build_sam import build_sam2, build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        params = dict(self._config.model_params)
        path = params.pop("path")
        torch_dtype_str = params.pop("torch_dtype", "bfloat16")
        compile_model = params.pop("compile", True)

        device = resolve_device(self._config)
        self._torch_dtype = resolve_torch_dtype(torch_dtype_str)

        if path is not None and Path(path).exists():
            model = build_sam2(ckpt_path=path, mode="eval", device=device, **params)
        else:
            model = build_sam2_hf(model_id=path, mode="eval", device=device, **params)

        if compile_model:
            model = torch.compile(model)

        self._predictor = SAM2ImagePredictor(model)
        logger.info("Sam2ImageModel '%s' loaded on %s (dtype=%s)", self.model_name, device, torch_dtype_str)

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata including path and dtype."""
        base = super().metadata
        params = self._config.model_params
        base["path"] = params.get("path")
        base["torch_dtype"] = params.get("torch_dtype", "bfloat16")
        if self._predictor is not None:
            base["device"] = str(self._predictor.device)
        return base

    def predict(self, input: SegmentationInput) -> SegmentationOutput:
        """Run SAM2 image mask generation.

        Args:
            input: Segmentation input with image, prompts, and options.

        Returns:
            Segmentation output with masks, scores, and optionally embeddings.
        """
        import torch

        from pixano_inference.schemas.nd_array import NDArrayFloat
        from pixano_inference.schemas.rle import CompressedRLE
        from pixano_inference.utils.media import convert_string_to_image

        pil_image = convert_string_to_image(input.image)
        validate_prompts(input.points, input.labels, input.boxes)

        if input.multimask_output and input.num_multimask_outputs != 3:
            raise ValueError("The number of multimask outputs is not configurable for SAM2 and must be 3.")

        # Handle predictor reset and optional embedding restoration
        if input.reset_predictor:
            self._predictor.reset_predictor()
            if input.image_embedding is not None and input.high_resolution_features is not None:
                self._set_image_embeddings(pil_image, input.image_embedding, input.high_resolution_features)
            elif input.image_embedding is not None or input.high_resolution_features is not None:
                raise ValueError("Both image_embedding and high_resolution_features must be provided.")

        # Prepare numpy inputs
        input_points: np.ndarray | None = None
        input_labels: np.ndarray | None = None
        input_boxes: np.ndarray | None = None

        if input.points is not None:
            input_points, input_labels = pad_points_and_labels(input.points, input.labels)
        if input.boxes is not None:
            input_boxes = np.array(input.boxes, dtype=np.int32)

        with torch.inference_mode():
            with torch.autocast(self._predictor.device.type, dtype=self._torch_dtype):
                if not self._predictor._is_image_set:
                    self._predictor.set_image(pil_image)

                masks, scores, _ = self._predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_boxes,
                    mask_input=None,
                    multimask_output=input.multimask_output,
                    return_logits=False,
                )

        # Ensure 4D: [num_prompts, num_masks, H, W]
        if len(masks.shape) == 3:
            masks = np.expand_dims(masks, 0)
            scores = np.expand_dims(scores, 0)

        # Build output
        out_masks = [
            [CompressedRLE.from_mask(mask.astype(np.uint8)) for mask in prediction_masks] for prediction_masks in masks
        ]
        out_scores = NDArrayFloat.from_numpy(scores)

        out_image_embedding: NDArrayFloat | None = None
        out_high_resolution_features: list[NDArrayFloat] | None = None

        if input.return_image_embedding:
            embed = self._predictor._features["image_embed"]
            embed_list = embed.to(torch.float32).flatten().tolist()
            out_image_embedding = NDArrayFloat(values=embed_list, shape=list(embed.shape[1:]))

            hr_feats = self._predictor._features["high_res_feats"]
            out_high_resolution_features = [
                NDArrayFloat(
                    values=feat.to(torch.float32).flatten().tolist(),
                    shape=list(feat.shape[1:]),
                )
                for feat in hr_feats
            ]

        return SegmentationOutput(
            masks=out_masks,
            scores=out_scores,
            image_embedding=out_image_embedding,
            high_resolution_features=out_high_resolution_features,
        )

    def _set_image_embeddings(
        self,
        image: Any,
        image_embedding: Any,
        high_resolution_features: list[Any],
    ) -> None:
        """Restore pre-computed image embeddings into the predictor.

        Args:
            image: PIL Image (used for ``_orig_hw``).
            image_embedding: ``NDArrayFloat`` instance.
            high_resolution_features: List of ``NDArrayFloat`` instances.
        """
        import torch

        from pixano_inference.schemas.nd_array import NDArrayFloat

        with torch.inference_mode():
            self._predictor.reset_predictor()

            w, h = image.size
            self._predictor._orig_hw = [(h, w)]

            if isinstance(image_embedding, dict):
                image_embedding = NDArrayFloat.model_validate(image_embedding)
            embed_tensor = image_embedding.to_torch()

            hr_tensors = []
            for feat in high_resolution_features:
                if isinstance(feat, dict):
                    feat = NDArrayFloat.model_validate(feat)
                hr_tensors.append(feat.to_torch())

            device = self._predictor.model.device

            self._predictor._features = {
                "image_embed": embed_tensor.unsqueeze(0).to(device=device),
                "high_res_feats": [feat.unsqueeze(0).to(device=device) for feat in hr_tensors],
            }
            self._predictor._is_image_set = True

    def unload(self) -> None:
        """Free resources."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
