# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""YOLOE custom model example.

Demonstrates how to wrap a third-party model (ultralytics YOLOE) as a
pixano-inference :class:`DetectionModel` and register it for deployment
with Ray Serve.

Requirements:
    pip install ultralytics

Usage:
    PYTHONPATH=examples:$PYTHONPATH pixano-inference --config examples/custom_yoloe/config.py
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np

from pixano_inference.models.detection import DetectionInput, DetectionModel, DetectionOutput
from pixano_inference.models.registry import register_model
from pixano_inference.ray.config import ModelDeploymentConfig
from pixano_inference.schemas.rle import CompressedRLE


logger = logging.getLogger(__name__)


@register_model("YOLOEModel")
class YOLOEModel(DetectionModel):
    """YOLOE instance segmentation model.

    Supports both **open-vocabulary** mode (classes provided at inference
    time via ``input.classes``) and **closed-vocabulary / prompt-free** mode
    (``input.classes`` is ``None``).

    ``model_params`` contract:

    - ``path`` (str, required): Model checkpoint path or name (e.g.
      ``"yoloe-11s-seg.pt"``).
    - ``config`` (dict, optional): Extra kwargs forwarded to
      ``YOLOE(path, **config)``.
    """

    def __init__(self, config: ModelDeploymentConfig) -> None:
        """Initialize the model.

        Args:
            config: Model deployment configuration.
        """
        super().__init__(config)
        self._model: Any = None

    def load_model(self) -> None:
        """Load the YOLOE model from ultralytics."""
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise ImportError("ultralytics is required for YOLOEModel. Install with: pip install ultralytics") from exc

        params = dict(self._config.model_params)
        path = params.pop("path")
        extra = params.pop("config", {})

        device = "cpu"
        if self._config.resources.num_gpus > 0:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass

        self._model = YOLOE(path, **extra)
        self._model.to(device)

        logger.info("YOLOEModel '%s' loaded on %s", self.model_name, device)

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata."""
        base = super().metadata
        base["path"] = self._config.model_params.get("path")
        return base

    def predict(self, input: DetectionInput) -> DetectionOutput:
        """Run detection / instance segmentation.

        Args:
            input: Detection input with image and optional classes.

        Returns:
            Detection output with boxes, scores, classes, and optional masks.
        """
        from pixano_inference.utils.media import convert_string_to_image

        pil_image = convert_string_to_image(input.image)

        # Open-vocab: set classes on the model before prediction
        if input.classes is not None:
            classes = [input.classes] if isinstance(input.classes, str) else list(input.classes)
            self._model.set_classes(classes)

        results = self._model.predict(pil_image, conf=input.box_threshold)
        result = results[0]

        # --- boxes, scores, class names ---
        boxes: list[list[int]] = []
        scores: list[float] = []
        class_names: list[str] = []

        if result.boxes is not None and len(result.boxes):
            for box, conf, cls_id in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                boxes.append([int(round(c)) for c in box.tolist()])
                scores.append(float(conf))
                class_names.append(result.names[int(cls_id)])

        # --- masks (instance segmentation) ---
        masks: list[CompressedRLE] | None = None
        if result.masks is not None and len(result.masks.data):
            masks = []
            for mask_tensor in result.masks.data.cpu():
                mask_np = mask_tensor.numpy().astype(np.uint8)
                masks.append(CompressedRLE.from_mask(mask_np))

        return DetectionOutput(
            boxes=boxes,
            scores=scores,
            classes=class_names,
            masks=masks,
        )

    def unload(self) -> None:
        """Free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
