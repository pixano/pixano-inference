# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""YOLO as custom model example.

Demonstrates how to wrap a third-party model (ultralytics YOLO) as a
pixano-inference :class:`DetectionModel` and register it for deployment
with Ray Serve.

Requirements:
    pip install ultralytics

Usage:
    PYTHONPATH=examples:$PYTHONPATH pixano-inference --config examples/yolo/config.py
"""

from __future__ import annotations

import gc
import logging
from typing import Any

from pixano_inference.models.detection import DetectionInput, DetectionModel, DetectionOutput
from pixano_inference.models.registry import register_model
from pixano_inference.ray.config import ModelDeploymentConfig


logger = logging.getLogger(__name__)


@register_model("YOLOModel")
class YOLOModel(DetectionModel):
    """YOLO Detection model."""

    def __init__(self, config: ModelDeploymentConfig) -> None:
        """Initialize the model.

        Args:
            config: Model deployment configuration.
        """
        super().__init__(config)
        self._model: Any = None

    def load_model(self) -> None:
        """Load the YOLO model from ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for YOLOModel. Install with: pip install ultralytics") from exc

        params = dict(self._config.model_params)
        path = params.pop("path")

        device = "cpu"
        if self._config.resources.num_gpus > 0:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass

        self._model = YOLO(path)
        self._model.to(device)

        logger.info("YOLOModel '%s' loaded on %s", self.model_name, device)

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

        return DetectionOutput(
            boxes=boxes,
            scores=scores,
            classes=class_names,
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
