# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Grounding DINO zero-shot detection model."""

from __future__ import annotations

import gc
import logging
from typing import Any

from pixano_inference.models.detection import DetectionInput, DetectionModel, DetectionOutput
from pixano_inference.models.registry import register_model
from pixano_inference.ray.config import ModelDeploymentConfig

from .._helpers import resolve_device


logger = logging.getLogger(__name__)


@register_model("GroundingDINOModel")
class GroundingDINOModel(DetectionModel):
    """Native Ray Serve model for Grounding DINO zero-shot detection.

    ``model_params`` contract:

    - ``path`` (str, required): HuggingFace model ID or local checkpoint path.
    - ``processor_config`` (dict, optional): Kwargs for ``AutoProcessor.from_pretrained``.
    - ``config`` (dict, optional): Kwargs for ``AutoModelForZeroShotObjectDetection.from_pretrained``.
    """

    def __init__(self, config: ModelDeploymentConfig) -> None:
        """Initialize the model.

        Args:
            config: Model deployment configuration.
        """
        super().__init__(config)
        self._model: Any = None
        self._processor: Any = None

    def load_model(self) -> None:
        """Load the Grounding DINO model and processor."""
        from pixano_inference.utils.package import assert_transformers_installed

        assert_transformers_installed()

        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        params = dict(self._config.model_params)
        path = params.pop("path")
        processor_config = params.pop("processor_config", {})
        model_config = params.pop("config", {})

        device = resolve_device(self._config)

        self._processor = AutoProcessor.from_pretrained(path, **processor_config)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(path, device_map=device, **model_config)
        self._model = self._model.eval()
        self._model = torch.compile(self._model)

        logger.info("GroundingDINOModel '%s' loaded on %s", self.model_name, device)

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata."""
        base = super().metadata
        params = self._config.model_params
        base["path"] = params.get("path")
        return base

    def predict(self, input: DetectionInput) -> DetectionOutput:
        """Run zero-shot detection.

        Args:
            input: Detection input with image, classes, and thresholds.

        Returns:
            Detection output with boxes, scores, and classes.

        Raises:
            ValueError: If ``classes`` is not provided (GroundingDINO is open-vocab only).
        """
        import torch

        from pixano_inference.utils.media import convert_string_to_image

        if input.classes is None:
            raise ValueError("GroundingDINOModel requires 'classes' (open-vocabulary model).")

        pil_image = convert_string_to_image(input.image)
        classes = input.classes

        if isinstance(classes, list):
            classes_str = ". ".join(classes)
        elif classes is not None:
            classes_str = classes
        else:
            classes_str = ""

        with torch.inference_mode():
            inputs = self._processor(images=pil_image, text=classes_str, return_tensors="pt").to(self._model.device)

            outputs = self._model(**inputs)

            target_size = (pil_image.height, pil_image.width)

            result = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=input.box_threshold,
                text_threshold=input.text_threshold,
                target_sizes=[target_size],
            )[0]

            return DetectionOutput(
                boxes=[[int(round(x, 0)) for x in box.tolist()] for box in result["boxes"]],
                scores=result["scores"].tolist() if hasattr(result["scores"], "tolist") else result["scores"],
                classes=result["labels"],
            )

    def unload(self) -> None:
        """Free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
