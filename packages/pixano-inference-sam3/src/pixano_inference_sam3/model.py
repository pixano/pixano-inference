# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""SAM 3 as a detection model: by text, by exemplar boxes, or both."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pixano_inference_torch import resolve_device

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models.detection import DetectionInput, DetectionInterface, DetectionModel, DetectionOutput
from pixano_inference.models.registry import register_model
from pixano_inference.schemas.detection import class_names_of
from pixano_inference.utils.package import assert_package_installed

from ._bpe import ensure_bpe


logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "sam3.pt"
CONFIG_NAME = "config.json"


def normalised_cxcywh(box: list[int], width: int, height: int) -> list[float]:
    """A pixel ``[x1, y1, x2, y2]`` box as SAM 3's geometric prompt takes it: centre and size, in [0, 1]."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]


def _to_numpy(value: Any) -> np.ndarray:
    """A tensor (any device, any dtype) or array-like as a numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
        if value.dtype.is_floating_point:
            value = value.float()
        return value.cpu().numpy()
    return np.asarray(value)


@register_model("Sam3DetectionModel")
class Sam3DetectionModel(DetectionModel):
    """SAM 3 open-vocabulary detection.

    A request names classes, each detected by its own text prompt; or it gives
    ``exemplar_boxes`` (and ``negative_exemplar_boxes``), a single prompt that finds more objects
    like the boxed ones, with the class it names (at most one) as text alongside them. The image is
    encoded once per request, whatever the number of prompts. ``box_threshold`` is SAM 3's score
    threshold. Boxes are ``[x1, y1, x2, y2]`` in pixels of the image as the server decodes it.

    ``model_params`` contract (``Sam3DetectionParams``):

    - ``path`` (str, default ``"facebook/sam3"``): Hugging Face repository or local checkpoint.
    - ``bpe_path`` (str, optional): the tokenizer vocabulary; fetched once when absent.
    - ``return_masks`` (bool, default ``True``): whether detections carry their masks.
    - ``compile`` (bool, default ``False``): whether to ``torch.compile`` the model.
    """

    def __init__(self, config: ModelDeploymentConfig) -> None:
        """Initialize the model.

        Args:
            config: Model deployment configuration.
        """
        super().__init__(config)
        self._processor: Any = None
        self._return_masks = bool(config.model_params.get("return_masks", True))

    def load_model(self) -> None:
        """Build SAM 3's image model and its processor.

        Raises:
            RuntimeError: If the installed core predates exemplars: its requests would drop them.
        """
        if "exemplar_boxes" not in DetectionInput.model_fields:
            raise RuntimeError(
                "Sam3DetectionModel needs a pixano-inference core whose detection requests carry exemplar_boxes; "
                "upgrade pixano-inference."
            )
        assert_package_installed("sam3", "sam3 is not installed. Please install it using 'pip install sam3==0.1.4'.")
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        params = self._config.model_params
        path = params.get("path", "facebook/sam3")
        bpe_path = ensure_bpe(params.get("bpe_path"))
        device = resolve_device(self._config)

        model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            device=str(device),
            checkpoint_path=self._checkpoint(path),
            load_from_HF=False,
            compile=bool(params.get("compile", False)),
        )
        self._processor = Sam3Processor(model, device=str(device))
        logger.info("Sam3DetectionModel '%s' loaded from %s on %s", self.model_name, path, device)

    @staticmethod
    def _checkpoint(path: str) -> str:
        """The local checkpoint file: ``path`` itself, or ``sam3.pt`` downloaded from that repository."""
        if Path(path).is_file():
            return path
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=path, filename=CONFIG_NAME)
        return hf_hub_download(repo_id=path, filename=CHECKPOINT_NAME)

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata including the checkpoint."""
        base = super().metadata
        base["path"] = self._config.model_params.get("path", "facebook/sam3")
        return base

    @property
    def interface(self) -> DetectionInterface:
        """Open vocabulary, by text and by exemplars (positive and negative); boxes, scores, classes, masks."""
        outputs: list[Any] = ["box", "score", "class"] + (["mask"] if self._return_masks else [])
        return DetectionInterface(
            classes="open", thresholds=["box"], exemplars=True, negative_exemplars=True, outputs=outputs
        )

    def predict(self, input: DetectionInput) -> DetectionOutput:
        """Detect what the request names or boxes.

        Args:
            input: Detection input: the image, classes and/or exemplar boxes, the box threshold.

        Returns:
            Boxes, scores, classes and, unless ``return_masks`` is off, compressed-RLE masks.

        Raises:
            ValueError: If the request names no class and gives no exemplar.
        """
        from pixano_inference.schemas.rle import CompressedRLE
        from pixano_inference.utils.media import convert_string_to_image

        names = class_names_of(input.classes)
        if input.exemplar_boxes:
            # One prompt: the exemplars, with the class they are of (if named) as text.
            prompts = [(names[0] if names else None, input.exemplar_boxes, input.negative_exemplar_boxes or [])]
        elif names:
            prompts = [(name, [], []) for name in names]
        else:
            raise ValueError("Sam3DetectionModel requires 'classes' or 'exemplar_boxes' (open-vocabulary model).")

        image = convert_string_to_image(input.image)
        width, height = image.size
        processor = self._processor
        processor.confidence_threshold = input.box_threshold
        state = processor.set_image(image)

        boxes: list[list[int]] = []
        scores: list[float] = []
        classes: list[str] = []
        masks: list[CompressedRLE] = []
        for text, positives, negatives in prompts:
            processor.reset_all_prompts(state)
            if text:
                state = processor.set_text_prompt(text, state)
            for prompt_boxes, positive in ((positives, True), (negatives, False)):
                for box in prompt_boxes:
                    state = processor.add_geometric_prompt(normalised_cxcywh(box, width, height), positive, state)

            found_boxes = _to_numpy(state["boxes"]).reshape(-1, 4)
            found_scores = _to_numpy(state["scores"]).reshape(-1)
            found_masks = _to_numpy(state["masks"]) if self._return_masks else None
            for index, (box, score) in enumerate(zip(found_boxes, found_scores, strict=True)):
                x1, y1 = max(0, round(float(box[0]))), max(0, round(float(box[1])))
                x2, y2 = min(width, round(float(box[2]))), min(height, round(float(box[3])))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))
                classes.append(text or "")
                if found_masks is not None:
                    # SAM 3 returns one (1, H, W) boolean mask per detection.
                    masks.append(CompressedRLE.from_mask(found_masks[index].reshape(height, width).astype(np.uint8)))

        return DetectionOutput(
            boxes=boxes, scores=scores, classes=classes, masks=masks if self._return_masks else None
        )
