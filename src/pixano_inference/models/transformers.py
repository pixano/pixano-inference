# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Inference models for Transformers."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pixano_inference.models.base import BaseInferenceModel
from pixano_inference.pydantic import (
    CompressedRLE,
    ImageMaskGenerationOutput,
    ImageZeroShotDetectionOutput,
    NDArrayFloat,
    TextImageConditionalGenerationOutput,
    UsageConditionalGeneration,
)
from pixano_inference.utils.media import encode_mask_to_rle
from pixano_inference.utils.package import (
    assert_transformers_installed,
    is_torch_installed,
    is_transformers_installed,
)


if is_torch_installed():
    import torch

if is_transformers_installed():
    import transformers
    from transformers import GenerationConfig


if TYPE_CHECKING:
    from PIL.Image import Image
    from torch import Tensor
    from transformers import PreTrainedModel, ProcessorMixin


class TransformerModel(BaseInferenceModel):
    """Inference model for transformers."""

    def __init__(self, name: str, path: Path | str, processor: "ProcessorMixin", model: "PreTrainedModel"):
        """Initialize the model.

        Args:
            name: Name of the model.
            path: Path to the model or its Hugging Face hub's identifier.
            processor: Processor for the model.
            model: Model for the inference.
        """
        assert_transformers_installed()

        super().__init__(name, provider="transformers")
        self.processor = processor
        self.path = path
        self.model = model.eval()

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata of the model."""
        return {
            "name": self.name,
            "path": self.path,
            "provider": self.provider,
            "provider_version": transformers.__version__,
            "processor_config": self.processor.to_dict(),
            "model_config": self.model.config.to_dict(),
        }

    def _fill_generation_config(self, generation_config: "GenerationConfig", **kwargs) -> "GenerationConfig":
        generation_config = generation_config.to_dict()
        for kwarg in kwargs:
            if kwarg in generation_config:
                generation_config[kwarg] = kwargs[kwarg]
        return GenerationConfig.from_dict(generation_config)

    def delete(self):
        """Delete the model."""
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()

    def image_mask_generation(
        self,
        image: "Tensor" | Image,
        image_embedding: "Tensor" | None = None,
        points: list[list[list[int]]] | None = None,
        labels: list[list[int]] | None = None,
        boxes: list[list[int]] | None = None,
        num_multimask_outputs: int = 3,
        multimask_output: bool = True,
        return_image_embedding: bool = False,
        **kwargs: Any,
    ) -> ImageMaskGenerationOutput:
        """Generate a mask from the image.

        Args:
            image: Image for the generation.
            image_embedding: Image embeddings for the generation.
            points: Points for the mask generation. The first fimension is the number of prompts, the
                second the number of points per mask and the third the coordinates of the points.
            labels: Labels for the mask generation. The first fimension is the number of prompts, the second
                the number of labels per mask.
            boxes: Boxes for the mask generation. The first fimension is the number of prompts, the second
                the coordinates of the boxes.
            num_multimask_outputs: Number of masks to generate per prediction.
            multimask_output: Whether to generate multiple masks per prediction.
            return_image_embedding: Whether to return the image embedding.
            kwargs: Additional keyword arguments.
        """
        with torch.inference_mode():
            inputs = self.processor(
                image,
                input_points=[points] if points is not None else None,
                input_boxes=[boxes] if boxes is not None else None,
                input_labels=[labels] if labels is not None else None,
                return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)

            if return_image_embedding:
                if image_embedding is None:  # Compute image embeddings if not provided
                    image_embedding = self.model.get_image_embeddings(inputs["pixel_values"])

            if image_embedding is not None:
                if image_embedding.ndim == 3:
                    image_embedding = image_embedding.unsqueeze(0)
                inputs.pop("pixel_values", None)
                inputs.update({"image_embeddings": image_embedding.to(self.model.device, dtype=self.model.dtype)})

            outputs = self.model(
                **inputs, num_multimask_outputs=num_multimask_outputs, multimask_output=multimask_output, **kwargs
            )

            masks = (
                self.processor.image_processor.post_process_masks(
                    outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
                )
            )[0].cpu()
            return ImageMaskGenerationOutput(
                masks=[
                    [CompressedRLE(**encode_mask_to_rle(mask)) for mask in prediction_masks]
                    for prediction_masks in masks
                ],
                scores=NDArrayFloat.from_torch(outputs.iou_scores[0].cpu()),
                image_embedding=(
                    NDArrayFloat.from_torch(image_embedding[0].cpu()) if return_image_embedding else None
                ),
            )

    def text_image_conditional_generation(
        self,
        prompt: str | list[dict[str, Any]],
        images: list["Tensor"],
        generation_config: "GenerationConfig" | None = None,
        **kwargs: Any,
    ) -> TextImageConditionalGenerationOutput:
        """Generate text from an image and a prompt.

        Args:
            prompt: Prompt for the generation.
            images: Images for the generation.
            generation_config: Configuration for the generation as Hugging Face's GenerationConfig.
            kwargs: Additional keyword arguments.
        """
        with torch.inference_mode():
            if generation_config is None:
                generation_config = GenerationConfig()

            generation_config = self._fill_generation_config(generation_config, **kwargs)

            if isinstance(prompt, list):
                prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)

            inputs = self.processor(prompt, images, return_tensors="pt").to(self.model.device)
            generate_ids = self.model.generate(**inputs, generation_config=generation_config)

            total_tokens: int = generate_ids.shape[1]
            prompt_tokens: int = inputs["input_ids"].shape[1]
            completion_tokens: int = total_tokens - prompt_tokens

            output = self.processor.decode(
                generate_ids[0, prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            return TextImageConditionalGenerationOutput(
                generated_text=output,
                usage=UsageConditionalGeneration(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
                generation_config=generation_config.to_diff_dict(),
            )

    def image_zero_shot_detection(
        self,
        image: "Tensor" | Image,
        classes: str,
        box_threshold: float,
        text_threshold: float,
        **kwargs: Any,
    ) -> ImageZeroShotDetectionOutput:
        """Perform zero shot detection on an image.

        Args:
            image: The image.
            classes: The list of classes to detect in the format 'class1. class2'.
            box_threshold: The threshold for bounding boxes detection.
            text_threshold: The threshold for the classes identification during zero shot learning phase.
            kwargs: Additional arguments.

        Returns:
            The output of image zero-shot detection task.
        """
        with torch.inference_mode():
            inputs = self.processor(images=image, text=classes, return_tensors="pt").to(self.model.device)

            outputs = self.model(**inputs)

            target_size = (
                (image.shape[-2], image.shape[-1]) if isinstance(image, torch.Tensor) else (image.height, image.width)
            )

            result = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[target_size],
            )[0]

            return ImageZeroShotDetectionOutput(
                boxes=[[int(round(x, 0)) for x in box.tolist()] for box in result["boxes"]],
                scores=result["scores"],
                classes=result["labels"],
            )
