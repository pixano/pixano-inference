# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Transformers-based VLM (Vision-Language Model)."""

from __future__ import annotations

import gc
import logging
from typing import Any

from pixano_inference.models.registry import register_model
from pixano_inference.models.vlm import UsageInfo, VLMInput, VLMModel, VLMOutput
from pixano_inference.ray.config import ModelDeploymentConfig

from .._helpers import resolve_device


logger = logging.getLogger(__name__)


@register_model("TransformersVLMModel")
class TransformersVLMModel(VLMModel):
    """Native Ray Serve model for Transformers-based VLMs.

    ``model_params`` contract:

    - ``path`` (str, required): HuggingFace model ID or local checkpoint path.
    - ``processor_config`` (dict, optional): Kwargs for ``AutoProcessor.from_pretrained``.
    - ``config`` (dict, optional): Kwargs for model ``from_pretrained``.
    - ``model_type`` (str, optional): Model type hint (e.g. "llava", "llava-next").
      If not provided, falls back to ``AutoModelForVision2Seq``.
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
        """Load the Transformers VLM model and processor."""
        from pixano_inference.utils.package import assert_transformers_installed

        assert_transformers_installed()

        import torch
        from transformers import AutoProcessor

        params = dict(self._config.model_params)
        path = params.pop("path")
        processor_config = params.pop("processor_config", {})
        model_config = params.pop("config", {})
        model_type = params.pop("model_type", None)

        device = resolve_device(self._config)

        self._processor = AutoProcessor.from_pretrained(path, **processor_config)
        self._model = self._load_vlm_model(path, model_type, device, model_config)
        self._model = self._model.eval()
        self._model = torch.compile(self._model)

        logger.info("TransformersVLMModel '%s' loaded on %s", self.model_name, device)

    @staticmethod
    def _load_vlm_model(path: str, model_type: str | None, device: Any, model_config: dict) -> Any:
        """Load the appropriate VLM model based on model_type.

        Args:
            path: HuggingFace model ID or local path.
            model_type: Model type hint.
            device: Torch device.
            model_config: Additional model kwargs.

        Returns:
            Loaded model.
        """
        name = (model_type or path).lower()
        if "llava" in name:
            if "next" in name:
                if "video" in name:
                    from transformers import LlavaNextVideoForConditionalGeneration

                    return LlavaNextVideoForConditionalGeneration.from_pretrained(
                        path, device_map=device, **model_config
                    )
                else:
                    from transformers import LlavaNextForConditionalGeneration

                    return LlavaNextForConditionalGeneration.from_pretrained(path, device_map=device, **model_config)
            else:
                from transformers import LlavaForConditionalGeneration

                return LlavaForConditionalGeneration.from_pretrained(path, device_map=device, **model_config)

        # Fallback to generic Vision2Seq
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(path, device_map=device, **model_config)

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata."""
        base = super().metadata
        params = self._config.model_params
        base["path"] = params.get("path")
        return base

    def predict(self, input: VLMInput) -> VLMOutput:
        """Run VLM generation.

        Args:
            input: VLM input with prompt, images, and generation parameters.

        Returns:
            VLM output with generated text, usage info, and generation config.
        """
        import torch
        from transformers import GenerationConfig

        from pixano_inference.utils.media import convert_string_to_image

        prompt = input.prompt
        images = input.images

        # Parse images from prompt if not provided separately
        pil_images: list[Any] | None
        if images is None:
            if isinstance(prompt, str):
                raise ValueError("Images must be provided if the prompt is a string.")
            pil_images = []
            for message in prompt:
                new_content = []
                for content in message["content"]:
                    if content["type"] == "image_url":
                        pil_images.append(convert_string_to_image(content["image_url"]["url"]))
                        new_content.append({"type": "image"})
                    else:
                        new_content.append(content)
                message["content"] = new_content
        else:
            pil_images = [convert_string_to_image(img) for img in images] if len(images) > 0 else None

        with torch.inference_mode():
            generation_config = GenerationConfig(max_new_tokens=input.max_new_tokens, temperature=input.temperature)

            if isinstance(prompt, list):
                prompt = self._processor.apply_chat_template(prompt, add_generation_prompt=True)

            inputs = self._processor(prompt, pil_images, return_tensors="pt").to(self._model.device)
            generate_ids = self._model.generate(**inputs, generation_config=generation_config)

            total_tokens: int = generate_ids.shape[1]
            prompt_tokens: int = inputs["input_ids"].shape[1]
            completion_tokens: int = total_tokens - prompt_tokens

            output = self._processor.decode(
                generate_ids[0, prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            return VLMOutput(
                generated_text=output,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
                generation_config=generation_config.to_diff_dict(),
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
