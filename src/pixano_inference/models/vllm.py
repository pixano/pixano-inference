# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Inference models for vLLM."""

from __future__ import annotations

import gc
from typing import Any

from pixano_inference.models.base import BaseInferenceModel
from pixano_inference.pydantic.tasks.multimodal.conditional_generation import (
    TextImageConditionalGenerationOutput,
    UsageConditionalGeneration,
)
from pixano_inference.utils.package import (
    assert_vllm_installed,
    is_torch_installed,
    is_vllm_installed,
)


if is_torch_installed():
    import torch

if is_vllm_installed():
    import msgspec
    import vllm
    from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams


class VLLMModel(BaseInferenceModel):
    """Inference model for vLLM."""

    def __init__(
        self,
        name: str,
        vllm_model: str,
        model_config: dict[str, Any],
        processor_config: dict[str, Any],
        device: torch.device | str | None = None,
    ):
        """Initialize the model.

        Args:
            name: Name of the model.
            vllm_model: The model Hugging Face hub's identifier.
            model_config: Configuration for the model.
            processor_config: Configuration for the processor of the model.
            device: The device to use for inference.
        """
        assert_vllm_installed()

        super().__init__(name, provider="vllm")
        self.vllm_model = vllm_model
        self.model_config = model_config
        self.processor_config = processor_config
        self.model = LLM(model=vllm_model, **model_config, **processor_config, device=device)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata of the model."""
        return {
            "name": self.name,
            "vllm_model": self.vllm_model,
            "provider": self.provider,
            "provider_version": vllm.__version__,
            "processor_config": self.processor_config,
            "model_config": self.model_config,
        }

    def delete(self):
        """Delete the model."""
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def text_image_conditional_generation(
        self,
        prompt: list[dict[str, Any]],
        temperature: float = 1.0,
        max_new_tokens: int = 16,
        **kwargs: Any,
    ) -> TextImageConditionalGenerationOutput:
        """Generate text from an image and a prompt from the vLLM's `LLM.chat` method.

        Args:
            prompt: Prompt for the generation.
            temperature: Temperature for the generation.
            max_new_tokens: Maximum number of tokens to generate.
            kwargs: Additional generation arguments.
        """
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, **kwargs)
        with torch.inference_mode():
            request_output: RequestOutput = self.model.chat(
                messages=prompt, use_tqdm=False, sampling_params=sampling_params
            )[0]
            prompt = request_output.prompt
            output: CompletionOutput = request_output.outputs[0]

            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = len(output.token_ids)
            total_tokens = prompt_tokens + completion_tokens

            return TextImageConditionalGenerationOutput(
                generated_text=output.text,
                usage=UsageConditionalGeneration(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
                generation_config=msgspec.to_builtins(sampling_params),
            )
