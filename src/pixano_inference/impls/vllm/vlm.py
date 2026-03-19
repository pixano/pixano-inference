# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""vLLM-based VLM (Vision-Language Model)."""

from __future__ import annotations

import gc
import logging
from typing import Any

from pixano_inference.models.registry import register_model
from pixano_inference.models.vlm import UsageInfo, VLMInput, VLMModel, VLMOutput
from pixano_inference.ray.config import ModelDeploymentConfig

from .._helpers import resolve_device


logger = logging.getLogger(__name__)


@register_model("VLLMVLMModel")
class VLLMVLMModel(VLMModel):
    """Native Ray Serve model for vLLM-based VLMs.

    ``model_params`` contract:

    - ``path`` (str, required): HuggingFace model ID.
    - ``config`` (dict, optional): Kwargs for ``vllm.LLM``.
    - ``processor_config`` (dict, optional): Kwargs for ``vllm.LLM`` processor options.
    """

    def __init__(self, config: ModelDeploymentConfig) -> None:
        """Initialize the model.

        Args:
            config: Model deployment configuration.
        """
        super().__init__(config)
        self._llm: Any = None

    def load_model(self) -> None:
        """Load the vLLM model."""
        from pixano_inference.utils.package import assert_vllm_installed

        assert_vllm_installed()

        from vllm import LLM

        params = dict(self._config.model_params)
        path = params.pop("path")
        model_config = params.pop("config", {})
        processor_config = params.pop("processor_config", {})

        device = resolve_device(self._config)

        self._llm = LLM(model=path, **model_config, **processor_config, device=str(device), tensor_parallel_size=1)

        logger.info("VLLMVLMModel '%s' loaded on %s", self.model_name, device)

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata."""
        base = super().metadata
        params = self._config.model_params
        base["path"] = params.get("path")
        return base

    def predict(self, input: VLMInput) -> VLMOutput:
        """Run VLM generation via vLLM.

        Args:
            input: VLM input with prompt and generation parameters.
                Images must be embedded in the prompt for vLLM (``input.images`` must be None).

        Returns:
            VLM output with generated text, usage info, and generation config.
        """
        import torch

        from pixano_inference.utils.package import assert_vllm_installed

        assert_vllm_installed()

        import msgspec
        from vllm import SamplingParams

        if input.images is not None:
            raise ValueError("images should be passed in the prompt for vLLM.")
        if isinstance(input.prompt, str):
            raise ValueError("Pixano-inference only supports a chat template for vLLM.")

        sampling_params = SamplingParams(temperature=input.temperature, max_tokens=input.max_new_tokens)

        with torch.inference_mode():
            request_output = self._llm.chat(messages=input.prompt, use_tqdm=False, sampling_params=sampling_params)[0]
            output = request_output.outputs[0]

            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = len(output.token_ids)
            total_tokens = prompt_tokens + completion_tokens

            return VLMOutput(
                generated_text=output.text,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
                generation_config=msgspec.to_builtins(sampling_params),
            )

    def unload(self) -> None:
        """Free resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
