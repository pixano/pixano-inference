# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Provider for vLLM models."""

from typing import Any, cast

from pixano_inference.models.vllm import VLLMModel
from pixano_inference.models_registry import register_model
from pixano_inference.pydantic.tasks.multimodal.conditional_generation import (
    TextImageConditionalGenerationOutput,
    TextImageConditionalGenerationRequest,
)
from pixano_inference.settings import PIXANO_INFERENCE_SETTINGS, Settings
from pixano_inference.tasks import Task, str_to_task
from pixano_inference.utils.package import (
    assert_vllm_installed,
)

from .base import ModelProvider
from .registry import register_provider


@register_provider("vllm")
class VLLMProvider(ModelProvider):
    """Provider for vLLM models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the vLLM provider."""
        assert_vllm_installed()
        super().__init__(*args, **kwargs)

    def load_model(
        self,
        name: str,
        task: Task | str,
        settings: Settings = PIXANO_INFERENCE_SETTINGS,
        path: str | None = None,  # type: ignore[override]
        processor_config: dict = {},
        config: dict = {},
    ) -> VLLMModel:
        """Load a model from vLLM.

        Args:
            name: Name of the model.
            task: Task of the model.
            settings: Settings to use for the provider.
            path: Path to the model or its Hugging Face hub's identifier.
            processor_config: Configuration for the processor.
            config: Configuration for the model.

        Returns:
            Loaded model.
        """
        if path is None:
            raise ValueError("Path is required to load a model from vLLm.")
        if isinstance(task, str):
            task = str_to_task(task)

        device = settings.gpus_available[0] if settings.gpus_available else "cpu"

        our_model = VLLMModel(
            name=name, vllm_model=path, model_config=config, processor_config=processor_config, device=device
        )

        if device != "cpu":
            device = cast(int, device)
            settings.gpus_used.append(device)

        register_model(our_model, self, task)
        return our_model

    def text_image_conditional_generation(
        self, request: TextImageConditionalGenerationRequest, model: VLLMModel
    ) -> TextImageConditionalGenerationOutput:
        """Generate text from an image and a prompt.

        Args:
            request: Request for text-image conditional generation.
            model: Model for text-image conditional generation

        Returns:
            Output of text-image conditional generation.
        """
        model_input = request.to_input()
        if model_input.images is not None:
            raise ValueError("images should be passed in the prompt for vLLM.")
        if isinstance(model_input.prompt, str):
            raise ValueError("Pixano-inference only support a chat template for vLLM.")

        output = model.text_image_conditional_generation(**model_input.model_dump())
        return output
