# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Provider for Hugging Face Transformers models."""

from pathlib import Path
from typing import Any, cast

from pixano_inference import PIXANO_INFERENCE_SETTINGS
from pixano_inference.models.transformers import TransformerModel
from pixano_inference.models_registry import register_model
from pixano_inference.providers.registry import register_provider
from pixano_inference.pydantic.tasks.image.mask_generation import ImageMaskGenerationOutput, ImageMaskGenerationRequest
from pixano_inference.pydantic.tasks.multimodal.conditional_generation import (
    TextImageConditionalGenerationOutput,
    TextImageConditionalGenerationRequest,
)
from pixano_inference.settings import Settings
from pixano_inference.tasks import ImageTask, MultimodalImageNLPTask, NLPTask, Task, str_to_task
from pixano_inference.utils.image import convert_string_to_image
from pixano_inference.utils.package import (
    assert_transformers_installed,
    is_torch_installed,
    is_transformers_installed,
)
from pixano_inference.utils.vector import vector_to_tensor

from .base import ModelProvider


if is_torch_installed():
    import torch

if is_transformers_installed():
    from transformers import AutoProcessor
    from transformers.modeling_utils import PreTrainedModel


def get_transformer_automodel_from_pretrained(
    pretrained_model_name_or_path: str | Path, task: Task, **model_kwargs: Any
):
    """Get a transformer model from transformers using automodel.

    Args:
        pretrained_model_name_or_path: Name or path of the pretrained model.
        task: Task of the model.
        model_kwargs: Additional keyword arguments for the model.
    """
    assert_transformers_installed()
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if isinstance(task, ImageTask):
        match task:
            case ImageTask.CLASSIFICATION:
                from transformers import AutoModelForImageClassification

                return AutoModelForImageClassification.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case ImageTask.DEPTH_ESTIMATION:
                from transformers import AutoModelForDepthEstimation

                return AutoModelForDepthEstimation.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case ImageTask.FEATURE_EXTRACTION:
                from transformers import AutoModel

                return AutoModel.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case ImageTask.KEYPOINT_DETECTION:
                from transformers import AutoModelForKeypointDetection

                return AutoModelForKeypointDetection.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case ImageTask.MASK_GENERATION:
                from transformers import AutoModelForMaskGeneration

                return AutoModelForMaskGeneration.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case ImageTask.OBJECT_DETECTION:
                from transformers import AutoModelForObjectDetection

                return AutoModelForObjectDetection.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case ImageTask.SEMANTIC_SEGMENTATION:
                from transformers import AutoModelForSemanticSegmentation

                return AutoModelForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case ImageTask.INSTANCE_SEGMENTATION:
                from transformers import AutoModelForInstanceSegmentation

                return AutoModelForInstanceSegmentation.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case ImageTask.UNIVERSAL_SEGMENTATION:
                from transformers import AutoModelForUniversalSegmentation

                return AutoModelForUniversalSegmentation.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case ImageTask.ZERO_SHOT_CLASSIFICATION:
                from transformers import AutoModelForZeroShotImageClassification

                return AutoModelForZeroShotImageClassification.from_pretrained(
                    pretrained_model_name_or_path, **model_kwargs
                )
            case ImageTask.ZERO_SHOT_DETECTION:
                from transformers import AutoModelForZeroShotObjectDetection

                return AutoModelForZeroShotObjectDetection.from_pretrained(
                    pretrained_model_name_or_path, **model_kwargs
                )
    elif isinstance(task, NLPTask):
        match task:
            case NLPTask.CAUSAL_LM:
                from transformers import AutoModelForCausalLM

                return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case NLPTask.MASKED_LM:
                from transformers import AutoModelForMaskedLM

                return AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case NLPTask.MASK_GENERATION:
                from transformers import AutoModelForMaskGeneration

                return AutoModelForMaskGeneration.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case NLPTask.SEQUENCE_CLASSIFICATION:
                from transformers import AutoModelForSequenceClassification

                return AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model_name_or_path, **model_kwargs
                )
            case NLPTask.MULTIPLE_CHOICE:
                from transformers import AutoModelForMultipleChoice

                return AutoModelForMultipleChoice.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case NLPTask.NEXT_SENTENCE_PREDICTION:
                from transformers import AutoModelForNextSentencePrediction

                return AutoModelForNextSentencePrediction.from_pretrained(
                    pretrained_model_name_or_path, **model_kwargs
                )
            case NLPTask.TOKEN_CLASSIFICATION:
                from transformers import AutoModelForTokenClassification

                return AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case NLPTask.QUESTION_ANSWERING:
                from transformers import AutoModelForQuestionAnswering

                return AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case NLPTask.TEXT_ENCODING:
                from transformers import AutoModelForTextEncoding

                return AutoModelForTextEncoding.from_pretrained(pretrained_model_name_or_path, **model_kwargs)
            case _:
                raise ValueError(f"Task {task} not supported.")


def get_conditional_generation_transformer_from_pretrained(
    name: str, path: Path | str | None, **model_kwargs: Any
) -> PreTrainedModel:
    """Get a transformer model from transformers using automodel.

    Args:
        name: Name of the model.
        path: Path to the model or its Hugging Face hub's identifier.
        model_kwargs: Additional keyword arguments for the model.

    Returns:
        Model from Transformers.
    """
    name = name.lower()
    if "llava" in name:
        if "next" in name:
            if "video" in name:
                from transformers import LlavaNextVideoForConditionalGeneration

                model = LlavaNextVideoForConditionalGeneration.from_pretrained(path, **model_kwargs)
            else:
                from transformers import LlavaNextForConditionalGeneration

                model = LlavaNextForConditionalGeneration.from_pretrained(path, **model_kwargs)
        else:
            from transformers import LlavaForConditionalGeneration

            model = LlavaForConditionalGeneration.from_pretrained(path, **model_kwargs)
    else:
        raise ValueError(f"Model {name} not supported.")
    return model


@register_provider("transformers")
class TransformersProvider(ModelProvider):
    """Provider for Hugging Face Transformers models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the transformer provider."""
        assert_transformers_installed()
        super().__init__(*args, **kwargs)

    def load_model(
        self,
        name: str,
        task: Task | str,
        settings: Settings = PIXANO_INFERENCE_SETTINGS,
        path: Path | str | None = None,
        processor_config: dict = {},
        config: dict = {},
    ) -> TransformerModel:
        """Load a model from transformers.

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
            raise ValueError("Path is required to load a model from transformers.")
        if isinstance(task, str):
            task = str_to_task(task)
        processor = AutoProcessor.from_pretrained(path, **processor_config)

        device = settings.gpus_available[0] if settings.gpus_available else "cpu"

        model = get_transformer_automodel_from_pretrained(path, task, device_map=device, **config)
        if model is None:
            if task in [NLPTask.CONDITONAL_GENERATION, MultimodalImageNLPTask.CONDITIONAL_GENERATION]:
                model = get_conditional_generation_transformer_from_pretrained(name, path, device_map=device, **config)

        if device != "cpu":
            device = cast(int, device)
            settings.gpus_used.append(device)

        model = model.eval()
        model = torch.compile(model)

        our_model = TransformerModel(name, path, processor, model)
        register_model(our_model, self, task)
        return model

    @torch.inference_mode()
    def image_mask_generation(
        self, request: ImageMaskGenerationRequest, model: TransformerModel, *args: Any, **kwargs: Any
    ) -> ImageMaskGenerationOutput:
        """Generate a mask from the image.

        Args:
            request: Request for the generation.
            model: Model to use for the generation.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Output of the generation
        """
        request_input = request.to_input()
        image = convert_string_to_image(request_input.image)

        if request_input.image_embedding is not None:
            image_embedding = vector_to_tensor(request_input.image_embedding)

        model_input = request_input.model_dump(exclude=["image", "image_embedding"])
        model_input["image"] = image
        model_input["image_embedding"] = image_embedding if request_input.image_embedding is not None else None
        output = model.image_mask_generation(**model_input)
        return output

    @torch.inference_mode()
    def text_image_conditional_generation(
        self, request: TextImageConditionalGenerationRequest, model: TransformerModel
    ) -> TextImageConditionalGenerationOutput:
        """Generate text from an image and a prompt.

        Args:
            request: Request for text-image conditional generation.
            model: Model for text-image conditional generation

        Returns:
            Output of text-image conditional generation.
        """
        model_input = request.to_input().model_dump()
        model_input["image"] = convert_string_to_image(model_input["image"])
        output = model.text_image_conditional_generation(**model_input)
        return output
