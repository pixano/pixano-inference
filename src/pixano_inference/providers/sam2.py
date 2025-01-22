# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Provider for the SAM2 model."""

from pathlib import Path
from typing import Any, cast

import torch

from pixano_inference import PIXANO_INFERENCE_SETTINGS
from pixano_inference.models.base import BaseInferenceModel
from pixano_inference.models.sam2 import Sam2Model
from pixano_inference.providers.registry import register_provider
from pixano_inference.pydantic.tasks.image.mask_generation import ImageMaskGenerationOutput, ImageMaskGenerationRequest
from pixano_inference.pydantic.tasks.video.mask_generation import (
    VideoMaskGenerationRequest,
    VideoMaskGenerationResponse,
)
from pixano_inference.settings import Settings
from pixano_inference.tasks.image import ImageTask
from pixano_inference.tasks.task import Task
from pixano_inference.tasks.utils import str_to_task
from pixano_inference.tasks.video import VideoTask
from pixano_inference.utils.image import convert_string_to_image
from pixano_inference.utils.package import assert_sam2_installed
from pixano_inference.utils.vector import vector_to_tensor

from .base import ModelProvider


@register_provider("sam2")
class Sam2Provider(ModelProvider):
    """Provider for the SAM2 model."""

    def load_model(
        self,
        name: str,
        task: Task | str,
        settings: Settings = PIXANO_INFERENCE_SETTINGS,
        path: Path | str | None = None,
        processor_config: dict = {},
        config: dict = {},
    ) -> Sam2Model:
        """Load the model.

        Args:
            name: Name of the model.
            task: Task of the model.
            settings: Settings to use for the provider.
            path: Path to the model.
            processor_config: Processor configuration.
            config: Configuration for the model.

        Returns:
            The loaded model.
        """
        assert_sam2_installed()
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = settings.gpus_available[0] if settings.gpus_available else "cpu"

        task = str_to_task(task) if isinstance(task, str) else task
        if task == ImageTask.MASK_GENERATION:
            model = build_sam2(ckpt_path=path, device=device, **config)
            model = model.eval()
            model = torch.compile(model)
            predictor = SAM2ImagePredictor(model)
        elif task == VideoTask.MASK_GENERATION:
            predictor = build_sam2_video_predictor(ckpt_path=path, device=device, **config)
            predictor = predictor.eval()
            predictor = torch.compile(predictor)
        else:
            raise ValueError(f"Invalid task '{task}' for the SAM2 provider.")

        if device != "cpu":
            device = cast(int, device)
            settings.gpus_used.append(device)

        return Sam2Model(
            name=name,
            provider="sam2",
            predictor=predictor,
            torch_dtype=config.get("torch_dtype", "bfloat16"),
            config=config,
        )

    @torch.inference_mode()
    def image_mask_generation(
        self, request: ImageMaskGenerationRequest, model: Sam2Model, *args: Any, **kwargs: Any
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

        if request_input.image_embedding is not None and request_input.high_resolution_features is not None:
            image_embedding = vector_to_tensor(request_input.image_embedding)
            high_resolution_features = vector_to_tensor(request_input.high_resolution_features)
            model.set_image_embeddings(image, image_embedding, high_resolution_features)
        elif request_input.image_embedding is not None or request_input.high_resolution_features is not None:
            raise ValueError("Both image_embedding and high_resolution_features must be provided.")

        model_input = request_input.model_dump(exclude=["image", "image_embedding", "high_resolution_features"])
        model_input["image"] = image
        output = model.image_mask_generation(**model_input)
        model.predictor.reset_predictor()
        return output

    @torch.inference_mode()
    def video_mask_generation(
        self, request: VideoMaskGenerationRequest, model: BaseInferenceModel, *args: Any, **kwargs: Any
    ) -> VideoMaskGenerationResponse:
        """Generate masks from the video.

        Args:
            request: Request for the generation.
            model: Model to use for the generation.
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Response of the generation.
        """
        request_input = request.to_input()
        output = model.video_mask_generation(**request_input)
        return output
