# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Provider for the SAM2 model."""

from pathlib import Path
from typing import Any

from pixano_inference.models.sam2 import Sam2Model
from pixano_inference.providers.registry import register_provider
from pixano_inference.pydantic.tasks.image.mask_generation import ImageMaskGenerationOutput, ImageMaskGenerationRequest
from pixano_inference.pydantic.tasks.video.mask_generation import (
    VideoMaskGenerationRequest,
    VideoMaskGenerationResponse,
)
from pixano_inference.tasks import ImageTask, Task, VideoTask, str_to_task
from pixano_inference.utils import (
    assert_sam2_installed,
    convert_string_to_image,
    convert_string_video_to_bytes_or_path,
    is_sam2_installed,
    is_torch_installed,
    vector_to_tensor,
)

from .base import ModelProvider


if is_torch_installed():
    import torch

if is_sam2_installed():
    from sam2.build_sam import build_sam2, build_sam2_hf, build_sam2_video_predictor, build_sam2_video_predictor_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor


@register_provider("sam2")
class Sam2Provider(ModelProvider):
    """Provider for the SAM2 model."""

    def __init__(self, **kwargs):
        """Initialize the SAM2 provider."""
        assert_sam2_installed()
        super().__init__(**kwargs)

    def load_model(
        self,
        name: str,
        task: Task | str,
        device: "torch.device",
        path: Path | str | None = None,
        processor_config: dict = {},
        config: dict = {},
    ) -> Sam2Model:
        """Load the model.

        Args:
            name: Name of the model.
            task: Task of the model.
            device: Device to use for the model.
            path: Path to the model.
            processor_config: Processor configuration.
            config: Configuration for the model.

        Returns:
            The loaded model.
        """
        task = str_to_task(task) if isinstance(task, str) else task
        if task == ImageTask.MASK_GENERATION:
            if path is not None and Path(path).exists():
                model = build_sam2(ckpt_path=path, mode="eval", device=device, **config)
            else:
                model = build_sam2_hf(model_id=path, mode="eval", device=device, **config)
            model = torch.compile(model)
            predictor = SAM2ImagePredictor(model)
        elif task == VideoTask.MASK_GENERATION:
            if path is not None and Path(path).exists():
                predictor = build_sam2_video_predictor(
                    ckpt_path=path, mode="eval", device=device, vos_optimized=True, **config
                )
            else:
                predictor = build_sam2_video_predictor_hf(
                    model_id=path, mode="eval", device=device, vos_optimized=True, **config
                )
            predictor = torch.compile(predictor)
        else:
            raise ValueError(f"Invalid task '{task}' for the SAM2 provider.")

        our_model = Sam2Model(
            name=name,
            provider="sam2",
            predictor=predictor,
            torch_dtype=config.get("torch_dtype", "bfloat16"),
            config=config,
        )

        return our_model

    def image_mask_generation(
        self,
        request: ImageMaskGenerationRequest,
        model: Sam2Model,  # type: ignore[override]
        *args: Any,
        **kwargs: Any,
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
        if request_input.reset_predictor:
            model.predictor.reset_predictor()
            # if new predictor, we may use given embeddings
            if request_input.image_embedding is not None and request_input.high_resolution_features is not None:
                image_embedding = vector_to_tensor(request_input.image_embedding)
                high_resolution_features = [vector_to_tensor(v) for v in request_input.high_resolution_features]
                model.set_image_embeddings(image, image_embedding, high_resolution_features)
            elif request_input.image_embedding is not None or request_input.high_resolution_features is not None:
                raise ValueError("Both image_embedding and high_resolution_features must be provided.")

        model_input = request_input.model_dump(exclude=["image", "image_embedding", "high_resolution_features"])
        model_input["image"] = image
        return model.image_mask_generation(**model_input)

    def video_mask_generation(
        self,
        request: VideoMaskGenerationRequest,
        model: Sam2Model,  # type: ignore[override]
        *args: Any,
        **kwargs: Any,
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
        request_input = request.to_input().model_dump()
        request_input["video"] = convert_string_video_to_bytes_or_path(request_input["video"])
        output = model.video_mask_generation(**request_input)
        return output
