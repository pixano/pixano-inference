# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""SAM2 video tracking model."""

from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pixano_inference.models.registry import register_model
from pixano_inference.models.tracking import TrackingInput, TrackingModel, TrackingOutput
from pixano_inference.ray.config import ModelDeploymentConfig

from .._helpers import resolve_device, resolve_torch_dtype, validate_prompts


logger = logging.getLogger(__name__)


@register_model("Sam2VideoModel")
class Sam2VideoModel(TrackingModel):
    """Native Ray Serve model for SAM2 video mask generation.

    ``model_params`` contract:

    - ``path`` (str, required): HuggingFace model ID or local checkpoint path.
    - ``torch_dtype`` (str, default ``"bfloat16"``): Torch dtype for autocast.
    - ``compile`` (bool, default ``True``): Whether to ``torch.compile`` the model.
    - ``vos_optimized`` (bool, default ``True``): Use VOS-optimised predictor.
    - ``propagate`` (bool, default ``True``): Whether to propagate masks across
      the full video after adding prompts.

    Any remaining keys are forwarded to the SAM2 predictor builder.
    """

    def __init__(self, config: ModelDeploymentConfig) -> None:
        """Initialize the model.

        Args:
            config: Model deployment configuration.
        """
        super().__init__(config)
        self._predictor: Any = None
        self._torch_dtype: Any = None
        self._propagate: bool = True

    def load_model(self) -> None:
        """Load the SAM2 video predictor."""
        from pixano_inference.utils.package import assert_sam2_installed

        assert_sam2_installed()

        import torch
        from sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_hf

        params = dict(self._config.model_params)
        path = params.pop("path")
        torch_dtype_str = params.pop("torch_dtype", "bfloat16")
        compile_model = params.pop("compile", True)
        vos_optimized = params.pop("vos_optimized", True)
        self._propagate = params.pop("propagate", True)

        device = resolve_device(self._config)
        self._torch_dtype = resolve_torch_dtype(torch_dtype_str)

        if path is not None and Path(path).exists():
            predictor = build_sam2_video_predictor(
                ckpt_path=path, mode="eval", device=device, vos_optimized=vos_optimized, **params
            )
        else:
            predictor = build_sam2_video_predictor_hf(
                model_id=path, mode="eval", device=device, vos_optimized=vos_optimized, **params
            )

        if compile_model:
            predictor = torch.compile(predictor)

        self._predictor = predictor
        logger.info("Sam2VideoModel '%s' loaded on %s (dtype=%s)", self.model_name, device, torch_dtype_str)

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata including path and dtype."""
        base = super().metadata
        params = self._config.model_params
        base["path"] = params.get("path")
        base["torch_dtype"] = params.get("torch_dtype", "bfloat16")
        base["propagate"] = self._propagate
        if self._predictor is not None:
            base["device"] = str(self._predictor.device)
        return base

    def predict(self, input: TrackingInput) -> TrackingOutput:
        """Run SAM2 video mask generation.

        Args:
            input: Tracking input with video, prompts, and object IDs.

        Returns:
            Tracking output with objects_ids, frame_indexes, and masks.
        """
        import torch

        from pixano_inference.schemas.rle import CompressedRLE

        objects_ids = input.objects_ids
        frame_indexes = input.frame_indexes

        if len(objects_ids) != len(frame_indexes):
            raise ValueError("objects_ids and frame_indexes must have the same length.")

        validate_prompts(input.points, input.labels, input.boxes)

        # Build per-object input arrays
        num_objects = len(objects_ids)
        if input.points is not None:
            input_points = [np.array(p, dtype=np.int32) for p in input.points]
        else:
            input_points = [None] * num_objects  # type: ignore[list-item]
        if input.labels is not None:
            input_labels = [np.array(lbl, dtype=np.int32) for lbl in input.labels]
        else:
            input_labels = [None] * num_objects  # type: ignore[list-item]
        if input.boxes is not None:
            input_boxes = [np.array(b, dtype=np.int32) for b in input.boxes]
        else:
            input_boxes = [None] * num_objects  # type: ignore[list-item]

        video_segments: dict[int, dict[int, np.ndarray]] = {}

        with torch.inference_mode():
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(self._predictor.device.type, dtype=self._torch_dtype):
                inference_state = self._init_video_state(input.video)

                for obj_id, frame_idx, obj_points, obj_labels, obj_box in zip(
                    objects_ids, frame_indexes, input_points, input_labels, input_boxes
                ):
                    _, out_obj_ids, out_mask_logits = self._predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=obj_points,
                        labels=obj_labels,
                        box=obj_box,
                    )

                    if not self._propagate:
                        video_segments[frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }

                if self._propagate:
                    video_segments = {}
                    for out_frame_idx, out_obj_ids, out_mask_logits in self._predictor.propagate_in_video(
                        inference_state
                    ):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }

        # Flatten to output lists
        out_objects_ids: list[int] = []
        out_frame_indexes: list[int] = []
        out_masks: list[CompressedRLE] = []

        for frame_index, object_masks in video_segments.items():
            for object_id, mask in object_masks.items():
                out_objects_ids.append(object_id)
                out_frame_indexes.append(frame_index)
                out_masks.append(CompressedRLE.from_mask(mask[0].astype(np.uint8)))

        return TrackingOutput(
            objects_ids=out_objects_ids,
            frame_indexes=out_frame_indexes,
            masks=out_masks,
        )

    def _init_video_state(self, video: Any) -> dict[str, Any]:
        """Initialize a SAM2 video inference state.

        Args:
            video: A list of frame path strings or a single video/directory path string.

        Returns:
            The SAM2 inference state dict.
        """
        from sam2.sam2_video_predictor import load_video_frames

        compute_device = self._predictor.device

        if isinstance(video, (str, Path)):
            path = Path(video) if isinstance(video, str) else video
            if path.is_file():
                images, video_height, video_width = load_video_frames(
                    video_path=path,
                    image_size=self._predictor.image_size,
                    offload_video_to_cpu=False,
                    async_loading_frames=False,
                    compute_device=compute_device,
                )
            elif path.is_dir():
                frames = sorted([f for f in path.glob("**/*.jpg") if f.is_file()])
                images, video_height, video_width = self._load_video_frames_from_images(
                    frames=frames,
                    image_size=self._predictor.image_size,
                    compute_device=compute_device,
                )
            else:
                raise ValueError(f"Path '{video}' is neither a file nor a directory.")
        elif isinstance(video, list):
            images, video_height, video_width = self._load_video_frames_from_images(
                frames=video,
                image_size=self._predictor.image_size,
                compute_device=compute_device,
            )
        else:
            raise ValueError(f"Unsupported video type: {type(video)}")

        inference_state: dict[str, Any] = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": video_height,
            "video_width": video_width,
            "device": compute_device,
            "storage_device": compute_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        # Warm up visual backbone on frame 0
        self._predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def _load_video_frames_from_images(
        self,
        frames: Sequence[str | Path],
        image_size: int,
        compute_device: Any,
        images_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        images_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> tuple[Any, int, int]:
        """Load and preprocess video frames from image paths.

        Args:
            frames: List of image paths or base64 strings.
            image_size: Target image size for the model.
            compute_device: Torch device for the tensors.
            images_mean: Normalisation mean per channel.
            images_std: Normalisation std per channel.

        Returns:
            Tuple of ``(tensor, video_height, video_width)``.

        Raises:
            RuntimeError: If no frames are provided.
        """
        import torch

        from pixano_inference.utils.media import convert_string_to_image

        from .._helpers import convert_image_pil_to_tensor

        num_frames = len(frames)
        if num_frames == 0:
            raise RuntimeError("No video frames provided.")

        mean = torch.tensor(images_mean, dtype=torch.float32, device=compute_device)[:, None, None]
        std = torch.tensor(images_std, dtype=torch.float32, device=compute_device)[:, None, None]

        torch_images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32, device=compute_device)
        video_height = 0
        video_width = 0
        for n, frame in enumerate(frames):
            pil_image = convert_string_to_image(frame)
            video_height = pil_image.height
            video_width = pil_image.width
            torch_images[n] = convert_image_pil_to_tensor(image=pil_image, size=image_size, device=compute_device)

        torch_images -= mean
        torch_images /= std
        return torch_images, video_height, video_width

    def unload(self) -> None:
        """Free resources."""
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
