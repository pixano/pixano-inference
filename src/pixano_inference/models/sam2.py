# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Inference model for the SAM2 model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL.Image import Image

from pixano_inference.models_registry import unregister_model
from pixano_inference.pydantic.nd_array import NDArrayFloat
from pixano_inference.pydantic.tasks.image.mask_generation import ImageMaskGenerationOutput
from pixano_inference.pydantic.tasks.image.utils import RLEMask
from pixano_inference.pydantic.tasks.video.mask_generation import VideoMaskGenerationOutput
from pixano_inference.utils.image import encode_mask_to_rle
from pixano_inference.utils.package import assert_sam2_installed, is_sam2_installed, is_torch_installed

from .base import BaseInferenceModel


if is_torch_installed():
    import torch

if is_sam2_installed():
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor

if TYPE_CHECKING:
    from torch import Tensor


class Sam2Model(BaseInferenceModel):
    """Inference model for the SAM2 model."""

    def __init__(
        self,
        name: str,
        provider: str,
        predictor: Any,
        torch_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16",
        config: dict[str, Any] = {},
    ):
        """Initialize the model.

        Args:
            name: Name of the model.
            provider: Provider of the model.
            predictor: The SAM2 image predictor.
            torch_dtype: The torch data type to use for inference.
            config: Configuration for the model.
        """
        assert_sam2_installed()

        super().__init__(name, provider)
        match torch_dtype:
            case "float32":
                self.torch_dtype = torch.float32
            case "float16":
                self.torch_dtype = torch.float16
            case "bfloat16":
                self.torch_dtype = torch.bfloat16
            case _:
                raise ValueError(f"Invalid torch_dtype: {torch_dtype}")
        self.predictor: SAM2ImagePredictor | SAM2VideoPredictor = predictor
        self.config = config

    def delete(self) -> None:
        """Delete the model."""
        del self.predictor
        unregister_model(self)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the metadata of the model."""
        return {
            "name": self.name,
            "provider": self.provider,
            "torch_dtype": self.torch_dtype,
            "config": self.config,
        }

    def set_image_embeddings(
        self,
        image: np.ndarray | "Tensor" | Image,
        image_embedding: Tensor,
        high_resolution_features: Tensor,
    ) -> None:
        """Calculates the image embeddings for the provided image.

        Adapted from https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py
        (Apache-2.0 License).

        Args:
            image: The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC
                format if PIL Image with or CHW format if torch.Tensor.
            image_embedding: The image embedding tensor.
            high_resolution_features: The high-resolution features tensor.
        """
        with torch.inference_mode():
            self.predictor.reset_predictor()
            # Transform the image to the form expected by the model
            if isinstance(image, np.ndarray):
                self.predictor._orig_hw = [image.shape[:2]]
            elif isinstance(image, Tensor):
                self.predictor._orig_hw = [image.shape[-2:]]
            elif isinstance(image, Image):
                w, h = image.size
                self.predictor._orig_hw = [(h, w)]
            else:
                raise NotImplementedError("Image format not supported")

            self.predictor._features = {
                "image_embed": image_embedding.unsqueeze(0),
                "high_res_features": high_resolution_features.unsqueeze(0),
            }
            self.predictor._is_image_set = True

    def image_mask_generation(
        self,
        image: np.ndarray | Image,
        points: list[list[list[int]]] | None,
        labels: list[list[int]] | None,
        boxes: list[list[int]] | None,
        multimask_output: bool = True,
        num_multimask_outputs: int = 3,
        return_image_embedding: bool = False,
        **kwargs: Any,
    ) -> ImageMaskGenerationOutput:
        """Generate masks from the image.

        Args:
            image: Image for the generation.
            points: Points for the mask generation. The first dimension is the number of prompts, the
                second the number of points per mask and the third the coordinates of the points.
            labels: Labels for the mask generation. The first dimension is the number of prompts, the second
                the number of labels per mask.
            boxes: Boxes for the mask generation. The first dimension is the number of prompts, the second
                the coordinates of the boxes.
            multimask_output: Whether to generate multiple masks per prediction.
            num_multimask_outputs: Number of masks to generate per prediction.
            return_image_embedding: Whether to return the image embedding and high-resolution features.
            kwargs: Additional keyword arguments.
        """
        with torch.inference_mode():
            # Check the input list types
            if not isinstance(image, (np.ndarray, Image)):
                raise ValueError("The image should be an numpy array or a PIL image.")
            if (
                points is not None
                and not isinstance(points, list)
                and not all(isinstance(point, list) for point in points)
            ):
                raise ValueError("The points should be a list of lists.")
            if (
                labels is not None
                and not isinstance(labels, list)
                and not all(isinstance(label, list) for label in labels)
            ):
                raise ValueError("The labels should be a list of lists.")
            if boxes is not None and not isinstance(boxes, list):
                if not all(isinstance(box, list) for box in boxes):
                    raise ValueError("The boxes should be a list of lists.")

            if multimask_output and not (num_multimask_outputs == 3):
                raise ValueError("The number of multimask outputs is not configurable for Sam2 and must be 3.")

            # Check the input batch size
            if points is None and labels is not None:
                raise ValueError("Labels are not supported without points.")
            if points is not None and labels is not None:
                if len(points) != len(labels):
                    raise ValueError("The number of points and labels should match.")
            if points is not None and boxes is not None:
                if len(points) != len(boxes):
                    raise ValueError("The number of points and boxes should match.")

            # Check the input shapes and value types
            if points is not None:
                for prompt_point in points:
                    for points_in_mask in prompt_point:
                        if len(points_in_mask) != 2:
                            raise ValueError("Each point should have 2 coordinates.")
                        if not all(isinstance(point, int) for point in points_in_mask):
                            raise ValueError("Each point should be an integer.")
            if labels is not None:
                for i, prompt_label in enumerate(labels):
                    if len(prompt_label) != len(points[i]):  # type: ignore[index]
                        raise ValueError("The number of labels should match the number of points.")

                    if not all(isinstance(label, int) for label in prompt_label):
                        raise ValueError("Each label should be an integer.")
            if boxes is not None:
                for prompt_box in boxes:
                    if len(prompt_box) != 4:
                        raise ValueError("Each box should have 4 coordinates.")
                    if not all(isinstance(box, int) for box in prompt_box):
                        raise ValueError("Each box should be an integer.")

            # Convert inputs to numpy arrays
            if points is not None:
                np_points = [np.array(prompt_points, dtype=np.int32) for prompt_points in points]
            if labels is not None:
                np_labels = [np.array(prompt_labels, dtype=np.int32) for prompt_labels in labels]
            if boxes is not None:
                boxes = np.array(boxes, dtype=np.int32)

            if points is not None:
                # Pad the points and labels to the same length
                # =============================================================================
                # From Hugging Face's implementation (Apache-2.0 License):
                # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/sam/processing_sam.py#L36
                expected_nb_points = max([point.shape[0] for point in np_points])
                processed_points = []
                for i, point in enumerate(np_points):
                    if point.shape[0] != expected_nb_points:
                        point = np.concatenate(
                            [point, np.zeros((expected_nb_points - point.shape[0], 2)) + -10], axis=0
                        )
                        if labels is not None:
                            np_labels[i] = np.append(np_labels[i], [-10])
                    processed_points.append(point)
                # =============================================================================
                np_points = np.array(processed_points)

            input_points = np_points if points is not None else None
            input_labels = np.array(np_labels) if labels is not None else None

            with torch.autocast(self.predictor.device.type, dtype=self.torch_dtype):
                if not self.predictor._is_image_set:
                    self.predictor.set_image(image)

                masks, scores, _ = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    mask_input=None,
                    multimask_output=multimask_output,
                    return_logits=False,
                )

                if len(masks.shape) == 3:
                    masks = np.expand_dims(masks, 0)
                    scores = np.expand_dims(scores, 0)

                masks = torch.from_numpy(masks)

                return ImageMaskGenerationOutput(
                    masks=[
                        [RLEMask(**encode_mask_to_rle(mask)) for mask in prediction_masks]
                        for prediction_masks in masks
                    ],
                    scores=NDArrayFloat.from_numpy(scores),
                    image_embedding=self.predictor._features["image_embed"][0].cpu()
                    if return_image_embedding
                    else None,
                    high_resolution_features=self.predictor._features["high_res_features"][0].cpu()
                    if return_image_embedding
                    else None,
                )

    def video_mask_generation(
        self,
        video_dir: Path,
        objects_ids: list[int],
        frame_indexes: list[int],
        points: list[list[list[int]]] | None = None,
        labels: list[list[int]] | None = None,
        boxes: list[list[int]] | None = None,
        propagate: bool = False,
        **kwargs: Any,
    ) -> VideoMaskGenerationOutput:
        """Generate masks from the video.

        Args:
            video_dir: Directory of the video.
            objects_ids: IDs of the objects to generate masks for.
            frame_indexes: Indexes of the frames where the objects are located.
            points: Points for the mask generation. The first fimension is the number of objects, the
                second the number of points for each object and the third the coordinates of the points.
            labels: Labels for the mask generation. The first fimension is the number of objects, the second
                the number of labels for each object.
            boxes: Boxes for the mask generation. The first fimension is the number of objects, the second
                the coordinates of the boxes.
            propagate: Whether to propagate the masks in the video.
            kwargs: Additional keyword arguments.

        Returns:
            Output of the generation.
        """
        with torch.inference_mode():
            # Check the input list types
            if not isinstance(video_dir, (Path)) or not video_dir.exists():
                raise ValueError("The video_dir should be a valid path.")
            if (
                points is not None
                and not isinstance(points, list)
                and not all(isinstance(point, list) for point in points)
            ):
                raise ValueError("The points should be a list of lists.")
            if (
                labels is not None
                and not isinstance(labels, list)
                and not all(isinstance(label, list) for label in labels)
            ):
                raise ValueError("The labels should be a list of lists.")
            if boxes is not None and not isinstance(boxes, list):
                if not all(isinstance(box, list) for box in boxes):
                    raise ValueError("The boxes should be a list of lists.")

            # Check the input batch size
            if points is None and labels is not None:
                raise ValueError("Labels are not supported without points.")
            if points is not None and labels is not None:
                if len(points) != len(labels):
                    raise ValueError("The number of points and labels should match.")
            if points is not None and boxes is not None:
                if len(points) != len(boxes):
                    raise ValueError("The number of points and boxes should match.")

            # Check the input shapes and value types
            if points is not None:
                for prompt_point in points:
                    for points_in_mask in prompt_point:
                        if len(points_in_mask) != 2:
                            raise ValueError("Each point should have 2 coordinates.")
                        if not all(isinstance(point, int) for point in points_in_mask):
                            raise ValueError("Each point should be an integer.")
            if labels is not None:
                for i, prompt_label in enumerate(labels):
                    if len(prompt_label) != len(points[i]):  # type: ignore[index]
                        raise ValueError("The number of labels should match the number of points.")

                    if not all(isinstance(label, int) for label in prompt_label):
                        raise ValueError("Each label should be an integer.")
            if boxes is not None:
                for prompt_box in boxes:
                    if len(prompt_box) != 4:
                        raise ValueError("Each box should have 4 coordinates.")
                    if not all(isinstance(box, int) for box in prompt_box):
                        raise ValueError("Each box should be an integer.")

            # Convert inputs to numpy arrays
            if points is not None:
                input_points = [np.array(prompt_points, dtype=np.int32) for prompt_points in points]
            else:
                input_points = [None for _ in objects_ids]
            if labels is not None:
                input_labels = [np.array(prompt_labels, dtype=np.int32) for prompt_labels in labels]
            else:
                input_labels = [None for _ in objects_ids]
            if boxes is not None:
                input_boxes = [np.array(box, dtype=np.int32) for box in boxes]
            else:
                input_boxes = [None for _ in objects_ids]

            video_segments: dict[int, dict[int, np.ndarray]] = {}
            with torch.autocast(self.predictor.device.type, dtype=self.torch_dtype):
                inference_state = self.predictor.init_state(video_path=str(video_dir))
                for object_id, object_frame_index, object_points, object_labels, object_boxes in zip(
                    objects_ids, frame_indexes, input_points, input_labels, input_boxes
                ):
                    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=object_frame_index,
                        obj_id=object_id,
                        points=object_points,
                        labels=object_labels,
                        box=object_boxes,
                    )

                    if not propagate:
                        video_segments[object_frame_index] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }

                if propagate:
                    video_segments = {}  # video_segments contains the per-frame segmentation results
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                        inference_state
                    ):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }

            out_objects_ids = []
            out_frame_indexes = []
            out_masks = []

            for frame_index, object_masks in video_segments.items():
                for object_id, mask in object_masks.items():
                    out_objects_ids.append(object_id)
                    out_frame_indexes.append(frame_index)
                    out_masks.append(mask)

            return VideoMaskGenerationOutput(
                objects_ids=out_objects_ids,
                frame_indexes=out_frame_indexes,
                masks=[RLEMask(**encode_mask_to_rle(torch.from_numpy(mask).squeeze(0))) for mask in out_masks],
            )
