# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Shared utility functions for built-in model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pixano_inference.ray.config import ModelDeploymentConfig
from pixano_inference.utils.package import assert_torch_installed


if TYPE_CHECKING:
    import torch
    from torch import Tensor


def resolve_torch_dtype(dtype_str: str) -> Any:
    """Map a dtype string to a ``torch.dtype``.

    Args:
        dtype_str: One of ``"float32"``, ``"float16"``, ``"bfloat16"``.

    Returns:
        Corresponding ``torch.dtype``.

    Raises:
        ValueError: If *dtype_str* is not recognised.
    """
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported torch_dtype '{dtype_str}'. Choose from {list(mapping)}")
    return mapping[dtype_str]


def resolve_device(config: ModelDeploymentConfig) -> Any:
    """Return ``torch.device('cuda')`` when a GPU is requested and available, else CPU.

    Args:
        config: Model deployment configuration.

    Returns:
        A ``torch.device``.
    """
    import torch

    if config.resources.num_gpus > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_prompts(
    points: list[list[list[int]]] | None,
    labels: list[list[int]] | None,
    boxes: list[list[int]] | None,
) -> None:
    """Validate point/label/box prompts.

    Args:
        points: Point prompts ``[num_prompts, num_points, 2]``.
        labels: Label prompts ``[num_prompts, num_points]``.
        boxes: Box prompts ``[num_prompts, 4]``.

    Raises:
        ValueError: On any validation failure.
    """
    if points is None and labels is not None:
        raise ValueError("Labels are not supported without points.")
    if points is not None and labels is not None and len(points) != len(labels):
        raise ValueError("The number of points and labels should match.")
    if points is not None and boxes is not None and len(points) != len(boxes):
        raise ValueError("The number of points and boxes should match.")

    if points is not None:
        for prompt_points in points:
            for pt in prompt_points:
                if len(pt) != 2:
                    raise ValueError("Each point should have 2 coordinates.")
                if not all(isinstance(c, int) for c in pt):
                    raise ValueError("Each point coordinate should be an integer.")

    if labels is not None:
        for i, prompt_labels in enumerate(labels):
            if points is not None and len(prompt_labels) != len(points[i]):
                raise ValueError("The number of labels should match the number of points.")
            if not all(isinstance(lbl, int) for lbl in prompt_labels):
                raise ValueError("Each label should be an integer.")

    if boxes is not None:
        for box in boxes:
            if len(box) != 4:
                raise ValueError("Each box should have 4 coordinates.")
            if not all(isinstance(c, int) for c in box):
                raise ValueError("Each box coordinate should be an integer.")


def pad_points_and_labels(
    points: list[list[list[int]]],
    labels: list[list[int]] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Pad ragged point/label lists to uniform length.

    Adapted from HuggingFace's SAM processing (Apache-2.0 License).

    Args:
        points: Point prompts ``[num_prompts, variable_num_points, 2]``.
        labels: Label prompts ``[num_prompts, variable_num_points]`` or ``None``.

    Returns:
        Tuple of ``(np_points, np_labels)`` with uniform second dimension.
    """
    np_points = [np.array(p, dtype=np.int32) for p in points]
    np_labels = [np.array(lbl, dtype=np.int32) for lbl in labels] if labels is not None else None

    expected_nb_points = max(p.shape[0] for p in np_points)
    processed_points: list[np.ndarray] = []
    for i, point in enumerate(np_points):
        if point.shape[0] != expected_nb_points:
            pad_size = expected_nb_points - point.shape[0]
            point = np.concatenate([point, np.zeros((pad_size, 2)) + -10], axis=0)
            if np_labels is not None:
                np_labels[i] = np.append(np_labels[i], [-10] * pad_size)
        processed_points.append(point)

    out_points = np.array(processed_points)
    out_labels = np.array(np_labels) if np_labels is not None else None
    return out_points, out_labels


def convert_image_pil_to_tensor(image: Any, device: "torch.device", size: int | None = None) -> "Tensor":
    """Convert an image in PIL format to a PyTorch tensor and optionally resize it.

    Args:
        image: PIL image.
        device: Torch device.
        size: Optional target size (both height and width).

    Returns:
        Image as a ``(C, H, W)`` float tensor.
    """
    import torch

    assert_torch_installed()
    image = image.convert("RGB")
    if size is not None:
        image = image.resize((size, size))
    image_np = np.array(image) / 255.0
    return torch.from_numpy(image_np).to(device=device).permute(2, 0, 1)


def encode_mask_to_rle(mask: "Tensor") -> dict[str, list[int]]:
    """Encode a binary mask using RLE.

    Args:
        mask: A binary mask of shape (height, width).

    Returns:
        RLE encoded mask as a dictionary.
    """
    import torch

    assert_torch_installed()
    rle: dict[str, Any] = {"counts": [], "size": list(mask.shape)}
    mask = mask.permute(1, 0).flatten()
    diff_arr = torch.diff(mask)
    nonzero_indices = torch.where(diff_arr != 0)[0] + 1
    lengths = torch.diff(torch.concatenate((torch.tensor([0]), nonzero_indices, torch.tensor([len(mask)]))))

    # note that the odd counts are always the numbers of zeros
    if mask[0] == 1:
        lengths = torch.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle
