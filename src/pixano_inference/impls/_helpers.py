# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Shared utility functions for built-in model implementations.

The torch-specific helpers (``resolve_device``, ``resolve_torch_dtype``,
``convert_image_pil_to_tensor``, ``encode_mask_to_rle``) now live in
:mod:`pixano_inference.frameworks.torch` and are re-exported here for the built-in
(torch-based) backends that import them. The remaining helpers below are numpy-only and
framework-agnostic.
"""

from __future__ import annotations

import numpy as np

# Re-exported for the built-in torch backends. Home is pixano_inference.frameworks.torch.
from pixano_inference.frameworks.torch import (
    convert_image_pil_to_tensor,
    encode_mask_to_rle,
    resolve_device,
    resolve_torch_dtype,
)


__all__ = [
    "convert_image_pil_to_tensor",
    "encode_mask_to_rle",
    "pad_points_and_labels",
    "resolve_device",
    "resolve_torch_dtype",
    "validate_prompts",
]


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
