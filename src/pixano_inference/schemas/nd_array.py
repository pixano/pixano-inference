# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for N-dimensional arrays.

Canonical home: :mod:`pixano_inference_client.nd_array`. Re-exported here so existing
``pixano_inference.schemas.nd_array`` imports keep working. Conversions to framework tensors
(torch/jax/tf/mlx) live in :mod:`pixano_inference.frameworks`.
"""

# ruff: noqa: F401

from pixano_inference_client.nd_array import NDArray, NDArrayFloat


__all__ = ["NDArray", "NDArrayFloat"]
