# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Utility functions for vector operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pixano_inference.utils.package import assert_torch_installed


if TYPE_CHECKING:
    from torch import Tensor

    from pixano_inference.schemas.nd_array import NDArrayFloat


def vector_to_tensor(vector: NDArrayFloat | Tensor | None) -> Tensor:
    """Convert a vector to a tensor.

    Args:
        vector: Vector to convert.
    """
    from torch import Tensor

    from pixano_inference.schemas.nd_array import NDArrayFloat

    assert_torch_installed()
    if not isinstance(vector, (NDArrayFloat, Tensor)) and vector is not None:
        raise ValueError(f"Unsupported vector type: {type(vector)}")

    if isinstance(vector, NDArrayFloat):
        return vector.to_torch()
    return vector
