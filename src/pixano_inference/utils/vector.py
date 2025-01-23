# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Utility functions for vector operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pixano_inference.pydantic.data.vector_database import LanceVector
from pixano_inference.pydantic.nd_array import NDArrayFloat
from pixano_inference.utils.package import assert_torch_installed, is_torch_installed


if is_torch_installed():
    from torch import Tensor

if TYPE_CHECKING:
    from torch import Tensor


def vector_to_tensor(vector: NDArrayFloat | LanceVector | "Tensor" | None) -> "Tensor":
    """Convert a vector to a tensor.

    Args:
        vector: Vector to convert.
    """
    assert_torch_installed()
    if not isinstance(vector, (NDArrayFloat, LanceVector, Tensor)) and vector is not None:
        raise ValueError(f"Unsupported vector type: {type(vector)}")

    if isinstance(vector, LanceVector):
        return vector.read_vector()
    elif isinstance(vector, NDArrayFloat):
        return vector.to_torch()
    return vector
