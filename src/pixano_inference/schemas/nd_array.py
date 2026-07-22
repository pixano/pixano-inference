# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for N-dimensional arrays.

This is a core, framework-agnostic wire type: it converts to and from ``numpy.ndarray``
only. Conversions to framework tensors (torch/jax/tf/mlx) live in
:mod:`pixano_inference.frameworks`, so the core does not privilege any one framework.
"""

from abc import ABC
from typing import ClassVar, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, field_validator
from typing_extensions import Self


T = TypeVar("T")


class NDArray(BaseModel, Generic[T], ABC):
    """Represents an N-dimensional array.

    Attributes:
        values: The list of values.
        shape: The shape of the array, represented as a list of integers.
        np_dtype: The NumPy data type of the array.
    """

    values: list[T]
    shape: list[int]
    np_dtype: ClassVar[np.dtype]

    @field_validator("shape", mode="after")
    @classmethod
    def _validate_shape(cls, v: list[int]) -> list[int]:
        if len(v) < 1:
            raise ValueError("Shape must have at least one element.")
        elif any(s < 1 for s in v):
            raise ValueError("Shape elements must be positive.")
        return v

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Self:
        """Create an instance of the class from a NumPy array.

        Args:
            arr: The NumPy array to convert.

        Returns:
            An instance of the class with values and shape derived from
                the input array.
        """
        shape = list(arr.shape)
        arr = arr.astype(dtype=cls.np_dtype)
        return cls(
            values=arr.reshape(-1).tolist(),
            shape=shape,
        )

    def to_numpy(self) -> np.ndarray:
        """Convert the instance to a NumPy array.

        Returns:
            A NumPy array with values and shape derived from the instance.
        """
        array = np.array(self.values, dtype=self.np_dtype).reshape(self.shape)
        return array


class NDArrayFloat(NDArray[float]):
    """Represents an N-dimensional array of 32-bit floating-point values.

    Attributes:
        values: The list of 32-bit floating-point values in the array.
        shape: The shape of the array, represented as a list of integers.
        np_dtype: The NumPy data type of the array.
    """

    np_dtype: ClassVar[np.dtype] = np.float32
