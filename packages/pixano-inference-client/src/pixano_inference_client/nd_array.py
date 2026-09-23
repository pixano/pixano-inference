# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for N-dimensional arrays.

This is a core, framework-agnostic wire type. It converts to and from ``numpy.ndarray``
only; conversions to framework tensors live in the model packages (e.g.
``pixano_inference_torch`` for PyTorch), so the core does not privilege any one framework.

Arrays serialize as ``{shape, dtype, data}`` where ``data`` is the base64-encoded raw bytes
of the array (compact and cheap to parse, unlike a JSON list of floats — important for
SAM2-scale embeddings).
"""

import base64
from abc import ABC
from typing import ClassVar, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, field_validator
from typing_extensions import Self


T = TypeVar("T")


class NDArray(BaseModel, Generic[T], ABC):
    """Represents an N-dimensional array serialized as base64-encoded raw bytes.

    Attributes:
        shape: The shape of the array.
        dtype: The numpy dtype name (informational; the concrete subclass fixes it).
        data: Base64-encoded raw bytes of the (C-contiguous) array.
    """

    shape: list[int]
    dtype: str = ""
    data: str

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
        """Create an instance from a NumPy array (cast to the subclass dtype).

        Args:
            arr: The NumPy array to convert.

        Returns:
            An instance holding the array's shape and base64-encoded bytes.
        """
        contiguous = np.ascontiguousarray(arr.astype(cls.np_dtype))
        return cls(
            shape=list(arr.shape),
            dtype=str(cls.np_dtype),
            data=base64.b64encode(contiguous.tobytes()).decode("ascii"),
        )

    def to_numpy(self) -> np.ndarray:
        """Convert the instance to a NumPy array.

        Returns:
            A NumPy array with the stored shape and dtype.
        """
        # bytearray (rather than bytes) yields a writable array, avoiding a torch warning.
        buffer = bytearray(base64.b64decode(self.data))
        return np.frombuffer(buffer, dtype=self.np_dtype).reshape(self.shape)


class NDArrayFloat(NDArray[float]):
    """An N-dimensional array of 32-bit floating-point values."""

    np_dtype: ClassVar[np.dtype] = np.dtype(np.float32)
