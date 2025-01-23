# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic model for the Vector databases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, model_validator
from typing_extensions import Self

from pixano_inference.data.read_vector_databases import read_lance_vector


if TYPE_CHECKING:
    from torch import Tensor


class LanceVector(BaseModel):
    """Lance vector model.

    Attributes:
        path: The path to the LANCE dataset.
        column: The column to read.
        indice: The index of the row to read.
        where: The filter to apply. If specified, only one row should be returned.
        shape: The shape of the vector.
    """

    path: str
    column: str
    indice: int | None = None
    where: str | None = None
    shape: list[int] | None = None

    @model_validator(mode="after")
    def validate_indice_or_where(self) -> Self:
        """Validates that only one of 'indice' and 'where' is specified."""
        if self.indice is not None and self.where is not None:
            raise ValueError("Only one of 'indice' and 'where' can be specified.")
        elif self.indice is None and self.where is None:
            raise ValueError("One of 'indice' and 'where' must be specified.")
        return self

    def read_vector(self, return_type: Literal["tensor", "numpy"] = "tensor") -> "Tensor" | np.ndarray:
        """Reads a vector from a Lance dataset.

        Args:
            return_type: The type of the return value. Either 'tensor' or 'numpy'.

        Returns:
            The vector.
        """
        return read_lance_vector(self.path, self.column, self.indice, self.where, self.shape, return_type)
