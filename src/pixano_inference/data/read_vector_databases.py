# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""This module contains functions for reading data from Vector databases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from pixano_inference.utils.package import (
    assert_lance_installed,
    assert_torch_installed,
    is_lance_installed,
    is_torch_installed,
)


if is_lance_installed():
    from lance import LanceDataset

if is_torch_installed():
    import torch


if TYPE_CHECKING:
    from torch import Tensor


def read_lance_vector(
    path: Path | str,
    column: str,
    indice: int | None = None,
    where: str | None = None,
    shape: list[int] | None = None,
    return_type: Literal["tensor", "numpy"] = "tensor",
) -> "Tensor" | np.ndarray:
    """Reads a vector from a Lance dataset.

    Args:
        path: The path to the Lance dataset.
        column: The vector column to read.
        indice: The index of the row to read.
        where: The filter to apply. If specified, only one row should be returned.
        shape: The shape of the vector.
        return_type: The type of the return value. Either 'tensor' or 'numpy'.

    Returns:
        The value of the column.
    """
    assert_lance_installed()
    if return_type not in ["tensor", "numpy"]:
        raise ValueError("return_type must be either 'tensor' or 'numpy'.")
    elif return_type == "tensor":
        assert_torch_installed()
    if indice is not None and where is not None:
        raise ValueError("Only one of 'index' and 'where' can be specified.")
    elif indice is None and where is None:
        raise ValueError("One of 'index' and 'where' must be specified.")
    elif indice is not None and not isinstance(indice, int):
        raise ValueError("index must be an integer.")
    elif where is not None and not isinstance(where, str):
        raise ValueError("where must be a string.")
    elif not isinstance(path, (Path, str)):
        raise ValueError("lance_path must be a Path or a string.")
    elif shape is not None and not isinstance(shape, list) and not all(isinstance(i, int) for i in shape):
        raise ValueError("shape must be a list of integers.")

    if isinstance(path, str):
        path = Path(path)

    lance_dataset = LanceDataset(path)
    if indice is not None:
        pa_table = lance_dataset.take(indices=[indice], columns=[column])
    else:
        pa_table = lance_dataset.scanner(columns=[column], filter=where).to_table()
    polar_df = pl.DataFrame(pa_table)
    if polar_df.shape[0] != 1:
        raise ValueError(f"Expected one row to be returned but found {polar_df.shape[0]} rows.")
    vector: np.ndarray = polar_df[column].to_numpy()
    vector = vector.reshape(shape)
    if return_type == "tensor":
        return torch.from_numpy(vector)
    return vector
