# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of :mod:`pixano_inference.schemas.nd_array`."""

# ruff: noqa: F401

from pixano_inference.schemas.nd_array import (
    NDArray,
    NDArrayFloat,
)


__all__ = ["NDArray", "NDArrayFloat"]
