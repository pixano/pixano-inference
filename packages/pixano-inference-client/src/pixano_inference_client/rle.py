# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of :mod:`pixano_inference.schemas.rle`."""

# ruff: noqa: F401

from pixano_inference.schemas.rle import (
    CompressedRLE,
    mask_to_rle,
    rle_to_mask,
)


__all__ = ["CompressedRLE", "mask_to_rle", "rle_to_mask"]
