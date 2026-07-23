# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for compressed and numeric RLE masks.

Canonical home: :mod:`pixano_inference_client.rle`. Re-exported here so existing
``pixano_inference.schemas.rle`` imports keep working.
"""

# ruff: noqa: F401

from pixano_inference_client.rle import CompressedRLE, mask_to_rle, rle_to_mask


__all__ = ["CompressedRLE", "mask_to_rle", "rle_to_mask"]
