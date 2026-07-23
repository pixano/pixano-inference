# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Torch helpers re-exported for the built-in torch backends.

The implementations live in :mod:`pixano_inference.frameworks.torch`; they are re-exported
here for the built-in (transformers-based) backends that import them.
"""

from __future__ import annotations

from pixano_inference.frameworks.torch import (
    convert_image_pil_to_tensor,
    encode_mask_to_rle,
    resolve_device,
    resolve_torch_dtype,
)


__all__ = [
    "convert_image_pil_to_tensor",
    "encode_mask_to_rle",
    "resolve_device",
    "resolve_torch_dtype",
]
