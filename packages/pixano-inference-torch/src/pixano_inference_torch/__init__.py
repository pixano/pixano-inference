# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""PyTorch helpers for Pixano Inference model packages.

The Pixano Inference core is framework-free; torch-based model packages (SAM2, CLIP,
Grounding DINO, ...) depend on this package for the device/dtype/tensor helpers they share.
"""

from .helpers import (
    convert_image_pil_to_tensor,
    encode_mask_to_rle,
    ndarray_to_tensor,
    resolve_device,
    resolve_device_from_num_gpus,
    resolve_torch_dtype,
    should_compile,
    tensor_to_ndarray,
    vector_to_tensor,
)


__all__ = [
    "convert_image_pil_to_tensor",
    "encode_mask_to_rle",
    "ndarray_to_tensor",
    "resolve_device",
    "resolve_device_from_num_gpus",
    "resolve_torch_dtype",
    "should_compile",
    "tensor_to_ndarray",
    "vector_to_tensor",
]
