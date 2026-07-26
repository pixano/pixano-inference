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

from typing import Any

from pixano_inference.frameworks.torch import (
    convert_image_pil_to_tensor,
    encode_mask_to_rle,
    resolve_device,
    resolve_torch_dtype,
)


def should_compile(device: Any, requested: bool | None) -> bool:
    """Decide whether to ``torch.compile`` a model.

    ``torch.compile`` needs a working compiler and only pays off on GPU; on CPU it is often a
    slow no-win and can fail outright. So honour an explicit ``compile`` param when given, else
    auto-detect: compile only on CUDA devices.

    Args:
        device: The resolved torch device (or a device string).
        requested: The user's ``compile`` param, or ``None`` for auto.

    Returns:
        Whether to compile the model.
    """
    if requested is not None:
        return bool(requested)
    device_type = getattr(device, "type", None) or str(device)
    return str(device_type).startswith("cuda")


__all__ = [
    "convert_image_pil_to_tensor",
    "encode_mask_to_rle",
    "resolve_device",
    "resolve_torch_dtype",
    "should_compile",
]
