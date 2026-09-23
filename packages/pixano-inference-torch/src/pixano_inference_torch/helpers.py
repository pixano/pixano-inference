# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""PyTorch helpers shared by the torch-based model packages.

Device/dtype resolution, image-to-tensor conversion, RLE encoding, and NDArray<->tensor
bridging. Every torch import is lazy, so importing this module (which model packages do at
plugin-discovery time) does not load torch until a helper actually runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from pixano_inference.ray.config import ModelDeploymentConfig
    from pixano_inference.schemas.nd_array import NDArray, NDArrayFloat


def resolve_torch_dtype(dtype_str: str) -> Any:
    """Map a dtype string to a ``torch.dtype``.

    Args:
        dtype_str: One of ``"float32"``, ``"float16"``, ``"bfloat16"``.

    Returns:
        Corresponding ``torch.dtype``.

    Raises:
        ValueError: If *dtype_str* is not recognised.
    """
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported torch_dtype '{dtype_str}'. Choose from {list(mapping)}")
    return mapping[dtype_str]


def resolve_device_from_num_gpus(num_gpus: float) -> Any:
    """Return ``torch.device('cuda')`` when GPUs are requested and CUDA is available.

    Falls back to Apple-Silicon MPS when requested and available, else CPU.

    Args:
        num_gpus: Number of GPUs requested for the deployment.

    Returns:
        A ``torch.device``.
    """
    import torch

    if num_gpus > 0:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def resolve_device(config: ModelDeploymentConfig) -> Any:
    """Return the torch device for a deployment config (see :func:`resolve_device_from_num_gpus`)."""
    return resolve_device_from_num_gpus(config.resources.num_gpus)


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


def convert_image_pil_to_tensor(image: Any, device: "torch.device", size: int | None = None) -> "Tensor":
    """Convert a PIL image to a ``(C, H, W)`` float tensor, optionally resizing it.

    Args:
        image: PIL image.
        device: Torch device.
        size: Optional target size (both height and width).

    Returns:
        Image as a ``(C, H, W)`` float tensor.
    """
    import torch

    image = image.convert("RGB")
    if size is not None:
        image = image.resize((size, size))
    image_np = np.array(image) / 255.0
    return torch.from_numpy(image_np).to(device=device).permute(2, 0, 1)


def encode_mask_to_rle(mask: "Tensor") -> dict[str, list[int]]:
    """Encode a binary mask tensor using RLE.

    Args:
        mask: A binary mask of shape (height, width).

    Returns:
        RLE encoded mask as a dictionary.
    """
    import torch

    rle: dict[str, Any] = {"counts": [], "size": list(mask.shape)}
    mask = mask.permute(1, 0).flatten()
    diff_arr = torch.diff(mask)
    nonzero_indices = torch.where(diff_arr != 0)[0] + 1
    lengths = torch.diff(torch.concatenate((torch.tensor([0]), nonzero_indices, torch.tensor([len(mask)]))))

    # note that the odd counts are always the numbers of zeros
    if mask[0] == 1:
        lengths = torch.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle


def ndarray_to_tensor(array: "NDArray") -> "Tensor":
    """Convert an :class:`NDArray` wire type to a torch tensor (via numpy)."""
    import torch

    return torch.from_numpy(array.to_numpy())


def tensor_to_ndarray(tensor: "Tensor") -> "NDArrayFloat":
    """Convert a torch tensor to an :class:`NDArrayFloat` wire type (via numpy)."""
    from pixano_inference.schemas.nd_array import NDArrayFloat

    return NDArrayFloat.from_numpy(tensor.detach().cpu().numpy())


def vector_to_tensor(vector: "NDArrayFloat | Tensor | None") -> "Tensor":
    """Coerce an :class:`NDArrayFloat` or tensor to a torch tensor.

    Args:
        vector: An ``NDArrayFloat`` or a torch tensor.

    Returns:
        The tensor.
    """
    from torch import Tensor

    from pixano_inference.schemas.nd_array import NDArrayFloat

    if not isinstance(vector, (NDArrayFloat, Tensor)) and vector is not None:
        raise ValueError(f"Unsupported vector type: {type(vector)}")
    if isinstance(vector, NDArrayFloat):
        return ndarray_to_tensor(vector)
    return vector
