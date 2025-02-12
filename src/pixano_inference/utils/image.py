# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Image utilities."""

import base64
import re
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import requests  # type: ignore[import-untyped]
from PIL import Image

from .package import assert_torch_installed, is_torch_installed
from .url import is_url


if is_torch_installed():
    import torch

if TYPE_CHECKING:
    from torch import Tensor

regex_base64 = r"^(data:image/[a-zA-Z]+;base64,)"


def is_base64_image(image: str) -> bool:
    """Check if a string is a base64 image.

    The expected format is "data:image/{image_format};base64,{base64}".
    """
    match = re.match(regex_base64, image)
    return match is not None


def extract_image_from_base64(image: str):
    """Extract from a base64 image the actual base64 part."""
    match = re.match(regex_base64, image)
    if match is None:
        return ""
    return image[len(match.group(1)) :]


def convert_string_to_image(str_image: str | Path) -> Image.Image:
    """Convert a string or path to an image.

    Args:
        str_image: Image as a string or path.

    Returns:
        Image.
    """
    if isinstance(str_image, str):
        if is_url(str_image):
            image_pil = Image.open(requests.get(str_image, stream=True).raw)
        else:
            if is_base64_image(str_image):
                image_bytes = base64.b64decode(extract_image_from_base64(str_image))
                image_pil = Image.open(BytesIO(image_bytes))
            elif Path(str_image).exists():
                image_pil = Image.open(str_image)
            else:
                raise ValueError("The image is not a valid path, URL or base64 string.")
    elif isinstance(str_image, Path):
        image_pil = Image.open(str_image)
    else:
        raise ValueError("The image is not a valid path, URL or base64 string.")
    image_converted = image_pil.convert("RGB")
    return image_converted


def encode_mask_to_rle(mask: "Tensor") -> dict[str, list[int]]:
    """Encode a binary mask using RLE.

    Args:
        mask: A binary mask of shape (height, width).

    Returns:
        RLE encoded mask as a dictionary.
    """
    assert_torch_installed()
    rle = {"counts": [], "size": list(mask.shape)}
    mask = mask.permute(1, 0).flatten()
    diff_arr = torch.diff(mask)
    nonzero_indices = torch.where(diff_arr != 0)[0] + 1
    lengths = torch.diff(torch.concatenate((torch.tensor([0]), nonzero_indices, torch.tensor([len(mask)]))))

    # note that the odd counts are always the numbers of zeros
    if mask[0] == 1:
        lengths = torch.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle


def compress_rle(rle: dict[str, Any]) -> dict[str, Any | str]:
    """Compress an RLE encoded mask.

    Args:
        rle: RLE encoded mask as a dictionary.

    Returns:
        Compressed RLE encoded mask as a string.
    """
    counts = np.array(rle["counts"], dtype=np.uint32).tobytes()
    rle["counts"] = base64.b64encode(counts).decode("utf-8")
    return rle


def decode_rle_to_mask(rle: dict) -> np.ndarray:
    """Decode an RLE encoded mask.

    Args:
        rle: RLE encoded mask as a dictionary.

    Returns:
        Decoded binary mask of shape (height, width).
    """
    height, width = rle["size"]
    mask = np.empty(height * width, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity = not parity
    mask = mask.reshape(width, height)
    return mask.transpose()  # Reshape to original shape


def decompress_rle(rle: dict[str, Any]) -> dict[str, Any]:
    """Decompress a compressed RLE encoded mask.

    Args:
        rle: Compressed RLE encoded mask as a string.

    Returns:
        Decompressed RLE encoded mask as a dictionary.
    """
    rle["counts"] = np.frombuffer(base64.b64decode(rle["counts"]), dtype=np.uint32).tolist()
    return rle
