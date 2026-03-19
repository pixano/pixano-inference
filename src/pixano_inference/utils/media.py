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
from typing import Any

import numpy as np
import requests  # type: ignore[import-untyped]
from PIL import Image

from .url import is_url


regex_media_base64 = r"^(data:[a-zA-Z]/[a-zA-Z]+;base64,)"


def match_base64_media(string: str, media: str | None = None) -> re.Match[str] | None:
    """Match a base64 media."""
    regex_media_base64 = rf"^(data:{media if media is not None else '[a-zA-Z]+'}/[a-zA-Z]+;base64,)"
    return re.match(regex_media_base64, string)


def is_base64_media(string: str, media: str | None) -> bool:
    """Check if a string is a base64 media.

    The expected format is "data:{media}/{image_format};base64,{base64}".
    """
    return match_base64_media(string, media) is not None


def is_base64_image(string: str) -> bool:
    """Check if a string is a base64 image.

    The expected format is "data:image/{image_format};base64,{base64}".
    """
    return is_base64_media(string, "image")


def is_base64_video(string: str) -> bool:
    """Check if a string is a base64 video.

    The expected format is "data:video/{video_format};base64,{base64}".
    """
    return is_base64_media(string, "video")


def extract_media_from_base64(string: str) -> str:
    """Extract from a base64 media the actual base64 part."""
    match = match_base64_media(string)
    if match is None:
        raise ValueError("The string does not match the expected format.")
    return string[len(match.group(1)) :]


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
                image_bytes = base64.b64decode(extract_media_from_base64(str_image))
                image_pil = Image.open(BytesIO(image_bytes))
            else:
                try:
                    path_exists = Path(str_image).exists()
                except OSError:
                    path_exists = False
                if path_exists:
                    image_pil = Image.open(str_image)
                else:
                    raise ValueError("The image is not a valid path, URL or base64 string.")
    elif isinstance(str_image, Path):
        image_pil = Image.open(str_image)
    else:
        raise ValueError("The image is not a valid path, URL or base64 string.")
    image_converted = image_pil.convert("RGB")
    return image_converted


def convert_string_video_to_bytes_or_path(str_video: str | Path) -> bytes | Path:
    """Convert a string to a video or video path.

    Args:
        str_video: Video as a string or path.

    Returns:
        The video.
    """
    if isinstance(str_video, list):
        return [convert_string_video_to_bytes_or_path(str_video_elem) for str_video_elem in str_video]
    if isinstance(str_video, str):
        if is_url(str_video):
            video_bytes = requests.get(str_video, stream=True).raw
        else:
            if is_base64_video(str_video):
                video_bytes = base64.b64decode(extract_media_from_base64(str_video))
            else:
                try:
                    path_exists = Path(str_video).exists()
                except OSError:
                    path_exists = False
                if path_exists:
                    return Path(str_video)
                else:
                    raise ValueError("The image is not a valid path, URL or base64 string.")
        return video_bytes
    elif isinstance(str_video, Path):
        return str_video
    else:
        raise ValueError("The image is not a valid path, URL or base64 string.")


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
