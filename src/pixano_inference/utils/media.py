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
from typing import Any, cast

import numpy as np
from PIL import Image

from .media_security import fetch_url_bytes, get_media_policy, is_http_url, resolve_local_path


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


def convert_string_to_image(str_image: str | Path | bytes) -> Image.Image:
    """Convert a string, path, or bytes reference to an image, under the media policy.

    URLs are fetched with SSRF protection, timeouts, and a size cap; local paths are only
    accepted when they resolve under a configured media root (see
    :mod:`pixano_inference.utils.media_security`).

    Args:
        str_image: Image as a URL, base64 data-URI, local path, ``Path``, or raw bytes.

    Returns:
        The RGB image.
    """
    policy = get_media_policy()
    if isinstance(str_image, str):
        if is_http_url(str_image):
            image_bytes = fetch_url_bytes(str_image, max_bytes=policy.max_image_bytes, policy=policy)
            image_pil = Image.open(BytesIO(image_bytes))
        elif is_base64_image(str_image):
            image_bytes = base64.b64decode(extract_media_from_base64(str_image))
            image_pil = Image.open(BytesIO(image_bytes))
        else:
            image_pil = Image.open(resolve_local_path(str_image, policy))
    elif isinstance(str_image, bytes):
        image_pil = Image.open(BytesIO(str_image))
    elif isinstance(str_image, Path):
        image_pil = Image.open(resolve_local_path(str_image, policy))
    else:
        raise ValueError("The image is not a valid path, URL or base64 string.")
    image_converted = image_pil.convert("RGB")
    return image_converted


def convert_string_video_to_bytes_or_path(
    str_video: list[str | Path | bytes] | str | Path | bytes,
) -> list[bytes | Path] | bytes | Path:
    """Convert a video reference to bytes or a path, under the media policy.

    URLs are fetched with SSRF protection, timeouts, and a size cap; local paths are only
    accepted when they resolve under a configured media root.

    Args:
        str_video: Video as a URL, base64 data-URI, local path, ``Path``, raw bytes, or a
            list of any of those (per-frame).

    Returns:
        The video as bytes or a resolved ``Path`` (or a list thereof).
    """
    policy = get_media_policy()
    if isinstance(str_video, list):
        return [
            cast(bytes | Path, convert_string_video_to_bytes_or_path(str_video_elem)) for str_video_elem in str_video
        ]
    if isinstance(str_video, bytes):
        return str_video
    if isinstance(str_video, str):
        if is_http_url(str_video):
            return fetch_url_bytes(str_video, max_bytes=policy.max_video_bytes, policy=policy)
        if is_base64_video(str_video):
            return base64.b64decode(extract_media_from_base64(str_video))
        return resolve_local_path(str_video, policy)
    elif isinstance(str_video, Path):
        return resolve_local_path(str_video, policy)
    else:
        raise ValueError("The video is not a valid path, URL or base64 string.")


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
