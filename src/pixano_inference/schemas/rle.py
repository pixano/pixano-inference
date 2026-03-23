# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for compressed and numeric RLE masks."""

from typing import Any

import numpy as np
from PIL.Image import Image
from pycocotools import mask as mask_api
from pydantic import BaseModel, field_serializer, field_validator, model_validator


def mask_to_rle(mask: Image | np.ndarray) -> dict:
    """Encode mask from Pillow or NumPy array to RLE.

    Args:
        mask: Mask as Pillow or NumPy array.

    Returns:
        Mask as RLE.
    """
    mask_array = np.asfortranarray(mask)
    return mask_api.encode(mask_array)


def rle_to_mask(rle: dict[str, list[int] | bytes]) -> np.ndarray:
    """Decode mask from RLE to NumPy array.

    Args:
        rle: Mask as RLE.

    Returns:
        Mask as NumPy array.
    """
    return mask_api.decode(rle)


class CompressedRLE(BaseModel):
    """Compressed or numeric RLE mask type.

    Attributes:
        size: Mask size.
        counts: Mask RLE encoding as compressed bytes/string or a numeric array.
    """

    size: list[int]
    counts: bytes | list[int]

    @field_validator("counts", mode="before")
    @classmethod
    def _validate_counts(cls, value: bytes | str | list[int]) -> bytes | list[int]:
        if isinstance(value, str):
            return bytes(value, "utf-8")
        if isinstance(value, list):
            if not all(isinstance(count, int) and count >= 0 for count in value):
                raise ValueError("Mask counts arrays must contain non-negative integers.")
            return value
        return value

    @model_validator(mode="after")
    def _validate_fields(self):
        if (
            len(self.size) != 2
            and not all(isinstance(s, int) and s > 0 for s in self.size)
            and not (self.size == [0, 0] and self.counts == b"")
        ):
            raise ValueError("Mask size must have 2 elements and be positive integers or [0, 0] for empty mask.")
        if isinstance(self.counts, list):
            compressed_rle = mask_api.frPyObjects(
                {"size": self.size, "counts": self.counts},
                self.size[0],
                self.size[1],
            )
            object.__setattr__(self, "counts", compressed_rle["counts"])
        return self

    @field_serializer("counts")
    def _serialize_counts(self, value: bytes | list[int]) -> str | list[int]:
        if isinstance(value, bytes):
            return str(value, "utf-8")
        return value

    @staticmethod
    def from_mask(mask: Image | np.ndarray, **kwargs: Any) -> "CompressedRLE":
        """Create a compressed RLE mask from a NumPy array.

        Args:
            mask: The mask as a NumPy array.
            kwargs: Additional arguments.

        Returns:
            The compressed RLE mask.
        """
        rle = mask_to_rle(mask)
        return CompressedRLE(size=rle["size"], counts=rle["counts"], **kwargs)

    def to_mask(self) -> np.ndarray:
        """Convert the compressed RLE mask to a NumPy array.

        Returns:
            The mask as a NumPy array.
        """
        return rle_to_mask({"size": self.size, "counts": self.counts})
