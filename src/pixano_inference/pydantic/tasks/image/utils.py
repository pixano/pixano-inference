# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for image tasks."""

import base64

import numpy as np
from pydantic import BaseModel, field_validator

from pixano_inference.utils.image import decode_rle_to_mask, decompress_rle


class RLEMask(BaseModel):
    """Compressed run-length encoded mask.

    Attributes:
        counts: List of counts compressed in a base64 encoded string.
        size: Size of the mask.
    """

    counts: str
    size: list[int]

    @field_validator("counts", mode="before")
    @classmethod
    def _convert_counts(cls, v: str | list[int]) -> str:
        if isinstance(v, list):
            if not all(isinstance(i, int) for i in v):
                raise ValueError("Counts should be integers.")
            counts = np.array(v, dtype=np.int32)
            return base64.b64encode(counts.tobytes()).decode("utf-8")
        return v

    def decode(self) -> np.ndarray:
        """Decode the compressed RLE mask.

        Returns:
            Decoded RLE mask.
        """
        rle_mask = decompress_rle(self.model_dump())
        mask = decode_rle_to_mask(rle_mask)
        return mask
