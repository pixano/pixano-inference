# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

import numpy as np
import pytest

from pixano_inference.schemas.rle import CompressedRLE


def _mask_to_numeric_rle(mask: np.ndarray) -> list[int]:
    flat_mask = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current = 0
    run_length = 0

    for value in flat_mask:
        if value == current:
            run_length += 1
            continue
        counts.append(run_length)
        run_length = 1
        current = int(value)

    counts.append(run_length)
    return counts


def test_compressed_rle_accepts_numeric_counts_arrays():
    mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    numeric_rle = _mask_to_numeric_rle(mask)

    rle = CompressedRLE(size=[2, 2], counts=numeric_rle)
    expected_rle = CompressedRLE.from_mask(mask)

    assert rle.counts == expected_rle.counts
    np.testing.assert_array_equal(rle.to_mask(), mask)
    assert rle.model_dump(mode="json") == expected_rle.model_dump(mode="json")


def test_compressed_rle_rejects_invalid_numeric_counts_arrays():
    with pytest.raises(ValueError, match="Mask counts arrays must contain non-negative integers."):
        CompressedRLE(size=[2, 2], counts=[1, -1, 2])
