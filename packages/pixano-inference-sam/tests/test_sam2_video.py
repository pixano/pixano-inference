# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

from __future__ import annotations

import numpy as np
import pytest
from pixano_inference_sam.video import Sam2VideoModel

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models.tracking import TrackingInput


def _mask(*rows: list[int]) -> np.ndarray:
    """A ``(1, H, W)`` boolean mask, the shape SAM2 yields for one object."""
    return np.array([rows], dtype=bool)


def test_build_output_groups_the_masks_by_frame():
    # Backward propagation: frames arrive in decreasing order, and object 5 is prompted first.
    video_segments = {
        1: {5: _mask([1, 1], [0, 0]), 2: _mask([0, 0], [0, 1])},
        0: {5: _mask([1, 0], [0, 0]), 2: _mask([0, 0], [0, 0])},
    }

    output = Sam2VideoModel._build_output(video_segments)

    assert [frame.frame_index for frame in output.frames] == [1, 0]
    assert [[tracked.track_id for tracked in frame.objects] for frame in output.frames] == [[5, 2], [5, 2]]
    first = output.frames[0].objects[0]
    assert first.box is None and first.score is None
    np.testing.assert_array_equal(first.mask.to_mask(), np.array([[1, 1], [0, 0]], dtype=np.uint8))
    # An object that is not visible in a frame keeps its place, with an empty mask.
    assert output.frames[1].objects[1].mask.to_mask().sum() == 0
    assert {track_id: [index for index, _ in track] for track_id, track in output.tracks().items()} == {
        5: [0, 1],
        2: [0, 1],
    }


def test_predict_rejects_a_prompt_free_request():
    model = Sam2VideoModel(
        ModelDeploymentConfig(name="sam2-video", capability="tracking", model_class="Sam2VideoModel")
    )

    with pytest.raises(ValueError, match="tracks prompted objects"):
        model.predict(TrackingInput(video=[b"frame"]))
