# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tracking wire types: the by-frame output and the prompted / prompt-free request."""

import numpy as np
import pytest
from pydantic import ValidationError

from pixano_inference.schemas import (
    CompressedRLE,
    TrackedFrame,
    TrackedObject,
    TrackingInput,
    TrackingOutput,
    TrackingRequestV1,
)


def _mask() -> CompressedRLE:
    return CompressedRLE.from_mask(np.array([[1, 1], [0, 0]], dtype=np.uint8))


# --- Output ---------------------------------------------------------------------------


def test_output_serializes_by_frame_in_camel_case():
    output = TrackingOutput(
        frames=[
            TrackedFrame(
                frame_index=0,
                objects=[
                    TrackedObject(track_id=1, box=[12, 30, 80, 190], score=0.91, class_name="person"),
                    TrackedObject(track_id=2, box=[200, 41, 260, 180], score=0.84, class_name="person"),
                ],
            ),
        ]
    )

    wire = output.model_dump(mode="json", by_alias=True)

    assert wire == {
        "frames": [
            {
                "frameIndex": 0,
                "objects": [
                    {"trackId": 1, "box": [12.0, 30.0, 80.0, 190.0], "score": 0.91, "class": "person", "mask": None},
                    {"trackId": 2, "box": [200.0, 41.0, 260.0, 180.0], "score": 0.84, "class": "person", "mask": None},
                ],
            }
        ]
    }
    assert TrackingOutput.model_validate(wire) == output


def test_object_is_located_by_a_mask_or_a_box():
    assert TrackedObject(track_id=1, mask=_mask()).box is None
    assert TrackedObject(track_id=1, box=[0, 0, 4, 4]).mask is None
    with pytest.raises(ValidationError, match="needs a box or a mask"):
        TrackedObject(track_id=1, score=0.5)


@pytest.mark.parametrize("box", [[1, 2, 3], [1, 2, 3, 4, 5]])
def test_box_has_four_values(box):
    with pytest.raises(ValidationError):
        TrackedObject(track_id=1, box=box)


def test_box_is_two_corners_not_width_and_height():
    # ByteTrack works in (x, y, width, height); returned as-is, such a box usually has x2 < x1.
    with pytest.raises(ValidationError, match=r"\[x1, y1, x2, y2\]"):
        TrackedObject(track_id=1, box=[200, 41, 60, 139])


def test_a_track_appears_once_per_frame_and_a_frame_once_per_output():
    tracked = TrackedObject(track_id=1, box=[0, 0, 4, 4])
    with pytest.raises(ValidationError, match="at most once in a frame"):
        TrackedFrame(frame_index=0, objects=[tracked, tracked])
    with pytest.raises(ValidationError, match="at most once in a tracking output"):
        TrackingOutput(frames=[TrackedFrame(frame_index=3), TrackedFrame(frame_index=3)])


def test_tracks_groups_the_frames_by_track_in_frame_order():
    def frame(index: int, *track_ids: int) -> TrackedFrame:
        return TrackedFrame(
            frame_index=index,
            objects=[TrackedObject(track_id=track_id, box=[index, 0, index + 4, 4]) for track_id in track_ids],
        )

    # Frames in processing order (backward propagation): track 2 enters at frame 1.
    output = TrackingOutput(frames=[frame(2, 1, 2), frame(1, 2, 1), frame(0, 1)])

    tracks = output.tracks()

    assert sorted(tracks) == [1, 2]
    assert [index for index, _ in tracks[1]] == [0, 1, 2]
    assert [index for index, _ in tracks[2]] == [1, 2]
    assert tracks[2][0][1].box == [1.0, 0.0, 5.0, 4.0]


# --- Input ----------------------------------------------------------------------------


def test_prompt_free_request_needs_only_the_video():
    request = TrackingRequestV1(model="bytetrack", video=["f0.png", "f1.png"], classes=["person"], box_threshold=0.4)

    tracking_input = request.to_input()

    assert tracking_input.objects_ids == [] and tracking_input.frame_indexes == []
    assert tracking_input.keyframes is None
    assert tracking_input.classes == ["person"]
    assert tracking_input.box_threshold == 0.4


def test_prompted_request_still_maps_keyframes_to_objects():
    request = TrackingRequestV1.model_validate(
        {
            "model": "sam2-video",
            "video": ["f0.png"],
            "objectsIds": [7],
            "frameIndexes": [0],
            "keyframes": [{"frameIndex": 0, "prompts": {"points": [{"x": 3, "y": 4, "label": 1}]}}],
        }
    )

    tracking_input = request.to_input()

    assert tracking_input.objects_ids == [7]
    assert tracking_input.keyframes[0].points[0].x == 3


@pytest.mark.parametrize(
    "prompts",
    [
        {"keyframes": [{"frameIndex": 0, "prompts": {"points": [{"x": 3, "y": 4, "label": 1}]}}]},
    ],
)
def test_request_with_prompts_but_no_object_is_rejected_when_parsed(prompts):
    with pytest.raises(ValidationError, match="Prompts require object IDs"):
        TrackingRequestV1.model_validate({"model": "sam2-video", "video": ["f0.png"], **prompts})


@pytest.mark.parametrize("prompts", [{"points": [[[1, 2]]]}, {"labels": [[1]]}, {"boxes": [[1, 2, 3, 4]]}])
def test_legacy_prompts_require_object_ids(prompts):
    with pytest.raises(ValidationError, match="Prompts require object IDs"):
        TrackingInput(video=["f0.png"], **prompts)


def test_object_ids_are_unique_and_match_the_keyframes():
    with pytest.raises(ValidationError, match="unique"):
        TrackingInput(video=["f0.png"], objects_ids=[1, 1], frame_indexes=[0, 0])
    with pytest.raises(ValidationError, match="exactly one keyframe per object ID"):
        TrackingRequestV1.model_validate(
            {
                "model": "sam2-video",
                "video": ["f0.png"],
                "objectsIds": [1, 2],
                "frameIndexes": [0, 0],
                "keyframes": [{"frameIndex": 0, "prompts": {"points": [{"x": 3, "y": 4, "label": 1}]}}],
            }
        )
