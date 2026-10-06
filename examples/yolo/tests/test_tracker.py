# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for YOLOByteTrackModel with a fake ultralytics model (no weights, no network)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pixano_yolo import YOLOByteTrackModel

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models.tracking import TrackingInput


class _Tensor:
    """Stands in for a torch tensor: ``.cpu().numpy()``."""

    def __init__(self, values) -> None:
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _Boxes:
    def __init__(self, tracks: list[tuple[int, list[float], float, int]] | None) -> None:
        tracks = tracks or []
        # ultralytics leaves ``id`` to None when the tracker returns no track for the frame.
        self.id = _Tensor([t[0] for t in tracks]) if tracks else None
        self.xyxy = _Tensor([t[1] for t in tracks])
        self.conf = _Tensor([t[2] for t in tracks])
        self.cls = _Tensor([t[3] for t in tracks])


class _Result:
    names = {0: "person", 2: "car"}

    def __init__(self, tracks: list[tuple[int, list[float], float, int]] | None) -> None:
        self.boxes = _Boxes(tracks)


class FakeYOLO:
    """Records the ``track`` calls and replays one prepared result per frame."""

    names = _Result.names

    def __init__(self, per_frame: list[list[tuple[int, list[float], float, int]] | None]) -> None:
        self._results = [_Result(tracks) for tracks in per_frame]
        self.calls: list[dict] = []

    def track(self, source=None, persist=False, stream=False, **options):
        call = {"persist": persist, "stream": stream, **options}
        if stream:
            call["source_existed"] = Path(source).is_file()
            call["source_name"] = Path(source).name
            self.calls.append(call)
            return iter(self._results)
        self.calls.append(call)
        return [self._results[len(self.calls) - 1]]


def _frame() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 120, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


def _model(per_frame) -> tuple[YOLOByteTrackModel, FakeYOLO]:
    model = YOLOByteTrackModel(
        ModelDeploymentConfig(name="yolo-bytetrack", capability="tracking", model_class="YOLOByteTrackModel")
    )
    fake = FakeYOLO(per_frame)
    model._model = fake
    return model, fake


def test_each_frame_lists_its_tracks_with_box_score_and_class():
    model, _ = _model(
        [
            [(1, [10, 20, 30, 40], 0.9, 0), (2, [50, 60, 70.26, 80], 0.6, 2)],
            [(1, [12, 20, 32, 40], 0.8, 0)],
        ]
    )

    output = model.predict(TrackingInput(video=[_frame(), _frame()]))

    assert [frame.frame_index for frame in output.frames] == [0, 1]
    first, second = output.frames[0].objects
    assert (first.track_id, first.class_name, first.box) == (1, "person", [10.0, 20.0, 30.0, 40.0])
    assert first.score == pytest.approx(0.9)
    assert (second.track_id, second.class_name, second.box) == (2, "car", [50.0, 60.0, 70.3, 80.0])
    assert first.mask is None
    assert [tracked.track_id for tracked in output.frames[1].objects] == [1]
    # The same result read as trajectories: track 1 lives on both frames, track 2 on the first.
    assert {track_id: [index for index, _ in track] for track_id, track in output.tracks().items()} == {
        1: [0, 1],
        2: [0],
    }


def test_one_tracker_per_request_started_on_the_first_frame():
    model, fake = _model([None, None, None])

    model.predict(TrackingInput(video=[_frame(), _frame(), _frame()]))

    # persist=False starts fresh trackers (IDs restart at 1); persist=True continues the sequence.
    assert [call["persist"] for call in fake.calls] == [False, True, True]
    assert all(call["tracker"] == "bytetrack.yaml" and call["verbose"] is False for call in fake.calls)
    # No class filter and no confidence floor unless the request asks for them.
    assert all("classes" not in call and "conf" not in call for call in fake.calls)


def test_a_frame_where_nothing_is_tracked_has_no_object():
    model, _ = _model([[(1, [10, 20, 30, 40], 0.9, 0)], None])

    output = model.predict(TrackingInput(video=[_frame(), _frame()]))

    assert [len(frame.objects) for frame in output.frames] == [1, 0]


@pytest.mark.parametrize(("classes", "expected"), [(["car", "person"], [2, 0]), ("person", [0])])
def test_class_names_and_threshold_reach_the_detector(classes, expected):
    model, fake = _model([None])

    model.predict(TrackingInput(video=[_frame()], classes=classes, box_threshold=0.4))

    assert fake.calls[0]["classes"] == expected
    assert fake.calls[0]["conf"] == 0.4


def test_unknown_class_is_rejected():
    model, fake = _model([None])

    with pytest.raises(ValueError, match=r"Unknown class name\(s\) \['unicorn'\]"):
        model.predict(TrackingInput(video=[_frame()], classes=["person", "unicorn"]))
    assert fake.calls == []


def test_a_prompted_request_is_rejected():
    model, fake = _model([None])
    prompted = TrackingInput(
        video=[_frame()],
        objects_ids=[1],
        frame_indexes=[0],
        keyframes=[{"frame_index": 0, "points": [{"x": 3, "y": 4, "label": 1}]}],
    )

    with pytest.raises(ValueError, match="tracks by detection"):
        model.predict(prompted)
    assert fake.calls == []


def test_a_single_video_is_streamed_from_a_file():
    model, fake = _model([[(1, [10, 20, 30, 40], 0.9, 0)], [(1, [11, 20, 31, 40], 0.9, 0)]])

    output = model.predict(TrackingInput(video=b"not-really-mp4-bytes"))

    assert [frame.frame_index for frame in output.frames] == [0, 1]
    (call,) = fake.calls
    assert call["stream"] is True and call["persist"] is False
    assert call["source_existed"] and call["source_name"] == "video.mp4"
