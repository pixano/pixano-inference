# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""What the YOLO models declare on ``/v1/models`` (no weights, no ultralytics import)."""

from __future__ import annotations

from pixano_yolo import YOLOByteTrackModel, YOLOModel

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models import DetectionInterface, TrackingInterface


class _FakeYOLO:
    """Only what the declaration reads: ultralytics keeps the class set as ``{index: name}``."""

    names = {2: "car", 0: "person", 1: "bicycle"}


def test_the_detector_publishes_its_class_set_once_loaded():
    model = YOLOModel(ModelDeploymentConfig(name="yolo", capability="detection", model_class="YOLOModel"))

    # Before the weights are loaded the set is unknown, not empty.
    assert model.interface == DetectionInterface(
        classes="closed", class_names=None, thresholds=["box"], outputs=["box", "score", "class"]
    )

    model._model = _FakeYOLO()
    assert model.interface.class_names == ["person", "bicycle", "car"]  # in class-index order


def test_the_tracker_is_prompt_free_with_the_detector_classes():
    model = YOLOByteTrackModel(
        ModelDeploymentConfig(name="bytetrack", capability="tracking", model_class="YOLOByteTrackModel")
    )
    model._model = _FakeYOLO()

    assert model.interface == TrackingInterface(
        prompts=[],
        prompt_free=True,
        classes="closed",
        class_names=["person", "bicycle", "car"],
        thresholds=["box"],
        interval=False,
        outputs=["box", "score", "class"],
    )
