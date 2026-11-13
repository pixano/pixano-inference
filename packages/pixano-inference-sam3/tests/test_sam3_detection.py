# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Sam3DetectionModel against a fake SAM 3 processor: which prompts a request becomes, what comes back."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from pixano_inference_sam3 import Sam3DetectionModel
from pixano_inference_sam3.model import normalised_cxcywh

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models import DetectionInterface
from pixano_inference.models.detection import DetectionInput
from pixano_inference.schemas.rle import CompressedRLE


WIDTH, HEIGHT = 40, 20


class FakeProcessor:
    """Records the prompts; every prompt finds one object, at a box that tells the prompt apart."""

    def __init__(self) -> None:
        self.confidence_threshold = None
        self.calls: list[tuple] = []
        self.images_set = 0

    def set_image(self, image):
        self.images_set += 1
        return {}

    def reset_all_prompts(self, state):
        self.calls.append(("reset",))
        state.clear()

    def set_text_prompt(self, text, state):
        self.calls.append(("text", text))
        return self._found(state)

    def add_geometric_prompt(self, box, label, state):
        self.calls.append(("box", [round(v, 3) for v in box], label))
        return self._found(state)

    def _found(self, state):
        offset = len(self.calls)
        mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=bool)
        mask[0, 0, 2:5, offset : offset + 3] = True
        state["boxes"] = np.array([[offset, 2.4, offset + 3, 4.6]], dtype=np.float32)
        state["scores"] = np.array([0.8], dtype=np.float32)
        state["masks"] = mask
        return state


def _model(monkeypatch, **params) -> tuple[Sam3DetectionModel, FakeProcessor]:
    config = ModelDeploymentConfig(
        name="sam3", capability="detection", model_class="Sam3DetectionModel", model_params=params
    )
    model = Sam3DetectionModel(config)
    model._processor = FakeProcessor()
    monkeypatch.setattr(
        "pixano_inference.utils.media.convert_string_to_image", lambda image: Image.new("RGB", (WIDTH, HEIGHT))
    )
    return model, model._processor


def test_interface_reads_exemplars_and_returns_masks():
    model = Sam3DetectionModel(
        ModelDeploymentConfig(name="sam3", capability="detection", model_class="Sam3DetectionModel")
    )

    assert model.interface == DetectionInterface(
        classes="open",
        thresholds=["box"],
        exemplars=True,
        negative_exemplars=True,
        outputs=["box", "score", "class", "mask"],
    )


def test_interface_without_masks():
    model = Sam3DetectionModel(
        ModelDeploymentConfig(
            name="sam3",
            capability="detection",
            model_class="Sam3DetectionModel",
            model_params={"return_masks": False},
        )
    )

    assert "mask" not in model.interface.outputs


def test_each_class_is_its_own_text_prompt_on_one_encoding(monkeypatch):
    model, processor = _model(monkeypatch)

    output = model.predict(DetectionInput(image="x", classes="screw . nut", box_threshold=0.3))

    assert processor.images_set == 1
    assert processor.confidence_threshold == 0.3
    assert [call for call in processor.calls if call[0] == "text"] == [("text", "screw"), ("text", "nut")]
    assert output.classes == ["screw", "nut"]
    assert len(output.boxes) == len(output.scores) == len(output.masks) == 2


def test_exemplars_are_one_prompt_with_the_class_as_text(monkeypatch):
    model, processor = _model(monkeypatch)

    output = model.predict(
        DetectionInput(
            image="x",
            classes=["screw"],
            exemplar_boxes=[[0, 0, 20, 10], [20, 10, 40, 20]],
            negative_exemplar_boxes=[[10, 5, 30, 15]],
        )
    )

    assert processor.calls == [
        ("reset",),
        ("text", "screw"),
        ("box", [0.25, 0.25, 0.5, 0.5], True),
        ("box", [0.75, 0.75, 0.5, 0.5], True),
        ("box", [0.5, 0.5, 0.5, 0.5], False),
    ]
    # One prompt, one answer: the state after its last box.
    assert output.classes == ["screw"]


def test_exemplars_without_a_class_are_a_visual_prompt(monkeypatch):
    model, processor = _model(monkeypatch)

    output = model.predict(DetectionInput(image="x", exemplar_boxes=[[0, 0, 20, 10]]))

    assert not [call for call in processor.calls if call[0] == "text"]
    assert output.classes == [""]


def test_boxes_are_integer_pixels_and_masks_are_rle(monkeypatch):
    model, _ = _model(monkeypatch)

    output = model.predict(DetectionInput(image="x", classes=["screw"]))

    # The fake answers [2, 2.4, 5, 4.6] for the second call (reset, then text).
    assert output.boxes == [[2, 2, 5, 5]]
    assert output.scores == pytest.approx([0.8])
    assert isinstance(output.masks[0], CompressedRLE)
    assert output.masks[0].to_mask().sum() == 9


def test_masks_can_be_left_out(monkeypatch):
    model, _ = _model(monkeypatch, return_masks=False)

    assert model.predict(DetectionInput(image="x", classes=["screw"])).masks is None


def test_a_request_needs_a_class_or_an_exemplar(monkeypatch):
    model, _ = _model(monkeypatch)

    with pytest.raises(ValueError, match="requires 'classes' or 'exemplar_boxes'"):
        model.predict(DetectionInput(image="x"))


def test_normalised_cxcywh():
    assert normalised_cxcywh([10, 0, 30, 20], WIDTH, HEIGHT) == [0.5, 0.5, 0.5, 1.0]
