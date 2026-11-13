# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Detection wire types: exemplar boxes ("more like these") and the interface flags that announce them."""

import pytest
from pydantic import ValidationError

from pixano_inference.schemas import DetectionInput, DetectionInterface, DetectionRequest
from pixano_inference.schemas.detection import class_names_of


IMAGE = "https://example.com/x.jpg"


def test_exemplars_travel_in_camel_case():
    request = DetectionRequest.model_validate(
        {
            "model": "sam3",
            "image": IMAGE,
            "classes": ["screw"],
            "exemplarBoxes": [[10, 20, 30, 40]],
            "negativeExemplarBoxes": [[50, 60, 70, 80]],
        }
    )

    detection_input = request.to_input()

    assert detection_input.exemplar_boxes == [[10, 20, 30, 40]]
    assert detection_input.negative_exemplar_boxes == [[50, 60, 70, 80]]
    wire = request.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert wire["exemplarBoxes"] == [[10, 20, 30, 40]]
    assert wire["negativeExemplarBoxes"] == [[50, 60, 70, 80]]


def test_a_request_without_exemplars_is_unchanged():
    detection_input = DetectionInput(image=IMAGE, classes=["cat", "dog"])

    assert detection_input.exemplar_boxes is None
    assert detection_input.negative_exemplar_boxes is None


def test_exemplars_may_name_no_class():
    assert DetectionInput(image=IMAGE, exemplar_boxes=[[0, 0, 5, 5]]).classes is None


@pytest.mark.parametrize("classes", [["screw", "nut"], "screw . nut"])
def test_exemplars_name_at_most_one_class(classes):
    with pytest.raises(ValidationError, match="at most one class"):
        DetectionInput(image=IMAGE, classes=classes, exemplar_boxes=[[0, 0, 5, 5]])


def test_negative_exemplars_require_exemplars():
    with pytest.raises(ValidationError, match="require exemplar_boxes"):
        DetectionInput(image=IMAGE, classes="screw", negative_exemplar_boxes=[[0, 0, 5, 5]])


@pytest.mark.parametrize(
    ("box", "message"),
    [([0, 0, 5], "4 coordinates"), ([10, 10, 5, 5], "two corners"), ([0, 0, 0, 5], "two corners")],
)
def test_an_exemplar_box_is_two_corners(box, message):
    with pytest.raises(ValidationError, match=message):
        DetectionInput(image=IMAGE, exemplar_boxes=[box])


@pytest.mark.parametrize(
    ("classes", "names"),
    [(None, []), ("screw", ["screw"]), ("cat . dog .", ["cat", "dog"]), (["cat", " ", "dog "], ["cat", "dog"])],
)
def test_class_names_of(classes, names):
    assert class_names_of(classes) == names


def test_interface_announces_exemplars():
    declared = DetectionInterface(classes="open", exemplars=True, negative_exemplars=True, outputs=["box"])

    wire = declared.model_dump(mode="json", by_alias=True)

    assert wire["exemplars"] is True
    assert wire["negativeExemplars"] is True
    assert DetectionInterface.model_validate(wire) == declared
    # A model that says nothing about exemplars does not read them.
    assert DetectionInterface(classes="open", outputs=["box"]).exemplars is False


def test_interface_negative_exemplars_require_exemplars():
    with pytest.raises(ValidationError, match="requires 'exemplars'"):
        DetectionInterface(classes="open", negative_exemplars=True, outputs=["box"])
