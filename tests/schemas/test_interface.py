# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""The per-capability ``interface`` descriptor a model publishes on ``/v1/models``."""

import pytest
from pydantic import TypeAdapter, ValidationError

from pixano_inference.schemas import (
    DetectionInterface,
    EmbeddingInterface,
    ImageRange,
    ModelInterface,
    ModelStatusInfo,
    NERInterface,
    SegmentationInterface,
    TrackingInterface,
    VLMInterface,
)


_ANY_INTERFACE = TypeAdapter(ModelInterface)


def test_tracking_interface_round_trips_in_camel_case():
    declared = TrackingInterface(
        prompts=[],
        prompt_free=True,
        classes="closed",
        class_names=["person", "car"],
        thresholds=["box"],
        outputs=["box", "score", "class"],
    )

    wire = declared.model_dump(mode="json", by_alias=True)

    assert wire == {
        "capability": "tracking",
        "prompts": [],
        "promptFree": True,
        "classes": "closed",
        "classNames": ["person", "car"],
        "thresholds": ["box"],
        "interval": False,
        "outputs": ["box", "score", "class"],
    }
    assert TrackingInterface.model_validate(wire) == declared
    # snake_case stays accepted for Python callers.
    assert TrackingInterface.model_validate({"prompt_free": True, "outputs": ["box"]}).prompt_free is True


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ({"capability": "tracking", "prompts": ["points", "box", "mask"], "outputs": ["mask"]}, TrackingInterface),
        (
            {"capability": "detection", "classes": "open", "thresholds": ["box", "text"], "outputs": ["box"]},
            DetectionInterface,
        ),
        (
            {"capability": "segmentation", "prompts": ["points"], "multimask": True, "outputs": ["mask", "score"]},
            SegmentationInterface,
        ),
        ({"capability": "vlm", "prompt": ["messages"], "images": {"min": 1, "max": 4}}, VLMInterface),
        ({"capability": "embedding", "modalities": ["image", "text"], "dim": 512}, EmbeddingInterface),
        ({"capability": "ner", "entityTypes": ["ORG", "PER"]}, NERInterface),
    ],
)
def test_capability_tells_the_descriptors_apart(wire, expected):
    declared = _ANY_INTERFACE.validate_python(wire)

    assert isinstance(declared, expected)
    assert declared.model_dump(mode="json", by_alias=True, exclude_unset=True) == wire


@pytest.mark.parametrize(
    ("wire", "message"),
    [
        ({"outputs": ["mask"]}, "Unable to extract tag using discriminator 'capability'"),
        ({"capability": "ocr", "outputs": ["mask"]}, "does not match any of the expected tags"),
        ({"capability": "detection", "classes": "any", "outputs": ["box"]}, "Input should be 'open' or 'closed'"),
        ({"capability": "tracking", "outputs": ["polygon"]}, "Input should be"),
    ],
)
def test_an_unknown_or_malformed_descriptor_is_rejected(wire, message):
    with pytest.raises(ValidationError, match=message):
        _ANY_INTERFACE.validate_python(wire)


@pytest.mark.parametrize("descriptor", [TrackingInterface, DetectionInterface])
def test_class_names_belong_to_a_closed_class_set(descriptor):
    assert descriptor(classes="closed", class_names=["cat"], outputs=["box"]).class_names == ["cat"]
    assert descriptor(classes="closed", outputs=["box"]).class_names is None  # fixed set, not published
    with pytest.raises(ValidationError, match="requires classes='closed'"):
        descriptor(classes="open", class_names=["cat"], outputs=["box"])


def test_image_range_is_ordered():
    assert ImageRange().model_dump() == {"min": 0, "max": None}
    with pytest.raises(ValidationError, match="'max' must be at least 'min'"):
        ImageRange(min=2, max=1)


def test_model_status_carries_the_interface_or_null():
    listed = ModelStatusInfo(
        name="sam2-video",
        capability="tracking",
        model_class="Sam2VideoModel",
        status="RUNNING",
        interface=TrackingInterface(prompts=["points", "box", "mask"], interval=True, outputs=["mask"]),
    )

    wire = listed.model_dump(mode="json", by_alias=True)

    assert wire["interface"]["capability"] == "tracking" and wire["interface"]["prompts"] == ["points", "box", "mask"]
    assert ModelStatusInfo.model_validate(wire) == listed
    # A model that predates the descriptor, or a 0.7.0 server that never sends the field.
    assert ModelStatusInfo.model_validate({**wire, "interface": None}).interface is None
    assert ModelStatusInfo.model_validate({"name": "m", "capability": "ner", "status": "RUNNING"}).interface is None
