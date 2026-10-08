# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""How a deployed model is called: one descriptor per capability.

A capability's ``Input`` is the union of what every model of that capability might accept: a SAM2
video model reads ``keyframes`` while a ByteTrack model reads ``classes``; Grounding DINO requires
``classes`` while a YOLO detector has its own class set. The capability alone does not tell a client
how to call a given model, what to show in a model picker, or which class names a closed-vocabulary
model knows. A model MAY describe itself with the descriptor of its capability
(:attr:`pixano_inference.models.InferenceModel.interface`); the server publishes it as
``ModelStatusInfo.interface`` on ``GET /v1/models``, ``None`` when the model declares none.

The descriptors are told apart on the wire by their ``capability`` field (:data:`ModelInterface`).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from .base import _BaseModel


Prompt = Literal["points", "box", "mask", "text"]
"""What a request may carry to name an object."""

Threshold = Literal["box", "text"]
"""A confidence threshold a request may set."""


def _check_class_names(classes: str, class_names: list[str] | None) -> None:
    if class_names is not None and classes != "closed":
        raise ValueError("'class_names' is the class set of a closed-vocabulary model: it requires classes='closed'.")


class TrackingInterface(_BaseModel):
    """How a tracking model is called.

    Attributes:
        capability: Always ``"tracking"``; tells the descriptors apart on the wire.
        prompts: What a keyframe may carry to name an object. Empty for a model that takes no prompt.
        prompt_free: Whether the model accepts a request that names no object (tracking by
            detection: the model detects the objects and creates the tracks itself).
        classes: ``"none"`` when the model has no notion of class, ``"open"`` when it detects the
            class names the request gives, ``"closed"`` when it has its own class set.
        class_names: The class set of a closed-vocabulary model, ``None`` otherwise.
        thresholds: The confidence thresholds the model reads from the request.
        interval: Whether the model honours the propagation ``interval`` of the request.
        outputs: What each tracked object carries.
    """

    capability: Literal["tracking"] = "tracking"
    prompts: list[Prompt] = []
    prompt_free: bool = False
    classes: Literal["none", "open", "closed"] = "none"
    class_names: list[str] | None = None
    thresholds: list[Threshold] = []
    interval: bool = False
    outputs: list[Literal["mask", "box", "score", "class"]]

    @model_validator(mode="after")
    def _check(self) -> TrackingInterface:
        _check_class_names(self.classes, self.class_names)
        return self


class DetectionInterface(_BaseModel):
    """How a detection model is called.

    Attributes:
        capability: Always ``"detection"``; tells the descriptors apart on the wire.
        classes: ``"open"`` when the model detects the class names the request gives (and requires
            them), ``"closed"`` when it has its own class set.
        class_names: The class set of a closed-vocabulary model, ``None`` otherwise.
        thresholds: The confidence thresholds the model reads from the request.
        outputs: What each detection carries.
    """

    capability: Literal["detection"] = "detection"
    classes: Literal["open", "closed"]
    class_names: list[str] | None = None
    thresholds: list[Threshold] = []
    outputs: list[Literal["box", "score", "class", "mask"]]

    @model_validator(mode="after")
    def _check(self) -> DetectionInterface:
        _check_class_names(self.classes, self.class_names)
        return self


class SegmentationInterface(_BaseModel):
    """How an image segmentation model is called.

    Attributes:
        capability: Always ``"segmentation"``; tells the descriptors apart on the wire.
        prompts: What a request may carry to name what to segment. Empty for a model that segments
            without a prompt.
        multimask: Whether the model can return several candidate masks per prompt.
        embeddings: Whether the model returns the image embedding on request and accepts it back,
            so a client can re-prompt the same image without re-encoding it.
        outputs: What a prediction carries.
    """

    capability: Literal["segmentation"] = "segmentation"
    prompts: list[Prompt] = []
    multimask: bool = False
    embeddings: bool = False
    outputs: list[Literal["mask", "score", "logits"]]


class ImageRange(_BaseModel):
    """How many images a request may carry.

    Attributes:
        min: The fewest images a request must carry.
        max: The most images a request may carry; ``None`` when there is no limit.
    """

    min: int = Field(default=0, ge=0)
    max: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> ImageRange:
        if self.max is not None and self.max < self.min:
            raise ValueError("'max' must be at least 'min'.")
        return self


class VLMInterface(_BaseModel):
    """How a vision-language model is called.

    Attributes:
        capability: Always ``"vlm"``; tells the descriptors apart on the wire.
        prompt: The prompt forms the model accepts: a ``text`` string, or chat ``messages``.
        images: How many images a request may carry, in ``images`` or embedded in the messages.
    """

    capability: Literal["vlm"] = "vlm"
    prompt: list[Literal["text", "messages"]]
    images: ImageRange = ImageRange()


class EmbeddingInterface(_BaseModel):
    """How an embedding model is called.

    Attributes:
        capability: Always ``"embedding"``; tells the descriptors apart on the wire.
        modalities: What the model embeds. A model listing both maps them into one shared space.
        dim: Dimensionality of the vectors, ``None`` when it is not known before a call.
    """

    capability: Literal["embedding"] = "embedding"
    modalities: list[Literal["image", "text"]]
    dim: int | None = Field(default=None, ge=1)


class NERInterface(_BaseModel):
    """How a named entity recognition model is called.

    Attributes:
        capability: Always ``"ner"``; tells the descriptors apart on the wire.
        entity_types: The labels the model can return, ``None`` when they are not known.
    """

    capability: Literal["ner"] = "ner"
    entity_types: list[str] | None = None


ModelInterface = Annotated[
    Union[
        TrackingInterface,
        DetectionInterface,
        SegmentationInterface,
        VLMInterface,
        EmbeddingInterface,
        NERInterface,
    ],
    Field(discriminator="capability"),
]
"""The descriptor of a model, whichever its capability: a union told apart by ``capability``."""
