# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""The capabilities served over HTTP, each described once.

A capability ties together a model base class, the ``Input``/``Output`` types of its ``predict``,
the request and response of its route, its default timeout and, when it accepts raw uploads, the
multipart field of its ``/binary`` route. :data:`CAPABILITIES` is the single place where that is
declared: the routes, the capability lookup of a model class and the default timeouts are derived
from it. Adding a capability means adding one entry here (plus its base class and schemas).
"""

# The capability base classes are abstract and are passed around as classes, never instantiated
# here, which mypy otherwise rejects for a ``type[...]`` argument.
# mypy: disable-error-code="type-abstract"

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from pixano_inference.models.base import InferenceModel
from pixano_inference.models.detection import DetectionModel
from pixano_inference.models.embedding import EmbeddingModel
from pixano_inference.models.ner import NERModel
from pixano_inference.models.segmentation import SegmentationModel
from pixano_inference.models.tracking import TrackingModel
from pixano_inference.models.vlm import VLMModel
from pixano_inference.schemas.base import BaseRequest, BaseResponse
from pixano_inference.schemas.detection import DetectionInput, DetectionOutput
from pixano_inference.schemas.embedding import EmbeddingInput, EmbeddingOutput
from pixano_inference.schemas.inference import (
    DetectionRequest,
    DetectionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    NERRequest,
    NERResponse,
    SegmentationRequest,
    SegmentationResponse,
    TrackingResponse,
    VLMRequest,
    VLMResponse,
)
from pixano_inference.schemas.ner import NERInput, NEROutput
from pixano_inference.schemas.segmentation import SegmentationInput, SegmentationOutput
from pixano_inference.schemas.tracking import TrackingInput, TrackingOutput
from pixano_inference.schemas.v1 import TrackingRequestV1
from pixano_inference.schemas.vlm import VLMInput, VLMOutput


@dataclass(frozen=True)
class BinaryUpload:
    """Multipart upload accepted by the ``/binary`` route of a capability.

    Attributes:
        file_field: Name of the multipart part(s) carrying the raw media. ``"image"`` is a single
            file; any other name is a list of files.
        payload_key: Field of the request model that receives the uploaded bytes.
    """

    file_field: str
    payload_key: str


@dataclass(frozen=True)
class CapabilitySpec:
    """Everything the server needs to know about one capability.

    Attributes:
        model_base: Base class a model subclasses to implement the capability. Its
            ``capability_name`` is the name of the capability.
        input_type: Type ``predict`` receives.
        output_type: Type ``predict`` returns.
        request_type: Body of the route; it exposes ``model`` and ``to_input()``.
        response_type: Response envelope of the route, whose ``data`` is ``output_type``.
        default_timeout_s: Inference timeout when the deployment does not set one.
        binary: Upload accepted by the ``/binary`` route, or ``None`` when there is no such route.
    """

    model_base: type[InferenceModel]
    input_type: type[BaseModel]
    output_type: type[BaseModel]
    request_type: type[BaseRequest]
    response_type: type[BaseResponse]
    default_timeout_s: float = 60.0
    binary: BinaryUpload | None = None

    def __post_init__(self) -> None:
        """Reject a base class that does not name its capability."""
        if not self.model_base.capability_name:
            raise ValueError(f"{self.model_base.__name__} does not define 'capability_name'.")

    @property
    def name(self) -> str:
        """Name of the capability: the route segment and the value stored in deployment configs."""
        name = self.model_base.capability_name
        assert name is not None  # checked in __post_init__
        return name


CAPABILITIES: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        model_base=SegmentationModel,
        input_type=SegmentationInput,
        output_type=SegmentationOutput,
        request_type=SegmentationRequest,
        response_type=SegmentationResponse,
        binary=BinaryUpload(file_field="image", payload_key="image"),
    ),
    CapabilitySpec(
        model_base=DetectionModel,
        input_type=DetectionInput,
        output_type=DetectionOutput,
        request_type=DetectionRequest,
        response_type=DetectionResponse,
        binary=BinaryUpload(file_field="image", payload_key="image"),
    ),
    CapabilitySpec(
        model_base=TrackingModel,
        input_type=TrackingInput,
        output_type=TrackingOutput,
        request_type=TrackingRequestV1,
        response_type=TrackingResponse,
        default_timeout_s=600.0,
        binary=BinaryUpload(file_field="frames", payload_key="video"),
    ),
    CapabilitySpec(
        model_base=VLMModel,
        input_type=VLMInput,
        output_type=VLMOutput,
        request_type=VLMRequest,
        response_type=VLMResponse,
        default_timeout_s=300.0,
        binary=BinaryUpload(file_field="images", payload_key="images"),
    ),
    CapabilitySpec(
        model_base=NERModel,
        input_type=NERInput,
        output_type=NEROutput,
        request_type=NERRequest,
        response_type=NERResponse,
    ),
    CapabilitySpec(
        model_base=EmbeddingModel,
        input_type=EmbeddingInput,
        output_type=EmbeddingOutput,
        request_type=EmbeddingRequest,
        response_type=EmbeddingResponse,
        binary=BinaryUpload(file_field="image", payload_key="image"),
    ),
)

_CAPABILITIES_BY_NAME: dict[str, CapabilitySpec] = {spec.name: spec for spec in CAPABILITIES}

HTTP_CAPABILITY_BASES: tuple[type[InferenceModel], ...] = tuple(spec.model_base for spec in CAPABILITIES)


def find_capability(name: str) -> CapabilitySpec | None:
    """Return the capability called *name*, or ``None`` if there is no such capability."""
    return _CAPABILITIES_BY_NAME.get(name)


def get_capability(name: str) -> CapabilitySpec:
    """Return the capability called *name*.

    Raises:
        KeyError: If there is no such capability.
    """
    spec = find_capability(name)
    if spec is None:
        raise KeyError(f"Unknown capability '{name}'. Known capabilities: {', '.join(_CAPABILITIES_BY_NAME)}.")
    return spec


def capability_of(model_cls: type[InferenceModel]) -> CapabilitySpec:
    """Return the capability a model class implements.

    Raises:
        ValueError: If the class subclasses none of the capability base classes.
    """
    for spec in CAPABILITIES:
        if issubclass(model_cls, spec.model_base):
            return spec

    supported = ", ".join(base.__name__ for base in HTTP_CAPABILITY_BASES)
    raise ValueError(
        f"Model class '{model_cls.__name__}' is not supported by the HTTP inference API. "
        f"Supported base classes: {supported}."
    )


def infer_http_capability(model_cls: type[InferenceModel]) -> str:
    """Infer the HTTP capability implemented by a model class."""
    return capability_of(model_cls).name
