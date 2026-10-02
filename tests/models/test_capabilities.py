# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""The capability table is consistent, and the routes, lookup and timeouts follow it."""

from __future__ import annotations

from typing import get_args, get_origin, get_type_hints

import pytest

from pixano_inference.api.v1.inference import build_inference_router
from pixano_inference.client import PixanoInferenceClient, SyncPixanoInferenceClient
from pixano_inference.models import (
    CAPABILITIES,
    HTTP_CAPABILITY_BASES,
    DetectionModel,
    EmbeddingModel,
    InferenceModel,
    NERModel,
    SegmentationModel,
    TrackingModel,
    VLMModel,
    capability_of,
    find_capability,
    get_capability,
    infer_http_capability,
)


_SPECS = pytest.mark.parametrize("spec", CAPABILITIES, ids=[spec.name for spec in CAPABILITIES])


def test_the_table_lists_each_capability_once():
    names = [spec.name for spec in CAPABILITIES]
    assert sorted(names) == ["detection", "embedding", "ner", "segmentation", "tracking", "vlm"]
    assert HTTP_CAPABILITY_BASES == (
        SegmentationModel,
        DetectionModel,
        TrackingModel,
        VLMModel,
        NERModel,
        EmbeddingModel,
    )


@_SPECS
def test_request_and_response_carry_the_capability_io_types(spec):
    assert get_type_hints(spec.request_type.to_input)["return"] is spec.input_type
    assert spec.response_type.model_fields["data"].annotation is spec.output_type
    assert "model" in spec.request_type.model_fields


@_SPECS
def test_model_base_is_parameterised_with_the_capability_io_types(spec):
    (generic_base,) = [base for base in spec.model_base.__orig_bases__ if get_origin(base) is InferenceModel]
    assert get_args(generic_base) == (spec.input_type, spec.output_type)


@_SPECS
def test_binary_upload_targets_a_field_of_the_request(spec):
    if spec.binary is None:
        pytest.skip(f"{spec.name} has no binary route.")
    assert spec.binary.payload_key in spec.request_type.model_fields


@_SPECS
def test_lookup_by_name_and_by_model_class(spec):
    class Model(spec.model_base):  # type: ignore[name-defined, misc]
        def load_model(self) -> None: ...

        def predict(self, input):
            return input

    assert get_capability(spec.name) is spec
    assert find_capability(spec.name) is spec
    assert capability_of(Model) is spec
    assert infer_http_capability(Model) == spec.name


def test_unknown_capability_and_unsupported_model_class():
    class Bare(InferenceModel):
        def load_model(self) -> None: ...

        def predict(self, input):
            return input

    assert find_capability("translation") is None
    with pytest.raises(KeyError, match="Unknown capability 'translation'"):
        get_capability("translation")
    with pytest.raises(ValueError, match="is not supported by the HTTP inference API"):
        capability_of(Bare)


def test_default_timeouts():
    timeouts = {spec.name: spec.default_timeout_s for spec in CAPABILITIES}
    assert timeouts == {
        "segmentation": 60.0,
        "detection": 60.0,
        "tracking": 600.0,
        "vlm": 300.0,
        "ner": 60.0,
        "embedding": 60.0,
    }


def test_routes_are_registered_from_the_table():
    router = build_inference_router(deployment_manager=None)  # type: ignore[arg-type]
    routes = {route.path: route for route in router.routes}

    expected = {f"/inference/{spec.name}" for spec in CAPABILITIES}
    expected |= {f"/inference/{spec.name}/binary" for spec in CAPABILITIES if spec.binary is not None}
    assert set(routes) == expected

    for spec in CAPABILITIES:
        route = routes[f"/inference/{spec.name}"]
        # The route name feeds the OpenAPI operationId, so it is part of the published contract.
        assert route.name == spec.name
        assert route.response_model is spec.response_type
        assert route.body_field is not None and route.body_field.field_info.annotation is spec.request_type
        if spec.binary is not None:
            assert routes[f"/inference/{spec.name}/binary"].name == f"{spec.name}_binary"


@_SPECS
def test_clients_expose_a_method_per_capability(spec):
    assert callable(getattr(PixanoInferenceClient, spec.name))
    assert callable(getattr(SyncPixanoInferenceClient, spec.name))
