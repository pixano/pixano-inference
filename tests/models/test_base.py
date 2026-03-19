# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from pixano_inference.models import InferenceModel


class MockInput(BaseModel):
    value: str = "test"


class MockOutput(BaseModel):
    result: str


class MockModel(InferenceModel):
    def load_model(self) -> None:
        pass

    def predict(self, input: MockInput) -> MockOutput:
        return MockOutput(result="ok")


@pytest.fixture(scope="module")
def model():
    config = MagicMock()
    config.name = "test_model"
    config.capability = "mock"
    config.model_class = "MockModel"
    return MockModel(config)


class TestInferenceModel:
    def test_model_name(self, model: MockModel):
        assert model.model_name == "test_model"

    def test_capability(self, model: MockModel):
        assert model.capability == "mock"

    def test_metadata(self, model: MockModel):
        meta = model.metadata
        assert meta["model_name"] == "test_model"
        assert meta["capability"] == "mock"
        assert meta["model_class"] == "MockModel"

    def test_predict(self, model: MockModel):
        result = model.predict(MockInput())
        assert result == MockOutput(result="ok")
