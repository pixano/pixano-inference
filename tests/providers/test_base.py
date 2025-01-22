# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

from typing import TYPE_CHECKING, Any

import pytest

from pixano_inference.models import BaseInferenceModel
from pixano_inference.pydantic.tasks.multimodal.conditional_generation import (
    TextImageConditionalGenerationOutput,
)


if TYPE_CHECKING:
    from torch import Tensor


class MockModel(BaseInferenceModel):
    def __init__(self, name: str, provider: str):
        super().__init__(name, provider)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
        }

    def delete(self):
        pass

    def image_mask_generation(
        self, image: "Tensor", points: list[list[int]], boxes: list[list[int]], **kwargs: Any
    ) -> "Tensor":
        return ...

    def text_image_conditional_generation(
        self, prompt: str, image: "Tensor", **kwargs: Any
    ) -> TextImageConditionalGenerationOutput:
        pass


@pytest.fixture(scope="module")
def model():
    return MockModel("test_model", "test_provider")


class TestBaseInferenceModel:
    def test_metadata(self, model: MockModel):
        assert model.metadata == {
            "name": "test_model",
            "provider": "test_provider",
        }
