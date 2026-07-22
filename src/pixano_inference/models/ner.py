# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""NER (Named Entity Recognition) model base class and I/O types."""

from abc import abstractmethod
from typing import ClassVar

from pixano_inference.schemas.base import CamelModel

from .base import InferenceModel


class NERInput(CamelModel):
    """Input for named entity recognition.

    Attributes:
        text: Text to analyse.
    """

    text: str


class NEREntity(CamelModel):
    """A single recognized entity.

    Attributes:
        text: The entity text span.
        label: The entity label.
        start: Start character offset.
        end: End character offset.
        score: Confidence score.
    """

    text: str
    label: str
    start: int
    end: int
    score: float


class NEROutput(CamelModel):
    """Output for named entity recognition.

    Attributes:
        entities: List of recognised entities.
    """

    entities: list[NEREntity]


class NERModel(InferenceModel):
    """Base class for named entity recognition models.

    Example:
        ```python
        @register_model("my-ner")
        class MyNER(NERModel):
            def load_model(self):
                self.model = load_weights(self.config.model_params["path"])

            def predict(self, input: NERInput) -> NEROutput:
                return NEROutput(entities=[...])
        ```
    """

    capability_name: ClassVar[str] = "ner"

    @abstractmethod
    def predict(self, input: NERInput) -> NEROutput:
        """Run named entity recognition.

        Args:
            input: NER input with text to analyse.

        Returns:
            NER output with recognised entities.
        """
