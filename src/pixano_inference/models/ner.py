# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""NER (Named Entity Recognition) model base class and I/O types (stub)."""

from abc import abstractmethod

from pydantic import BaseModel

from .base import InferenceModel


class NERInput(BaseModel):
    """Input for named entity recognition.

    Attributes:
        text: Text to analyse.
    """

    text: str


class NEREntity(BaseModel):
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


class NEROutput(BaseModel):
    """Output for named entity recognition.

    Attributes:
        entities: List of recognised entities.
    """

    entities: list[NEREntity]


class NERModel(InferenceModel):
    """Base class for named entity recognition models (stub)."""

    @abstractmethod
    def predict(self, input: NERInput) -> NEROutput:
        """Run named entity recognition.

        Args:
            input: NER input with text to analyse.

        Returns:
            NER output with recognised entities.
        """
