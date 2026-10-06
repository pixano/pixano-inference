# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""NER (Named Entity Recognition) I/O types."""

from .base import _BaseModel


class NERInput(_BaseModel):
    """Input for named entity recognition.

    Attributes:
        text: Text to analyse.
    """

    text: str


class NEREntity(_BaseModel):
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


class NEROutput(_BaseModel):
    """Output for named entity recognition.

    Attributes:
        entities: List of recognised entities.
    """

    entities: list[NEREntity]
