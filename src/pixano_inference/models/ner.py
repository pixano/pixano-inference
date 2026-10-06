# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""NER (Named Entity Recognition) model base class.

The I/O types live in :mod:`pixano_inference.schemas.ner` and are re-exported here so
``from pixano_inference.models.ner import NERInput`` keeps working.
"""

from typing import ClassVar

from pixano_inference.schemas.ner import NEREntity, NERInput, NEROutput  # noqa: F401

from .base import InferenceModel


class NERModel(InferenceModel[NERInput, NEROutput]):
    """Base class for named entity recognition models.

    ``predict`` receives a :class:`NERInput` (the text to analyse) and returns a :class:`NEROutput`
    (the recognised entities).

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
