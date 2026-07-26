# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""NER (Named Entity Recognition) model base class.

The I/O types live in :mod:`pixano_inference_client.ner` and are re-exported here so
``from pixano_inference.models.ner import NERInput`` keeps working.
"""

from abc import abstractmethod
from typing import ClassVar

from pixano_inference_client.ner import NEREntity, NERInput, NEROutput  # noqa: F401

from .base import InferenceModel


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
