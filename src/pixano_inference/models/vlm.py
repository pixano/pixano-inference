# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""VLM (Vision-Language Model) base class.

The I/O types live in :mod:`pixano_inference_client.vlm` and are re-exported here so
``from pixano_inference.models.vlm import VLMInput`` keeps working.
"""

from abc import abstractmethod
from typing import ClassVar

from pixano_inference_client.vlm import UsageInfo, VLMInput, VLMOutput  # noqa: F401

from .base import InferenceModel


class VLMModel(InferenceModel):
    """Base class for vision-language models.

    Example:
        ```python
        @register_model("my-vlm")
        class MyVLM(VLMModel):
            def load_model(self):
                self.model = load_weights(self.config.model_params["path"])

            def predict(self, input: VLMInput) -> VLMOutput:
                text = self.model.generate(input.prompt, input.images)
                return VLMOutput(generated_text=text, usage=..., generation_config=...)
        ```
    """

    capability_name: ClassVar[str] = "vlm"

    @abstractmethod
    def predict(self, input: VLMInput) -> VLMOutput:
        """Run vision-language generation.

        Args:
            input: VLM input with prompt, images, and generation parameters.

        Returns:
            VLM output with generated text, usage info, and generation config.
        """
