# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""VLM (Vision-Language Model) base class.

The I/O types live in :mod:`pixano_inference.schemas.vlm` and are re-exported here so
``from pixano_inference.models.vlm import VLMInput`` keeps working.
"""

from typing import ClassVar

from pixano_inference.schemas.vlm import UsageInfo, VLMInput, VLMOutput  # noqa: F401

from .base import InferenceModel


class VLMModel(InferenceModel[VLMInput, VLMOutput]):
    """Base class for vision-language models.

    ``predict`` receives a :class:`VLMInput` (prompt, images and generation parameters) and returns
    a :class:`VLMOutput` (generated text, usage and generation config).

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
