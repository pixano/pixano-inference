# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""vLLM-served vision-language models, packaged as a Pixano Inference plugin.

Importing this package registers ``VLLMVLMModel`` (and its params), so it is the target of
the ``pixano_inference.models`` entry point. ``vllm`` is imported lazily at ``load_model``
time, so discovery does not load it.
"""

from .model import VLLMVLMModel
from .params import VLLMVLMParams


__all__ = ["VLLMVLMModel", "VLLMVLMParams"]
