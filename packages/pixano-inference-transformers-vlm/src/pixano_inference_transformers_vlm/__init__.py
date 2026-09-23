# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Hugging Face Transformers vision-language models, packaged as a Pixano Inference plugin.

Importing this package registers ``TransformersVLMModel`` (and its params), so it is the
target of the ``pixano_inference.models`` entry point. ``transformers`` is imported lazily
at ``load_model`` time, so discovery does not load it.
"""

from .model import TransformersVLMModel
from .params import TransformersVLMParams


__all__ = ["TransformersVLMModel", "TransformersVLMParams"]
