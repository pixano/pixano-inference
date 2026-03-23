# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Transformers model implementations.

Importing this module triggers ``@register_model`` for:
- :class:`GroundingDINOModel` -- zero-shot detection
- :class:`TransformersVLMModel` -- vision-language generation
"""

from .grounding_dino import GroundingDINOModel
from .vlm import TransformersVLMModel


__all__ = ["GroundingDINOModel", "TransformersVLMModel"]
