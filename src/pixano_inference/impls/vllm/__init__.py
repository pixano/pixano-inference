# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""vLLM model implementations.

Importing this module triggers ``@register_model`` for:
- :class:`VLLMVLMModel` -- vLLM-based vision-language generation
"""

from .vlm import VLLMVLMModel


__all__ = ["VLLMVLMModel"]
