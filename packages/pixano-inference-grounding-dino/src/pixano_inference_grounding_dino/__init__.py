# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Grounding DINO zero-shot detection, packaged as a Pixano Inference plugin.

Importing this package registers ``GroundingDINOModel`` (and its params), so it is the
target of the ``pixano_inference.models`` entry point. ``transformers`` is imported lazily
at ``load_model`` time, so discovery does not load it.
"""

from .model import GroundingDINOModel
from .params import GroundingDINOParams


__all__ = ["GroundingDINOModel", "GroundingDINOParams"]
