# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""CLIP-style embedding models packaged as a Pixano Inference plugin.

Importing this package registers ``OpenClipEmbeddingModel`` (and its params), so it is the
target of the ``pixano_inference.models`` entry point. ``open_clip`` is imported lazily at
``load_model`` time, so discovery does not require it.
"""

from .model import OpenClipEmbeddingModel
from .params import OpenClipParams


__all__ = ["OpenClipEmbeddingModel", "OpenClipParams"]
