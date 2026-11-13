# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""SAM 3 packaged as a Pixano Inference plugin.

Importing this package registers ``Sam3DetectionModel`` (and its typed params), so it is the
target of the ``pixano_inference.models`` entry point declared in ``pyproject.toml``.
"""

from .model import Sam3DetectionModel
from .params import Sam3DetectionParams


__all__ = ["Sam3DetectionModel", "Sam3DetectionParams"]
