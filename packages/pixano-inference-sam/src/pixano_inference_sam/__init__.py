# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""SAM2 models packaged as a Pixano Inference plugin.

Importing this package registers the SAM2 image and video models (and their typed params),
so it is the target of the ``pixano_inference.models`` entry point declared in
``pyproject.toml``. Installing the package makes ``Sam2ImageModel`` and ``Sam2VideoModel``
available to the server by name.
"""

from .image import Sam2ImageModel
from .params import Sam2ImageParams, Sam2VideoParams
from .video import Sam2VideoModel


__all__ = ["Sam2ImageModel", "Sam2ImageParams", "Sam2VideoModel", "Sam2VideoParams"]
