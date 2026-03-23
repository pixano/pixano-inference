# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""SAM2 model implementations.

Importing this module triggers ``@register_model`` for:
- :class:`Sam2ImageModel` -- image segmentation
- :class:`Sam2VideoModel` -- video tracking
"""

from .image import Sam2ImageModel
from .video import Sam2VideoModel


__all__ = ["Sam2ImageModel", "Sam2VideoModel"]
