# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""A framework-free (numpy-only) example custom model for Pixano Inference."""

from .model import NumpyDetector, NumpyDetectorParams


__all__ = ["NumpyDetector", "NumpyDetectorParams"]
