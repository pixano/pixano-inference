# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Ultralytics YOLO as an installable Pixano Inference custom-model plugin.

Two models: ``YOLOModel`` (detection) and ``YOLOByteTrackModel`` (multi-object tracking by
detection, YOLO followed by ByteTrack).
"""

from .model import YOLOModel
from .tracker import YOLOByteTrackModel


__all__ = ["YOLOByteTrackModel", "YOLOModel"]
