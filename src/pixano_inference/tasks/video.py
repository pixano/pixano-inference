# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Video tasks."""

from .task import Task


class VideoTask(Task):
    """Video tasks."""

    MASK_GENERATION = "video_mask_generation"
