# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Multimodal tasks."""

from .task import Task


class MultimodalImageNLPTask(Task):
    """Multimodal tasks."""

    CAPTIONING = "image_captioning"
    CONDITIONAL_GENERATION = "text_image_conditional_generation"
    EMBEDDING = "text_image_embedding"
    MATCHING = "text_image_matching"
