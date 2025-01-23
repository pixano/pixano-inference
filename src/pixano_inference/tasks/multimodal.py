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
    CONDITIONAL_GENERATION = "image_text_conditional_generation"
    EMBEDDING = "image_text_embedding"
    MATCHING = "image_text_matching"
    QUESTION_ANSWERING = "image_question_answering"
