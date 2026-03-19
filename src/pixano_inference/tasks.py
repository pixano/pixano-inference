# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Task definitions for pixano inference.

All supported task types are defined here as enums. Use ``str_to_task()``
to convert a task string to the corresponding enum member.
"""

from enum import Enum


class Task(Enum):
    """Task base class."""

    pass


class ImageTask(Task):
    """Image tasks."""

    CLASSIFICATION = "image_classification"
    DEPTH_ESTIMATION = "depth_estimation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    FEATURE_EXTRACTION = "image_feature_extraction"
    KEYPOINT_DETECTION = "keypoint_detection"
    MASK_GENERATION = "image_mask_generation"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    UNIVERSAL_SEGMENTATION = "universal_segmentation"
    ZERO_SHOT_CLASSIFICATION = "image_zero_shot_classification"
    ZERO_SHOT_DETECTION = "image_zero_shot_detection"


class MultimodalImageNLPTask(Task):
    """Multimodal tasks."""

    CAPTIONING = "image_captioning"
    CONDITIONAL_GENERATION = "text_image_conditional_generation"
    EMBEDDING = "text_image_embedding"
    MATCHING = "text_image_matching"


class NLPTask(Task):
    """Natural Language Processing tasks."""

    CAUSAL_LM = "causal_lm"
    CONDITONAL_GENERATION = "text_conditional_generation"
    MASKED_LM = "masked_lm"
    MASK_GENERATION = "text_mask_generation"
    MULTIPLE_CHOICE = "multiple_choice"
    NEXT_SENTENCE_PREDICTION = "next_sentence_prediction"
    QUESTION_ANSWERING = "question_answering"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TEXT_ENCODING = "text_encoding"
    TOKEN_CLASSIFICATION = "token_classification"


class VideoTask(Task):
    """Video tasks."""

    MASK_GENERATION = "video_mask_generation"


# ---------------------------------------------------------------------------
# Derived sets for convenience
# ---------------------------------------------------------------------------

STR_IMAGE_TASKS = {task.value for task in ImageTask}
STR_NLP_TASKS = {task.value for task in NLPTask}
STR_MULTIMODAL_TASKS = {task.value for task in MultimodalImageNLPTask}
STR_VIDEO_TASKS = {task.value for task in VideoTask}


def get_tasks() -> set[str]:
    """Get all tasks."""
    return {task for s in [STR_IMAGE_TASKS, STR_NLP_TASKS, STR_MULTIMODAL_TASKS, STR_VIDEO_TASKS] for task in s}


def is_task(task: str) -> bool:
    """Check if a task is valid."""
    return task in get_tasks()


def str_to_task(task: str) -> Task:
    """Convert a task string to its task enum.

    Args:
        task: Task string (e.g. ``"image_mask_generation"``).

    Returns:
        The corresponding ``Task`` enum member.

    Raises:
        ValueError: If the task string is not recognised.
    """
    if task in STR_IMAGE_TASKS:
        return ImageTask(task)
    elif task in STR_NLP_TASKS:
        return NLPTask(task)
    elif task in STR_MULTIMODAL_TASKS:
        return MultimodalImageNLPTask(task)
    elif task in STR_VIDEO_TASKS:
        return VideoTask(task)
    raise ValueError(f"Invalid task '{task}'")
