# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Task utilities."""

from .image import ImageTask
from .multimodal import MultimodalImageNLPTask
from .nlp import NLPTask
from .task import Task
from .video import VideoTask


STR_IMAGE_TASKS = {task.value for task in ImageTask}
STR_NLP_TASKS = {task.value for task in NLPTask}
STR_MULTIMODAL_TASKS = {task.value for task in MultimodalImageNLPTask}
STR_VIDEO_TASKS = {task.value for task in VideoTask}


def get_tasks() -> set[str]:
    """Get all tasks."""
    return {task for set in [STR_IMAGE_TASKS, STR_NLP_TASKS, STR_MULTIMODAL_TASKS, STR_VIDEO_TASKS] for task in set}


def is_task(task: str) -> bool:
    """Check if a task is valid."""
    return task in get_tasks()


def str_to_task(task: str) -> Task:
    """Convert a task string to its task."""
    if task in STR_IMAGE_TASKS:
        return ImageTask(task)
    elif task in STR_NLP_TASKS:
        return NLPTask(task)
    elif task in STR_MULTIMODAL_TASKS:
        return MultimodalImageNLPTask(task)
    elif task in STR_VIDEO_TASKS:
        return VideoTask(task)
    raise ValueError(f"Invalid task '{task}'")
