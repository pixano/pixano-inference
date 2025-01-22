# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

# ruff: noqa: F401
# ruff: noqa: D104

from .image import ImageTask
from .multimodal import MultimodalImageNLPTask
from .nlp import NLPTask
from .task import Task
from .utils import STR_IMAGE_TASKS, STR_MULTIMODAL_TASKS, STR_NLP_TASKS, get_tasks, is_task, str_to_task
from .video import VideoTask
