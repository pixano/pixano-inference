# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

import pytest

from pixano_inference.tasks import ImageTask, MultimodalImageNLPTask, NLPTask, VideoTask
from pixano_inference.tasks.utils import get_tasks, is_task, str_to_task


def test_is_task():
    assert is_task("image_classification")
    assert not is_task("not_a_task")


def test_get_tasks():
    tasks = get_tasks()
    assert len(tasks) == len(ImageTask) + len(NLPTask) + len(MultimodalImageNLPTask) + len(VideoTask)
    assert tasks == {task.value for task in ImageTask} | {task.value for task in NLPTask} | {
        task.value for task in MultimodalImageNLPTask
    } | {task.value for task in VideoTask}


def test_str_to_task():
    assert str_to_task("image_classification") == ImageTask.CLASSIFICATION
    assert str_to_task("token_classification") == NLPTask.TOKEN_CLASSIFICATION
    assert str_to_task("image_captioning") == MultimodalImageNLPTask.CAPTIONING
    assert str_to_task("video_mask_generation") == VideoTask.MASK_GENERATION

    with pytest.raises(ValueError, match="Invalid task 'not_a_task'"):
        str_to_task("not_a_task")
