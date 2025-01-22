# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Image tasks."""

from .task import Task


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
