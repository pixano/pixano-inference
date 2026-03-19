# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""SAM2 deployment configuration for Pixano Inference.

Usage:
    pixano-inference --config deploy/sam2_example.py
"""

from pixano_inference.configs import DeploymentConfig, ModelConfig, Sam2ImageParams, Sam2VideoParams
from pixano_inference.impls.sam2.image import Sam2ImageModel
from pixano_inference.impls.sam2.video import Sam2VideoModel
from pixano_inference.tasks import ImageTask, VideoTask


models = [
    ModelConfig(
        name="sam2-image",
        task=ImageTask.MASK_GENERATION,
        model_class=Sam2ImageModel,
        model_params=Sam2ImageParams(path="facebook/sam2-hiera-base-plus", torch_dtype="float32"),
        deployment=DeploymentConfig(num_gpus=0, min_replicas=0, max_replicas=1, max_batch_size=8),
    ),
    ModelConfig(
        name="sam2-video",
        task=VideoTask.MASK_GENERATION,
        model_class=Sam2VideoModel,
        model_params=Sam2VideoParams(path="facebook/sam2-hiera-large", torch_dtype="float32"),
        deployment=DeploymentConfig(num_gpus=0, min_replicas=0, max_replicas=1, max_batch_size=1),
    ),
]
