# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

from pixano_inference.configs import DeploymentConfig, ModelConfig
from pixano_inference_sam import Sam2ImageParams, Sam2VideoParams

models = [
    ModelConfig(
        name="sam2-image",
        model_class="Sam2ImageModel",
        model_params=Sam2ImageParams(path="facebook/sam2-hiera-base-plus"),
        # Half a GPU each: both models share the single GPU on this box. Ray's num_gpus is a
        # scheduling fraction, not a memory cap, so each replica still sees the whole device.
        deployment=DeploymentConfig(num_gpus=0.5, num_cpus=8),
    ),
    ModelConfig(
        name="sam2-video",
        model_class="Sam2VideoModel",
        model_params=Sam2VideoParams(path="facebook/sam2-hiera-large"),
        deployment=DeploymentConfig(num_gpus=0.5, num_cpus=8),
    ),
]
