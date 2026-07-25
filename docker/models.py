# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Example deployment config for the default (GPU + SAM) Docker image.

Mounted at /config/models.py by docker-compose. Edit it (or mount your own) to deploy the
models you need. Weights are cached under /data/hf (the persistent volume).
"""

from pixano_inference_sam import Sam2ImageParams

from pixano_inference.configs import DeploymentConfig, ModelConfig


models = [
    ModelConfig(
        name="sam2-image",
        model_class="Sam2ImageModel",
        # compile=False avoids a slow first-inference torch.compile step; set it True for
        # steady-state throughput (the image ships a compiler so it works either way).
        model_params=Sam2ImageParams(path="facebook/sam2-hiera-base-plus", compile=False),
        deployment=DeploymentConfig(num_gpus=1, num_cpus=2, min_replicas=1, max_replicas=2),
    ),
]
