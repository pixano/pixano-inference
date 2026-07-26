# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Framework-free deployment config (numpy example plugin).

Used by the CPU/CI smoke test: no torch, no GPU. Build the image with
``--build-arg INSTALL_EXAMPLE=true`` (and an empty ``TORCH_INDEX_URL``) to include the
``pixano-numpy-detector`` plugin, then run with this config.
"""

from pixano_inference.configs import DeploymentConfig, ModelConfig


models = [
    ModelConfig(
        name="numpy-detector",
        model_class="NumpyDetector",
        model_params={"threshold": 20},
        deployment=DeploymentConfig(num_gpus=0, num_cpus=1, min_replicas=1, max_replicas=1),
    ),
]
