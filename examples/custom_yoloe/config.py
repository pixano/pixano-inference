# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""YOLOE deployment configuration.

Usage::

    pixano-inference --host 0.0.0.0 --port 8000
        --module-path examples
        --config examples/custom_yoloe/config.py
"""

from pixano_inference.configs import DeploymentConfig, ModelConfig


models = [
    ModelConfig(
        name="yoloe",
        model_class="YOLOEModel",
        model_module="custom_yoloe.model",
        model_params={"path": "yoloe-11s-seg.pt"},
        deployment=DeploymentConfig(num_gpus=0, num_cpus=1, min_replicas=0, max_replicas=2),
    ),
]
