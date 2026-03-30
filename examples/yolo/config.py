# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""YOLOE deployment configuration.

Usage::
    PYTHONPATH=examples:$PYTHONPATH uv run pixano-inference --config examples/custom_yoloe/config.py
"""

from pixano_inference.configs import DeploymentConfig, ModelConfig
from yolo.model import YOLOModel


models = [
    ModelConfig(
        name="yolo26s",
        model_class=YOLOModel,
        model_params={"path": "yolo26s.pt"},
        deployment=DeploymentConfig(num_gpus=1, num_cpus=1, min_replicas=0, max_replicas=2),
    ),
]
