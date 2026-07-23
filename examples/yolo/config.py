# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""YOLO deployment configuration.

Install the plugin package first (``uv pip install -e examples/yolo``); the model is then
discovered via its entry point and referenced here by name — no import needed.

Usage::
    uv run pixano-inference --config examples/yolo/config.py
"""

from pixano_inference.configs import DeploymentConfig, ModelConfig


models = [
    ModelConfig(
        name="yolo26s",
        model_class="YOLOModel",
        model_params={"path": "yolo26s.pt"},
        deployment=DeploymentConfig(num_gpus=1, num_cpus=1, min_replicas=1, max_replicas=2),
    ),
]
