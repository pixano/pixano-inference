# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""YOLO deployment configuration.

Run the server from the plugin package's own environment, which holds the core plus
ultralytics; the model is discovered via its entry point and referenced here by name — no
import needed.

Usage::
    uv run --project examples/yolo pixano-inference --config examples/yolo/config.py
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
