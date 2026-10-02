# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""YOLO + ByteTrack deployment configuration (multi-object tracking by detection).

Run the server from the plugin package's own environment, which holds the core plus
ultralytics; the model is discovered via its entry point and referenced here by name.

Usage::
    uv run --project examples/yolo pixano-inference --config examples/yolo/config_tracking.py
"""

from pixano_inference.configs import DeploymentConfig, ModelConfig


models = [
    ModelConfig(
        name="yolo-bytetrack",
        model_class="YOLOByteTrackModel",
        # "tracker" is an ultralytics tracker config; "bytetrack.yaml" is the default.
        model_params={"path": "yolo26n.pt", "tracker": "bytetrack.yaml"},
        deployment=DeploymentConfig(num_gpus=1, num_cpus=1, min_replicas=1, max_replicas=2),
    ),
]
