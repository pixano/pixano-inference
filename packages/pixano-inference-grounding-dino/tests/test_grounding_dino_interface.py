# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""What Grounding DINO declares on ``/v1/models`` (static: no weights, no torch import)."""

from __future__ import annotations

from pixano_inference_grounding_dino import GroundingDINOModel

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models import DetectionInterface


def test_open_vocabulary_with_both_thresholds():
    model = GroundingDINOModel(
        ModelDeploymentConfig(name="gd", capability="detection", model_class="GroundingDINOModel")
    )

    assert model.interface == DetectionInterface(
        classes="open", class_names=None, thresholds=["box", "text"], outputs=["box", "score", "class"]
    )
