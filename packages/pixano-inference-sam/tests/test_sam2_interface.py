# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""What the SAM2 models declare on ``/v1/models`` (static: no weights, no torch import)."""

from __future__ import annotations

from pixano_inference_sam import Sam2ImageModel, Sam2VideoModel

from pixano_inference.configs import ModelDeploymentConfig
from pixano_inference.models import SegmentationInterface, TrackingInterface


def test_image_model_is_prompted_with_candidate_masks_and_reusable_embeddings():
    model = Sam2ImageModel(
        ModelDeploymentConfig(name="sam2-image", capability="segmentation", model_class="Sam2ImageModel")
    )

    assert model.interface == SegmentationInterface(
        prompts=["points", "box", "mask"], multimask=True, embeddings=True, outputs=["mask", "score", "logits"]
    )


def test_video_model_tracks_prompted_objects_only():
    model = Sam2VideoModel(
        ModelDeploymentConfig(name="sam2-video", capability="tracking", model_class="Sam2VideoModel")
    )

    declared = model.interface

    assert declared == TrackingInterface(
        prompts=["points", "box", "mask"], prompt_free=False, classes="none", interval=True, outputs=["mask"]
    )
    # A client picks it for prompted tracking, and never sends it a prompt-free request.
    assert declared.prompts and not declared.prompt_free
    assert declared.model_dump(by_alias=True)["promptFree"] is False
