# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pixano_inference.impls.sam2.image import Sam2ImageModel
from pixano_inference.models.segmentation import SegmentationInput
from pixano_inference.ray.config import ModelDeploymentConfig
from pixano_inference.schemas.rle import CompressedRLE


class FakePredictor:
    def __init__(self) -> None:
        self.device = SimpleNamespace(type="cpu")
        self._is_image_set = True
        self._features = {
            "image_embed": None,
            "high_res_feats": [],
        }
        self.predict_calls: list[dict[str, object]] = []

    def reset_predictor(self) -> None:
        self._is_image_set = False

    def set_image(self, image: Image.Image) -> None:
        self._is_image_set = True

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return (
            np.array([[[1, 0], [0, 1]]], dtype=np.uint8),
            np.array([0.95], dtype=np.float32),
            np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
        )


@pytest.fixture
def sam2_model(monkeypatch: pytest.MonkeyPatch) -> tuple[Sam2ImageModel, FakePredictor]:
    import torch

    config = ModelDeploymentConfig(
        name="sam2-image",
        capability="segmentation",
        model_class="Sam2ImageModel",
    )
    model = Sam2ImageModel(config)
    predictor = FakePredictor()
    model._predictor = predictor
    model._torch_dtype = torch.float32

    monkeypatch.setattr(torch, "inference_mode", lambda: nullcontext())
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        "pixano_inference.utils.media.convert_string_to_image",
        lambda image: Image.new("RGB", (2, 2), color="black"),
    )

    return model, predictor


def test_predict_keeps_binary_masks_while_returning_low_res_logits(
    sam2_model: tuple[Sam2ImageModel, FakePredictor],
):
    model, predictor = sam2_model

    output = model.predict(
        SegmentationInput(
            model="sam2-image",
            image=b"binary-image",
            points=[[[1, 1]]],
            labels=[[1]],
            multimask_output=False,
            return_logits=True,
        )
    )

    assert predictor.predict_calls[0]["return_logits"] is False
    assert output.masks[0][0] == CompressedRLE.from_mask(np.array([[1, 0], [0, 1]], dtype=np.uint8))
    assert output.mask_logits is not None
    assert output.mask_logits.values == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert output.mask_logits.shape == [1, 2, 2]


def test_predict_accepts_native_low_res_mask_input_shape(
    sam2_model: tuple[Sam2ImageModel, FakePredictor],
):
    model, predictor = sam2_model

    first_output = model.predict(
        SegmentationInput(
            model="sam2-image",
            image=b"binary-image",
            points=[[[1, 1]]],
            labels=[[1]],
            multimask_output=False,
            return_logits=True,
        )
    )

    model.predict(
        SegmentationInput(
            model="sam2-image",
            image=b"binary-image",
            points=[[[1, 1], [2, 2]]],
            labels=[[1, 0]],
            mask_input=first_output.mask_logits,
            multimask_output=False,
            return_logits=True,
        )
    )

    mask_input = predictor.predict_calls[1]["mask_input"]
    assert isinstance(mask_input, np.ndarray)
    assert mask_input.shape == (1, 2, 2)
