# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""A minimal, framework-free custom detection model.

``NumpyDetector`` needs no ML framework — it uses numpy only — which demonstrates that the
Pixano Inference model contract does not privilege any framework. It finds the bounding box
of the "foreground" (pixels that differ from the top-left background colour), so it produces
a real, deterministic detection from an image.

It is exposed to Pixano Inference through the ``pixano_inference.models`` entry point declared
in this package's ``pyproject.toml``; once the package is installed, the server discovers and
registers it automatically.
"""

from __future__ import annotations

import numpy as np
from pydantic import Field

from pixano_inference.configs import BaseModelParams, register_model_params
from pixano_inference.models import DetectionInput, DetectionModel, DetectionOutput, register_model
from pixano_inference.utils.media import convert_string_to_image


@register_model_params("NumpyDetector")
class NumpyDetectorParams(BaseModelParams):
    """Typed params for ``NumpyDetector``, validated when a config is built.

    Attributes:
        path: There is no checkpoint; this only names the deployment when ``name`` is omitted.
        threshold: Summed per-channel colour difference above which a pixel is foreground.
    """

    path: str = "numpy-detector"
    threshold: int = Field(default=20, ge=0)


@register_model("NumpyDetector")
class NumpyDetector(DetectionModel):
    """Detect the bounding box of the non-background region using numpy only."""

    def load_model(self) -> None:
        """Read the (optional) difference threshold from the deployment params."""
        self._threshold = int(self.config.model_params.get("threshold", 20))

    def predict(self, input: DetectionInput) -> DetectionOutput:
        """Return one box around pixels that differ from the top-left background colour."""
        image = np.asarray(convert_string_to_image(input.image), dtype=np.int16)  # (H, W, 3) RGB
        background = image[0, 0]
        foreground = np.abs(image - background).sum(axis=-1) > self._threshold

        ys, xs = np.where(foreground)
        if xs.size == 0:
            return DetectionOutput(boxes=[], scores=[], classes=[], masks=None)

        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        score = round(float(foreground.mean()), 4)
        return DetectionOutput(boxes=[box], scores=[score], classes=["object"], masks=None)
