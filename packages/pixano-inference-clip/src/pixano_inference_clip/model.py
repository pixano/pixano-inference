# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""CLIP-style image/text embedding model backed by open_clip.

``open_clip`` gives one API over the CLIP family (MobileCLIP, OpenCLIP, MetaCLIP, ...), so
this single plugin loads any of them by checkpoint spec — defaulting to MobileCLIP2, which is
recent, performant, and runs comfortably on CPU. Images and text are embedded into the same
vector space (so they are directly comparable for text-to-image search).
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

from pixano_inference_torch import resolve_device

from pixano_inference.models.embedding import EmbeddingInput, EmbeddingModel, EmbeddingOutput
from pixano_inference.models.registry import register_model
from pixano_inference.ray.config import ModelDeploymentConfig
from pixano_inference.schemas.nd_array import NDArrayFloat


logger = logging.getLogger(__name__)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


@register_model("OpenClipEmbeddingModel")
class OpenClipEmbeddingModel(EmbeddingModel):
    """Image/text embedding model using open_clip (default MobileCLIP2).

    ``model_params`` contract:

    - ``path`` (str): open_clip model spec — an architecture name (default ``MobileCLIP2-S2``,
      used with ``pretrained``) or an ``hf-hub:<repo>`` reference to an open_clip-format repo.
    - ``pretrained`` (str, optional): open_clip pretrained tag for architecture-name specs
      (default ``dfndr2b``, the released MobileCLIP2-S2 weights).
    - ``compile`` (bool, default False): whether to ``torch.compile`` the model.
    """

    def __init__(self, config: ModelDeploymentConfig) -> None:
        """Initialize the model."""
        super().__init__(config)
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

    def load_model(self) -> None:
        """Load the open_clip model, preprocessing transform, and tokenizer."""
        from pixano_inference.utils.package import assert_package_installed

        assert_package_installed("open_clip", "open_clip is not installed. Install pixano-inference-clip.")

        import open_clip
        import torch

        params = dict(self._config.model_params)
        spec = params.pop("path")
        pretrained = params.pop("pretrained", None)
        compile_model = params.pop("compile", False)

        self._device = resolve_device(self._config)

        model, _, preprocess = open_clip.create_model_and_transforms(spec, pretrained=pretrained, device=self._device)
        model = model.eval()
        if compile_model:
            model = torch.compile(model)

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(spec)
        logger.info("OpenClipEmbeddingModel '%s' loaded (%s) on %s", self.model_name, spec, self._device)

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata including the checkpoint spec and device."""
        base = super().metadata
        base["path"] = self._config.model_params.get("path")
        if self._device is not None:
            base["device"] = str(self._device)
        return base

    def predict(self, input: EmbeddingInput) -> EmbeddingOutput:
        """Embed the input image(s) or text(s) into the shared CLIP space.

        Args:
            input: Embedding input with exactly one of image/text (single or list).

        Returns:
            Embedding output with a ``[num_inputs, dim]`` vector array.
        """
        import torch

        with torch.inference_mode():
            if input.text is not None:
                features = self._encode_text(_as_list(input.text))
            else:
                features = self._encode_image(_as_list(input.image))

            if input.normalize:
                features = features / features.norm(dim=-1, keepdim=True)

            array = features.detach().to(torch.float32).cpu().numpy()

        return EmbeddingOutput(embeddings=NDArrayFloat.from_numpy(array), dim=int(array.shape[-1]))

    def _encode_text(self, texts: list[str]) -> Any:
        tokens = self._tokenizer(texts).to(self._device)
        return self._model.encode_text(tokens)

    def _encode_image(self, images: list[str | Path | bytes]) -> Any:
        import torch

        from pixano_inference.utils.media import convert_string_to_image

        tensors = [self._preprocess(convert_string_to_image(image)) for image in images]
        batch = torch.stack(tensors).to(self._device)
        return self._model.encode_image(batch)

    def unload(self) -> None:
        """Free resources."""
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
