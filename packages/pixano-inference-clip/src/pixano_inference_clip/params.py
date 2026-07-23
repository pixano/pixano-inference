# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Typed params for the open_clip embedding model."""

from pixano_inference.configs.base import BaseModelParams, register_model_params


@register_model_params("OpenClipEmbeddingModel")
class OpenClipParams(BaseModelParams):
    """Typed parameters for ``OpenClipEmbeddingModel``.

    Attributes:
        path: open_clip model spec — an architecture name (default ``MobileCLIP2-S2``, paired
            with ``pretrained``) or an ``hf-hub:<repo>`` reference to an open_clip-format repo.
        pretrained: open_clip ``pretrained`` tag when ``path`` is an architecture name (default
            ``dfndr2b``, the released MobileCLIP2-S2 weights). Leave ``None`` for ``hf-hub:`` specs.
        compile: Whether to ``torch.compile`` the model.
    """

    path: str = "MobileCLIP2-S2"
    pretrained: str | None = "dfndr2b"
    compile: bool = False
