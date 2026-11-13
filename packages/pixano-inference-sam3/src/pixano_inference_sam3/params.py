# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Typed model params for SAM 3."""

from pixano_inference.configs.base import BaseModelParams, register_model_params


@register_model_params("Sam3DetectionModel")
class Sam3DetectionParams(BaseModelParams):
    """Typed parameters for ``Sam3DetectionModel``.

    Attributes:
        path: Hugging Face repository holding ``sam3.pt`` (and ``config.json``), or a local
            checkpoint file. The default, ``facebook/sam3``, is gated: the token in ``HF_TOKEN``
            must belong to an account whose access request was accepted.
        bpe_path: The text tokenizer's vocabulary (``bpe_simple_vocab_16e6.txt.gz``). ``None``
            fetches it once, hash-checked, next to the Hugging Face cache (see ``_bpe.py``).
        return_masks: Whether detections carry their instance masks.
        compile: Whether to ``torch.compile`` the model.
    """

    path: str = "facebook/sam3"
    bpe_path: str | None = None
    return_masks: bool = True
    compile: bool = False
