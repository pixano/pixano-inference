# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""The text tokenizer's vocabulary, which the ``sam3`` 0.1.4 wheel leaves out.

``build_sam3_image_model`` looks for ``sam3/assets/bpe_simple_vocab_16e6.txt.gz`` inside the
installed package, and the wheel does not ship that directory. The file (CLIP's BPE vocabulary)
is fetched once from SAM 3's repository at a pinned commit, checked against its SHA-256, and kept
next to the Hugging Face cache, so a server started offline afterwards still finds it.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path


BPE_URL = "https://raw.githubusercontent.com/facebookresearch/sam3/86ed770/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"


def default_bpe_path() -> Path:
    """Where the vocabulary is kept: ``$HF_HOME/sam3/``, else ``~/.cache/huggingface/sam3/``."""
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(home) / "sam3" / "bpe_simple_vocab_16e6.txt.gz"


def ensure_bpe(path: str | Path | None = None) -> Path:
    """Return the vocabulary's path, downloading it first when it is missing.

    Args:
        path: Where the vocabulary is, or should be put. ``None`` means :func:`default_bpe_path`.

    Returns:
        The path of a vocabulary file with the expected hash.

    Raises:
        ValueError: The file there does not have the expected SHA-256.
    """
    target = Path(path) if path is not None else default_bpe_path()
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(BPE_URL, timeout=60) as response, open(partial, "wb") as handle:  # noqa: S310
            handle.write(response.read())
        os.replace(partial, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != BPE_SHA256:
        raise ValueError(f"{target}: sha256 {digest}, expected {BPE_SHA256} (SAM 3's tokenizer vocabulary).")
    return target
