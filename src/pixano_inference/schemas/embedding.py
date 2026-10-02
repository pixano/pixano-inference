# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Embedding I/O types.

A single capability that embeds **either** an image **or** text into a shared vector space
(CLIP-style), so image and text embeddings are directly comparable (text-to-image search).
"""

from pathlib import Path

from pydantic import model_validator

from .base import CamelModel
from .nd_array import NDArrayFloat


class EmbeddingInput(CamelModel):
    """Input for embedding computation.

    Exactly one of ``image`` or ``text`` must be provided. Either may be a single value or a
    list (batch). Images are passed by value (path/URL/base64/bytes) and resolved through the
    media security policy.

    Attributes:
        image: Image(s) to embed (path, URL, base64, or raw bytes).
        text: Text(s) to embed.
        normalize: Whether to L2-normalize the output vectors (default True).
    """

    image: list[str | Path | bytes] | str | Path | bytes | None = None
    text: list[str] | str | None = None
    normalize: bool = True

    @model_validator(mode="after")
    def _check_exactly_one_modality(self) -> "EmbeddingInput":
        has_image = self.image is not None
        has_text = self.text is not None
        if has_image == has_text:
            raise ValueError("Provide exactly one of 'image' or 'text'.")
        return self


class EmbeddingOutput(CamelModel):
    """Output for embedding computation.

    Attributes:
        embeddings: Embedding vectors as a ``[num_inputs, dim]`` array.
        dim: Dimensionality of each embedding vector.
    """

    embeddings: NDArrayFloat
    dim: int
