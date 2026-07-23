# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Embedding model base class and I/O types.

A single capability that embeds **either** an image **or** text into a shared vector space
(CLIP-style), so image and text embeddings are directly comparable (text-to-image search).
"""

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from pydantic import model_validator

from pixano_inference.schemas.base import CamelModel
from pixano_inference.schemas.nd_array import NDArrayFloat

from .base import InferenceModel


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


class EmbeddingModel(InferenceModel):
    """Base class for image/text embedding models (CLIP-style shared space).

    Example:
        ```python
        @register_model("my-embedder")
        class MyEmbedder(EmbeddingModel):
            def load_model(self):
                self.model = load_weights(self.config.model_params["path"])

            def predict(self, input: EmbeddingInput) -> EmbeddingOutput:
                vectors = self.model.encode(input.image or input.text)
                return EmbeddingOutput(embeddings=NDArrayFloat.from_numpy(vectors), dim=vectors.shape[-1])
        ```
    """

    capability_name: ClassVar[str] = "embedding"

    @abstractmethod
    def predict(self, input: EmbeddingInput) -> EmbeddingOutput:
        """Compute embeddings for the given image(s) or text(s).

        Args:
            input: Embedding input with exactly one of image/text.

        Returns:
            Embedding output with a ``[num_inputs, dim]`` vector array.
        """
