# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Embedding model base class.

A single capability that embeds **either** an image **or** text into a shared vector space
(CLIP-style), so image and text embeddings are directly comparable (text-to-image search). The
I/O types live in :mod:`pixano_inference.schemas.embedding` and are re-exported here so
``from pixano_inference.models.embedding import EmbeddingInput`` keeps working.
"""

from typing import ClassVar

from pixano_inference.schemas.embedding import EmbeddingInput, EmbeddingOutput  # noqa: F401

from .base import InferenceModel


class EmbeddingModel(InferenceModel[EmbeddingInput, EmbeddingOutput]):
    """Base class for image/text embedding models (CLIP-style shared space).

    ``predict`` receives an :class:`EmbeddingInput` (exactly one of image or text) and returns an
    :class:`EmbeddingOutput` (a ``[num_inputs, dim]`` vector array).

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
