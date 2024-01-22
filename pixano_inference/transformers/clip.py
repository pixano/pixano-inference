# @Copyright: CEA-LIST/DIASI/SIALV/LVA (2023)
# @Author: CEA-LIST/DIASI/SIALV/LVA <pixano@cea.fr>
# @License: CECILL-C
#
# This software is a collaborative computer program whose purpose is to
# generate and explore labeled data for computer vision applications.
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
#
# http://www.cecill.info


import numpy as np
import pyarrow as pa
from pixano.core import Image
from pixano.models import InferenceModel
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast


class CLIP(InferenceModel):
    """CLIP: Connecting text and images

    Attributes:
        name (str): Model name
        model_id (str): Model ID
        device (str): Model GPU or CPU device (e.g. "cuda", "cpu")
        description (str): Model description
        model (CLIPModel): CLIP model
        processor (CLIPProcessor): CLIP processor
        tokenizer (CLIPTokenizerFast): CLIP tokenizer
        pretrained_model (str): Pretrained model name or path
    """

    def __init__(
        self,
        pretrained_model: str = "openai/clip-vit-base-patch32",
        model_id: str = "",
    ) -> None:
        """Initialize model

        Args:
            pretrained_model (str): Pretrained model name or path
            model_id (str, optional): Previously used ID, generate new ID if "". Defaults to "".
        """

        super().__init__(
            name="CLIP",
            model_id=model_id,
            device="cpu",
            description=f"From HuggingFace Transformers. CLIP: Connecting text and images. {pretrained_model}.",
        )

        # Model
        self.model = CLIPModel.from_pretrained(pretrained_model)
        self.processor = CLIPProcessor.from_pretrained(pretrained_model)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_model)

        # Model name or path
        self.pretrained_model = pretrained_model

    def precompute_embeddings(
        self,
        batch: pa.RecordBatch,
        views: list[str],
        uri_prefix: str,
    ) -> list[dict]:
        """Embedding precomputing for a batch

        Args:
            batch (pa.RecordBatch): Input batch
            views (list[str]): Dataset views
            uri_prefix (str): URI prefix for media files

        Returns:
            pa.RecordBatch: Embedding rows
        """

        rows = [
            {
                "id": batch["id"][x].as_py(),
            }
            for x in range(batch.num_rows)
        ]

        for view in views:
            # Iterate manually
            for x in range(batch.num_rows):
                # Preprocess image
                im = Image.from_dict(batch[view][x].as_py())
                im.uri_prefix = uri_prefix
                im = im.as_pillow()

                # Inference
                inputs = self.processor(images=im, padded=True, return_tensors="pt")
                image_features = self.model.get_image_features(**inputs)
                vect = image_features.detach().numpy()[0]

                # Process model outputs
                rows[x][view] = vect

        return rows

    def semantic_search(self, query: str) -> np.ndarray:
        """Process semantic search query with CLIP

        Args:
            query (str): Search query text

        Returns:
            np.ndarray: Search query vector
        """

        inputs = self.tokenizer([query], padding=True, return_tensors="pt")
        text_features = self.model.get_text_features(**inputs)

        return text_features.detach().numpy()[0]
