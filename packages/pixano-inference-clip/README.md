<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# pixano-inference-clip

CLIP-style **image and text embedding** models for Pixano Inference, distributed as a
standalone plugin package. It serves the `embedding` capability: turn an image or text into a
vector in a **shared space**, so image and text embeddings are directly comparable (for
semantic search, text-to-image search, dedup, clustering).

Backed by [`open_clip`](https://github.com/mlfoundations/open_clip), so one plugin loads the
whole CLIP family by checkpoint spec. The default is **MobileCLIP2** — recent (Apple 2025),
performant, and CPU-friendly (no big GPU needed).

## Install

```bash
pip install pixano-inference[clip]
```

## Use

Reference the model by name in a config (the default checkpoint is `MobileCLIP2-S2` with the
`dfndr2b` weights):

```python
from pixano_inference.configs import DeploymentConfig, ModelConfig
from pixano_inference_clip import OpenClipParams

models = [
    ModelConfig(
        name="clip",
        model_class="OpenClipEmbeddingModel",
        model_params=OpenClipParams(),  # defaults to MobileCLIP2-S2 / dfndr2b
        deployment=DeploymentConfig(num_gpus=0, num_cpus=2),  # runs on CPU
    ),
]
```

```python
from pixano_inference.client import SyncPixanoInferenceClient
from pixano_inference.schemas import EmbeddingRequest

client = SyncPixanoInferenceClient("http://localhost:7463")

text_vec = client.embedding(EmbeddingRequest(model="clip", text="a photo of a cat"))
image_vec = client.embedding(EmbeddingRequest(model="clip", image="data:image/png;base64,..."))

# Same space: cosine similarity between text_vec and image_vec is meaningful.
print(text_vec.data.dim, text_vec.data.embeddings.to_numpy().shape)  # (dim, [1, dim])
```

Both `text` and `image` accept a list for batch embedding (output is `[N, dim]`). Vectors are
L2-normalized by default (`normalize=False` to disable).

## Other checkpoints

Any open_clip spec works — e.g. another size like `path="MobileCLIP2-S0"` (`pretrained="dfndr2b"`),
`path="ViT-B-32"` with `pretrained="laion2b_s34b_b79k"`, or an `hf-hub:<repo>` reference to an
open_clip-format repo (leave `pretrained=None` for those).
