<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# pixano-inference-torch

PyTorch helpers shared by the torch-based Pixano Inference model packages
(`pixano-inference-sam`, `pixano-inference-clip`, `pixano-inference-grounding-dino`,
`pixano-inference-transformers-vlm`).

The Pixano Inference core is framework-free: it depends on no ML framework. A model package
that runs on PyTorch depends on this package for device/dtype resolution
(`resolve_device`, `resolve_torch_dtype`, `should_compile`), image-to-tensor conversion and
NDArray/tensor bridging. torch is imported lazily, so importing these helpers during plugin
discovery does not load torch.

## Development

The package is self-contained, with its own lock file:

```bash
cd packages/pixano-inference-torch
uv sync
uv run pytest
```
