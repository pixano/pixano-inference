<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Server Deployment

Pixano-Inference uses [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
to deploy models as GPU-aware actors with built-in autoscaling, request batching,
and multi-model serving. It is framework-agnostic: you can deploy PyTorch, JAX,
TensorFlow, or any other Python-based model.

## Key features

- **GPU management** -- Per-model CPU/GPU resources in Python config files
- **Autoscaling** -- Scale-to-zero when idle, scale-up under load, per model
- **Request batching** -- Configurable batch size and wait timeout per deployment
- **Multi-model serving** -- Deploy multiple models in a single server, each with dedicated resources
- **Custom models** -- Bring any model by subclassing a base class
- **Python configuration** -- Typed `ModelConfig` declarations in `.py` files

## Architecture

```
Python config file (`models.py`)
    │
    ▼
InferenceServer
    │
    ├── Ray init (GPU/CPU resources)
    │
    ├── DeploymentManager
    │       │
    │       ├── ModelActor (sam2-image)     ← Ray actor, owns GPU
    │       ├── ModelActor (grounding-dino) ← Ray actor, owns GPU
    │       └── ModelActor (llava)          ← Ray actor, owns GPU
    │
    └── FastAPI app (uvicorn)
            │
            ├── /health
            ├── /app/models/
            ├── /tasks/image/mask_generation/
            ├── /tasks/video/mask_generation/
            ├── /tasks/multimodal/text-image/conditional_generation/
            ├── /tasks/image/zero_shot_detection/
            └── /tasks/image/instance_segmentation/
```

The `InferenceServer` reads a Python config file, initializes Ray, and deploys
each model as a separate Ray actor. The FastAPI application runs in-process via
uvicorn and routes inference requests to the appropriate actor.

## Next steps

- **[Quickstart](quickstart.md)** -- Deploy built-in models in 5 minutes.
- **[Custom Models Guide](custom_models.md)** -- Write and deploy your own models.
