<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Pixano-Inference

## Overview

Pixano-Inference serves multimodal inference models behind a REST API powered by
[Ray Serve](https://docs.ray.io/en/latest/serve/index.html). Models are declared
in Python config files, deployed as Ray actors, and invoked through either the
HTTP API or the Python client.

Key capabilities:

- **Python-config deployment** -- Declare models in `models.py` with typed validation
- **GPU-aware actors** -- Assign CPU/GPU resources per model deployment
- **Autoscaling and batching** -- Tune replicas and batch size per model
- **Multi-model serving** -- Run several deployments in one server
- **Custom models** -- Register your own `InferenceModel` subclasses

## How it works

1. **Write a Python config** with `pixano_inference.configs.ModelConfig`
2. **Start the server** with `pixano-inference --config models.py`
3. **Send requests** via the Python client or HTTP API

## Next steps

- **[Getting Started](../getting_started/index.md)** -- Install, configure, and run your first inference
- **[Server Deployment](../ray_serve/index.md)** -- Advanced configuration, autoscaling, and custom models
- **[HTTP API Reference](../api-reference.md)** -- Full endpoint and schema documentation

## Contributing

We welcome contributions from the community! Please open issues and pull requests for any bugs or feature requests you may have.

## License

Pixano-Inference is released under the terms of the CeCILL-C license.
