<!---
# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================
--->

# Pixano Inference API reference

This section documents the public Python modules for the Ray Serve-based
Pixano-Inference API.

## Public modules

- `pixano_inference.client`
  Python client for the HTTP API.
- `pixano_inference.configs`
  Typed deployment configuration objects used in Python config files.
- `pixano_inference.models`
  Base classes, I/O models, and `register_model` for custom deployments.
- `pixano_inference.ray`
  Server bootstrap and Ray Serve integration.
- `pixano_inference.schemas`
  HTTP-layer request/response schemas and shared helper types.
- `pixano_inference.settings`
  Runtime settings exposed by the server.
- `pixano_inference.tasks`
  Task enums and task-string helpers.
- `pixano_inference.utils`
  Shared helper utilities.
