# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Ray Serve route modules."""

# ruff: noqa: F401

from .inference import register_inference_routes
from .service import register_service_routes
