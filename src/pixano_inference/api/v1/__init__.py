# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Version 1 of the Pixano Inference HTTP API."""

from .router import register_v1_api


__all__ = ["register_v1_api"]
