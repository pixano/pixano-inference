# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of :mod:`pixano_inference.schemas.base`."""

# ruff: noqa: F401

from pixano_inference.schemas.base import BaseRequest, BaseResponse

# 0.1.0 exported the camelCase base model under this name; the core keeps it private.
from pixano_inference.schemas.base import _BaseModel as CamelModel


__all__ = ["BaseRequest", "BaseResponse", "CamelModel"]
