# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of :mod:`pixano_inference.schemas.base`."""

# ruff: noqa: F401

from pixano_inference.schemas.base import (
    BaseRequest,
    BaseResponse,
    CamelModel,
)


__all__ = ["BaseRequest", "BaseResponse", "CamelModel"]
