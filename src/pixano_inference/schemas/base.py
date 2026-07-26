# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic base models for request and response.

Canonical home: :mod:`pixano_inference_client.base`. Re-exported here so existing
``pixano_inference.schemas.base`` imports keep working.
"""

# ruff: noqa: F401

from pixano_inference_client.base import BaseRequest, BaseResponse, CamelModel


__all__ = ["CamelModel", "BaseRequest", "BaseResponse"]
