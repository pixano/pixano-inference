# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Pydantic models for model configuration.

Canonical home: :mod:`pixano_inference_client.models_info`. Re-exported here so existing
``pixano_inference.schemas.models`` imports keep working.
"""

# ruff: noqa: F401

from pixano_inference_client.models_info import ModelInfo


__all__ = ["ModelInfo"]
