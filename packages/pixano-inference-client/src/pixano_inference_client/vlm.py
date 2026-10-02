# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of :mod:`pixano_inference.schemas.vlm`."""

# ruff: noqa: F401

from pixano_inference.schemas.vlm import (
    UsageInfo,
    VLMInput,
    VLMOutput,
)


__all__ = ["UsageInfo", "VLMInput", "VLMOutput"]
