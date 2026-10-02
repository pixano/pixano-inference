# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of :mod:`pixano_inference.schemas.ner`."""

# ruff: noqa: F401

from pixano_inference.schemas.ner import (
    NEREntity,
    NERInput,
    NEROutput,
)


__all__ = ["NEREntity", "NERInput", "NEROutput"]
