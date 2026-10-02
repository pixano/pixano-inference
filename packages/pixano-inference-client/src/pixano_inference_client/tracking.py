# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of :mod:`pixano_inference.schemas.tracking`."""

# ruff: noqa: F401

from pixano_inference.schemas.tracking import (
    TrackingBoxPrompt,
    TrackingInput,
    TrackingInterval,
    TrackingKeyframe,
    TrackingOutput,
    TrackingPointPrompt,
)


__all__ = [
    "TrackingBoxPrompt",
    "TrackingInput",
    "TrackingInterval",
    "TrackingKeyframe",
    "TrackingOutput",
    "TrackingPointPrompt",
]
