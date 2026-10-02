# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Deprecated alias of :mod:`pixano_inference.client`."""

# ruff: noqa: F401

from pixano_inference.client import (
    DEFAULT_TIMEOUT,
    DEPLOY_TIMEOUT,
    TRACKING_TIMEOUT,
    PixanoInferenceClient,
    PixanoInferenceError,
    SyncPixanoInferenceClient,
)


__all__ = [
    "DEFAULT_TIMEOUT",
    "DEPLOY_TIMEOUT",
    "TRACKING_TIMEOUT",
    "PixanoInferenceClient",
    "PixanoInferenceError",
    "SyncPixanoInferenceClient",
]
