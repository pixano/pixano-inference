# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Client for the Pixano Inference /v1 API.

Canonical home: the standalone :mod:`pixano_inference_client` distribution (installable on its
own, without the server stack). Re-exported here so ``from pixano_inference.client import
PixanoInferenceClient`` keeps working for callers that install the full ``pixano-inference``.
"""

# ruff: noqa: F401

from pixano_inference_client.client import (
    DEFAULT_TIMEOUT,
    DEPLOY_TIMEOUT,
    TRACKING_TIMEOUT,
    PixanoInferenceClient,
    PixanoInferenceError,
    SyncPixanoInferenceClient,
)


__all__ = [
    "PixanoInferenceClient",
    "SyncPixanoInferenceClient",
    "PixanoInferenceError",
    "DEFAULT_TIMEOUT",
    "TRACKING_TIMEOUT",
    "DEPLOY_TIMEOUT",
]
