# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

import pytest

from pixano_inference.client import PixanoInferenceClient, SyncPixanoInferenceClient


@pytest.fixture
def simple_pixano_inference_client() -> PixanoInferenceClient:
    return PixanoInferenceClient(url="http://localhost:8081", max_retries=0)


@pytest.fixture
def sync_pixano_inference_client() -> SyncPixanoInferenceClient:
    return SyncPixanoInferenceClient(url="http://localhost:8081", max_retries=0)
