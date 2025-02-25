# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

# ruff: noqa: F401
# ruff: noqa: D104

from .base import BaseProvider
from .registry import register_provider
from .sam2 import Sam2Provider
from .transformers import TransformersProvider
from .utils import get_provider, get_providers, instantiate_provider, is_provider
from .vllm import VLLMProvider
