# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

# ruff: noqa: F401
# ruff: noqa: D104

from .base import BaseProvider
from .registry import get_provider, get_providers, is_provider, register_provider
from .sam2 import Sam2Provider
from .transformers import TransformersProvider
from .vllm import VLLMProvider
