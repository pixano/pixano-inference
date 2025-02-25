# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Provider for the SAM2 model."""

from .base import BaseProvider
from .registry import PROVIDERS_REGISTRY


def get_provider(provider: str) -> type[BaseProvider]:
    """Return the provider from the registry."""
    if (actual_provider := PROVIDERS_REGISTRY.get(provider)) is not None:
        return actual_provider
    raise ValueError(f"Provider {provider} not found.")


def get_providers() -> list[str]:
    """Return the list of providers in the registry."""
    return list(PROVIDERS_REGISTRY.keys())


def is_provider(provider: str) -> bool:
    """Return True if the provider is in the registry."""
    return provider in PROVIDERS_REGISTRY


def get_provider_name(provider: BaseProvider) -> str:
    """Return the name of the provider."""
    for name, provider_cls in PROVIDERS_REGISTRY.items():
        if isinstance(provider, provider_cls):
            return name
    raise ValueError(f"Provider {provider} not found.")


def instantiate_provider(provider: str) -> BaseProvider:
    """Instantiate a provider."""
    return get_provider(provider)()
