# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Registry for providers."""

from pixano_inference.providers.base import BaseProvider


PROVIDERS_REGISTRY: dict[str, BaseProvider] = {}


def register_provider(provider: str):
    """Return a decorator to register a provider in the registry.

    Args:
        provider: Name of the provider.

    Returns:
        The decorator.
    """

    def decorator(cls):
        """Register the provider in the registry.

        Args:
            cls: Class to register.

        Returns:
            The class.
        """
        if provider in PROVIDERS_REGISTRY:
            raise ValueError(f"Provider {provider} already registered.")
        PROVIDERS_REGISTRY[provider] = cls
        return cls

    return decorator


def get_provider(provider: str) -> BaseProvider:
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
