# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Registry for providers."""

from pixano_inference.providers.base import BaseProvider


PROVIDERS_REGISTRY: dict[str, type[BaseProvider]] = {}


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
