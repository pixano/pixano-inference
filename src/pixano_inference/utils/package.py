# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Utility functions for working with Python packages."""

import importlib.util


def is_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed.

    Args:
        package_name: The name of the package to check.

    Returns:
        True if the package is installed, False otherwise
    """
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None


def assert_package_installed(package_name: str, error_message: str | None = None) -> None:
    """Assert that a Python package is installed.

    Args:
        package_name: The name of the package to check.
        error_message: The error message to raise if the package is not installed.

    Raises:
        ImportError: If the package is not installed
    """
    if not is_package_installed(package_name):
        message = error_message or f"Package '{package_name}' is not installed."
        raise ImportError(message)
