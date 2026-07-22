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


def is_sam2_installed() -> bool:
    """Check if the sam2 package is installed.

    Returns:
        True if the sam2 package is installed, False otherwise
    """
    return is_package_installed("sam2")


def is_sam3_installed() -> bool:
    """Check if the sam3 package is installed.

    Returns:
        True if the sam3 package is installed, False otherwise
    """
    return is_package_installed("sam3")


def is_torch_installed() -> bool:
    """Check if the torch package is installed.

    Returns:
        True if the torch package is installed, False otherwise
    """
    return is_package_installed("torch")


def is_jax_installed() -> bool:
    """Check if the jax package is installed.

    Returns:
        True if the jax package is installed, False otherwise
    """
    return is_package_installed("jax")


def is_tensorflow_installed() -> bool:
    """Check if the tensorflow package is installed.

    Returns:
        True if the tensorflow package is installed, False otherwise
    """
    return is_package_installed("tensorflow")


def is_mlx_installed() -> bool:
    """Check if the mlx package is installed.

    Returns:
        True if the mlx package is installed, False otherwise
    """
    return is_package_installed("mlx")


def is_transformers_installed() -> bool:
    """Check if the transformers package is installed.

    Returns:
        True if the transformers package is installed, False otherwise
    """
    return is_package_installed("transformers")


def is_vllm_installed() -> bool:
    """Check if the vllm package is installed.

    Returns:
        True if the vLLM package is installed, False otherwise
    """
    return is_package_installed("vllm")


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


def assert_sam2_installed() -> None:
    """Assert that the sam2 package is installed."""
    assert_package_installed(
        "sam2",
        "sam2 is not installed. Please install it using 'pip install git+https://github.com/facebookresearch/sam2.git'.",
    )


def assert_sam3_installed() -> None:
    """Assert that the sam3 package is installed."""
    assert_package_installed(
        "sam3",
        "sam3 is not installed. Please install it using 'pip install pixano-inference[sam3]'.",
    )


def assert_transformers_installed() -> None:
    """Assert that the transformers package is installed."""
    assert_package_installed(
        "transformers",
        "transformers is not installed. Please install it using 'pip install pixano-inference[transformers]'.",
    )


def assert_torch_installed() -> None:
    """Assert that the torch package is installed."""
    assert_package_installed(
        "torch", "torch is not installed. Please install it using 'pip install pixano-inference[torch]'."
    )


def assert_jax_installed() -> None:
    """Assert that the jax package is installed."""
    assert_package_installed(
        "jax", "jax is not installed. Please install it using 'pip install pixano-inference[jax]'."
    )


def assert_tensorflow_installed() -> None:
    """Assert that the tensorflow package is installed."""
    assert_package_installed(
        "tensorflow",
        "tensorflow is not installed. Please install it using 'pip install pixano-inference[tensorflow]'.",
    )


def assert_mlx_installed() -> None:
    """Assert that the mlx package is installed."""
    assert_package_installed(
        "mlx", "mlx is not installed. Please install it using 'pip install pixano-inference[mlx]'."
    )


def assert_vllm_installed() -> None:
    """Assert that the vllm package is installed."""
    assert_package_installed(
        "vllm", "vLLM is not installed. Please install it using 'pip install pixano-inference[vllm]'."
    )
