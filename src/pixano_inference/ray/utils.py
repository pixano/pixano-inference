# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Utility functions for Ray Serve infrastructure."""

import logging

from pixano_inference.utils.package import (
    is_sam2_installed,
    is_transformers_installed,
    is_vllm_installed,
)


logger = logging.getLogger(__name__)


def detect_optional_packages() -> list[str]:
    """Auto-detect installed optional dependencies for Ray workers.

    Returns:
        List of pip package names to install in Ray workers.
    """
    packages: list[str] = []

    if is_transformers_installed():
        packages.extend(["transformers", "accelerate", "torch"])

    if is_sam2_installed():
        packages.append("sam2")

    if is_vllm_installed():
        packages.append("vllm")

    return packages


_DEFAULT_EXCLUDES = [
    ".git",
    ".venv",
    "pyproject.toml",
    "uv.lock",
]
"""Paths excluded from Ray's automatic working-directory packaging.

Without these excludes, Ray auto-detects the local module directory and
uploads the entire project tree (including ``.git``, ``.venv``, and
``pyproject.toml``).  When workers extract the package, ``uv`` (if
present) sees the ``pyproject.toml`` and creates a fresh virtual
environment—often with a different Python version—causing workers to
hang during startup.
"""


def build_runtime_env(
    pip_packages: list[str] | None = None,
    working_dir: str | None = None,
    auto_detect: bool = True,
) -> dict | None:
    """Build a Ray runtime environment configuration.

    Args:
        pip_packages: Explicit list of pip packages to install. If None and
            auto_detect is True, packages will be auto-detected.
        working_dir: Working directory for Ray workers.
        auto_detect: Whether to auto-detect installed packages if pip_packages is None.

    Returns:
        Runtime environment dictionary for Ray, or None if empty.
    """
    env: dict = {}

    if pip_packages is None and auto_detect:
        pip_packages = detect_optional_packages()

    if pip_packages:
        env["pip"] = pip_packages

    if working_dir:
        env["working_dir"] = working_dir

    # Always set excludes to prevent Ray from packaging .git, .venv,
    # and project metadata files that cause uv to create new venvs
    # in worker processes.
    env["excludes"] = _DEFAULT_EXCLUDES

    return env if env else None
