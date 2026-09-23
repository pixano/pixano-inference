# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Utility functions for Ray Serve infrastructure."""

import logging


logger = logging.getLogger(__name__)


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

Because these excludes strip the project metadata, Ray must not relaunch workers
through ``uv run`` either — see ``pixano_inference.ray.app._disable_ray_uv_run_hook``,
which turns off Ray's ``uv run`` runtime-env hook before ``ray.init``.
"""


def build_runtime_env(
    pip_packages: list[str] | None = None,
    working_dir: str | None = None,
) -> dict | None:
    """Build a Ray runtime environment configuration.

    Args:
        pip_packages: Explicit list of pip packages to install in Ray workers.
        working_dir: Working directory for Ray workers.

    Returns:
        Runtime environment dictionary for Ray, or None if empty.
    """
    env: dict = {}

    if pip_packages:
        env["pip"] = pip_packages

    if working_dir:
        env["working_dir"] = working_dir

    # Always set excludes to prevent Ray from packaging .git, .venv,
    # and project metadata files that cause uv to create new venvs
    # in worker processes.
    env["excludes"] = _DEFAULT_EXCLUDES

    return env if env else None
