# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Launcher of the ``pixano-inference`` command.

The command itself (:mod:`pixano_inference.main`) needs the server stack. This launcher imports
nothing from it until that stack is known to be installed, so a base install reports what is
missing instead of failing with an import traceback.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec


_SERVER_MODULES = ("typer", "ray", "fastapi", "uvicorn")


def main() -> None:
    """Run the server command, or explain how to install the server."""
    missing = [module for module in _SERVER_MODULES if find_spec(module) is None]
    if missing:
        sys.stderr.write(
            f"pixano-inference: the server is not installed (missing: {', '.join(missing)}).\n"
            'Install it with:  pip install "pixano-inference[server]"\n'
        )
        raise SystemExit(1)

    from pixano_inference.main import app

    app()
