# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Guardrail: the contract, the model API and the client import without the server stack.

A base install of pixano-inference has pydantic, numpy and httpx only. The modules an application
or a model package imports (``schemas``, ``models``, ``configs``, ``client``, ``plugins``,
``utils``) must therefore load neither the server runtime (Ray, FastAPI, uvicorn, typer) nor the
optional image libraries, and each layer must stay independent of the ones above it.

Each import runs in a fresh subprocess, so modules loaded elsewhere in the test session cannot
mask an eager import.
"""

import subprocess
import sys

import pytest


_SERVER_ONLY = (
    "ray",
    "fastapi",
    "starlette",
    "uvicorn",
    "typer",
    "pydantic_settings",
    "PIL",
    "pycocotools",
    "requests",
    "pixano_inference.ray",
    "pixano_inference.api",
    "pixano_inference.main",
    "pixano_inference.server_settings",
)

_LIGHT_MODULES = (
    "pixano_inference",
    "pixano_inference.schemas",
    "pixano_inference.models",
    "pixano_inference.configs",
    "pixano_inference.client",
    "pixano_inference.plugins",
    "pixano_inference.utils",
    "pixano_inference.cli",
)

# Layer rules: importing the module on the left must not load any module on the right.
_LAYER_RULES = (
    ("pixano_inference.schemas", ("pixano_inference.models", "pixano_inference.configs", "pixano_inference.client")),
    ("pixano_inference.client", ("pixano_inference.models", "pixano_inference.configs")),
    ("pixano_inference.models", ("pixano_inference.configs", "pixano_inference.client")),
)


def _loaded_after_import(module: str, candidates: tuple[str, ...]) -> list[str]:
    """Import *module* in a fresh interpreter and return the *candidates* found in sys.modules."""
    code = (
        "import sys\n"
        f"import {module}\n"
        f"candidates = {candidates!r}\n"
        "loaded = [c for c in candidates if c in sys.modules or any(m.startswith(c + '.') for m in sys.modules)]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Importing {module} failed:\n{result.stderr}"
    loaded = result.stdout.strip()
    return loaded.split(",") if loaded else []


@pytest.mark.parametrize("module", _LIGHT_MODULES)
def test_light_module_loads_no_server_dependency(module):
    loaded = _loaded_after_import(module, _SERVER_ONLY)
    assert loaded == [], f"Importing {module} loaded server-only module(s): {loaded}"


@pytest.mark.parametrize(("module", "forbidden"), _LAYER_RULES, ids=[rule[0] for rule in _LAYER_RULES])
def test_layer_does_not_import_the_layers_above(module, forbidden):
    loaded = _loaded_after_import(module, forbidden)
    assert loaded == [], f"Importing {module} loaded {loaded}, which belongs to another layer."
