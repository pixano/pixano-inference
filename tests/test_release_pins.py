# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Guardrail: every package in the repository pins a core range that admits the current core.

A model package constrains ``pixano-inference`` to the minor version whose contract it was written
against (see RELEASING.md). When the core's minor version changes, each pin under ``packages/`` and
``examples/`` must follow; this test fails on the ones that were forgotten.
"""

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from pixano_inference.__version__ import __version__


tomllib = pytest.importorskip("tomllib")

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECTS = sorted([*_ROOT.glob("packages/*/pyproject.toml"), *_ROOT.glob("examples/*/pyproject.toml")])


def _core_requirements(pyproject: Path) -> list[Requirement]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    declared = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    requirements = [Requirement(dependency) for dependency in declared]
    return [requirement for requirement in requirements if canonicalize_name(requirement.name) == "pixano-inference"]


def test_repository_packages_are_discovered():
    assert _PYPROJECTS, "No package found under packages/ or examples/."


@pytest.mark.parametrize("pyproject", _PYPROJECTS, ids=lambda path: path.parent.name)
def test_package_pin_admits_current_core(pyproject):
    requirements = _core_requirements(pyproject)
    if not requirements:
        pytest.skip(f"{pyproject.parent.name} does not depend on pixano-inference.")
    for requirement in requirements:
        assert str(requirement.specifier), f"{pyproject.parent.name} leaves pixano-inference unpinned."
        assert requirement.specifier.contains(
            __version__, prereleases=True
        ), f"{pyproject.parent.name} pins '{requirement}', which excludes the core {__version__}."
