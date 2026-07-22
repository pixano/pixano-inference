# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Guardrail: the core package must not import any ML framework at module load.

pixano-inference is framework-agnostic. The core (``models``, ``schemas``, ``configs``,
``ray``, ``client``, ``utils``) depends on numpy only; PyTorch/JAX/TensorFlow/MLX are
optional extras resolved lazily inside ``pixano_inference.frameworks`` and ``impls``.

These tests run each import in a fresh subprocess so a framework imported elsewhere in the
test session (fixtures, other modules) cannot mask an eager import in the core.
"""

import subprocess
import sys


_FRAMEWORKS = ("torch", "tensorflow", "jax", "flax", "mlx")

_CORE_IMPORTS = (
    "import pixano_inference",
    "import pixano_inference.models",
    "import pixano_inference.schemas",
    "import pixano_inference.schemas.nd_array",
    "import pixano_inference.schemas.inference",
    "import pixano_inference.configs",
    "import pixano_inference.client",
    "import pixano_inference.utils",
    "import pixano_inference.ray",
    "import pixano_inference.ray.app",
    "import pixano_inference.ray.config",
    # The frameworks registry and each adapter module must import without their framework.
    "import pixano_inference.frameworks",
    "import pixano_inference.frameworks.torch",
    "import pixano_inference.frameworks.jax",
    "import pixano_inference.frameworks.tensorflow",
    "import pixano_inference.frameworks.mlx",
)


def _import_and_report(statements: list[str]) -> list[str]:
    """Import the given modules in a fresh interpreter, return frameworks in sys.modules."""
    frameworks = ", ".join(repr(f) for f in _FRAMEWORKS)
    code = (
        "import sys\n" + "\n".join(statements) + "\n" + f"leaked = [f for f in ({frameworks},) if f in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Core import failed:\n{result.stderr}"
    leaked = result.stdout.strip()
    return leaked.split(",") if leaked else []


def test_core_imports_pull_no_ml_framework():
    """Importing the whole core must not load torch/tf/jax/mlx."""
    leaked = _import_and_report(list(_CORE_IMPORTS))
    assert leaked == [], f"Core imports eagerly loaded ML framework(s): {leaked}"


def test_each_core_module_is_framework_free():
    """Each core module, imported alone, must not load an ML framework."""
    offenders = {}
    for stmt in _CORE_IMPORTS:
        leaked = _import_and_report([stmt])
        if leaked:
            offenders[stmt] = leaked
    assert not offenders, f"Modules that eagerly import an ML framework: {offenders}"
