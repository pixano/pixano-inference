# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Framework adapter protocol and registry.

pixano-inference is framework-agnostic: the core depends on numpy only, and each ML
framework (PyTorch, JAX, TensorFlow, MLX) is an optional extra reached through an
*adapter*. An adapter gives a model author framework-neutral helpers — device resolution,
dtype resolution, and numpy interchange — so a custom model can be written against any
supported framework without the core privileging one of them.

numpy is the interchange format: adapters convert their native arrays/tensors to and from
``numpy.ndarray`` (zero-copy via DLPack where the framework supports it).

This module imports NO framework at load time. Adapter modules (``torch``, ``jax``, …) are
imported lazily by :func:`get_adapter`, and each performs its own framework import lazily,
so ``import pixano_inference.frameworks`` stays dependency-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Framework-neutral operations a model backend needs.

    Implementations live in ``pixano_inference.frameworks.<name>`` and import their
    framework lazily.
    """

    name: str

    def is_available(self) -> bool:
        """Whether the underlying framework is importable in this environment."""
        ...

    def resolve_device(self, num_gpus: float) -> Any:
        """Return the framework's device object for *num_gpus* requested GPUs."""
        ...

    def resolve_dtype(self, dtype: str) -> Any:
        """Map a dtype string (e.g. ``"float32"``) to the framework's dtype."""
        ...

    def to_numpy(self, array: Any) -> "np.ndarray":
        """Convert a framework array/tensor to a numpy array (host memory)."""
        ...

    def from_numpy(self, array: "np.ndarray", device: Any = None, dtype: Any = None) -> Any:
        """Convert a numpy array to a framework array/tensor on *device* with *dtype*."""
        ...


# Registered adapters are the module paths to import lazily; importing the module registers
# its adapter instance via ``register_adapter``.
_ADAPTER_MODULES: dict[str, str] = {
    "torch": "pixano_inference.frameworks.torch",
    "jax": "pixano_inference.frameworks.jax",
    "tensorflow": "pixano_inference.frameworks.tensorflow",
    "mlx": "pixano_inference.frameworks.mlx",
}

_ADAPTERS: dict[str, FrameworkAdapter] = {}


def register_adapter(adapter: FrameworkAdapter) -> None:
    """Register a framework adapter instance (called by adapter modules on import)."""
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str) -> FrameworkAdapter:
    """Return the adapter for framework *name*, importing its module lazily.

    Args:
        name: Framework name (``"torch"``, ``"jax"``, ``"tensorflow"``, ``"mlx"``).

    Returns:
        The registered :class:`FrameworkAdapter`.

    Raises:
        ValueError: If *name* is not a known framework.
    """
    key = name.lower()
    if key not in _ADAPTERS:
        module_path = _ADAPTER_MODULES.get(key)
        if module_path is None:
            raise ValueError(f"Unknown framework '{name}'. Known: {sorted(_ADAPTER_MODULES)}.")
        import importlib

        importlib.import_module(module_path)
    return _ADAPTERS[key]


def available_frameworks() -> list[str]:
    """Return the names of frameworks whose package is importable in this environment."""
    available: list[str] = []
    for name in _ADAPTER_MODULES:
        try:
            if get_adapter(name).is_available():
                available.append(name)
        except Exception:
            continue
    return available
