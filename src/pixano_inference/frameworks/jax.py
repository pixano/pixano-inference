# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""JAX framework adapter.

Provides framework-neutral device/dtype resolution and numpy interchange for custom models
written in JAX. JAX is imported lazily; importing this module does not require it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pixano_inference.utils.package import is_jax_installed

from .base import register_adapter


class JaxAdapter:
    """:class:`~pixano_inference.frameworks.base.FrameworkAdapter` for JAX."""

    name = "jax"

    def is_available(self) -> bool:
        """Whether jax is importable."""
        return is_jax_installed()

    def resolve_device(self, num_gpus: float) -> Any:
        """Return a JAX device: a GPU/TPU device when requested and present, else CPU."""
        import jax

        if num_gpus > 0:
            for platform in ("gpu", "tpu"):
                try:
                    devices = jax.devices(platform)
                except RuntimeError:
                    devices = []
                if devices:
                    return devices[0]
        return jax.devices("cpu")[0]

    def resolve_dtype(self, dtype: str) -> Any:
        """Resolve a jax.numpy dtype from a string (e.g. ``"float32"``, ``"bfloat16"``)."""
        import jax.numpy as jnp

        try:
            return jnp.dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"Unsupported jax dtype '{dtype}'.") from exc

    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert a JAX array to a numpy array (host memory)."""
        return np.asarray(array)

    def from_numpy(self, array: np.ndarray, device: Any = None, dtype: Any = None) -> Any:
        """Convert a numpy array to a JAX array on *device* with *dtype*."""
        import jax
        import jax.numpy as jnp

        result = jnp.asarray(array, dtype=dtype)
        if device is not None:
            result = jax.device_put(result, device)
        return result


register_adapter(JaxAdapter())
