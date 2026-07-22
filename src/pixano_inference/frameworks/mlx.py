# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""MLX framework adapter (Apple Silicon).

Provides framework-neutral device/dtype resolution and numpy interchange for custom models
written in MLX. MLX runs on Apple-Silicon GPUs (Metal) and CPU. MLX is imported lazily;
importing this module does not require it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pixano_inference.utils.package import is_mlx_installed

from .base import register_adapter


class MlxAdapter:
    """:class:`~pixano_inference.frameworks.base.FrameworkAdapter` for MLX."""

    name = "mlx"

    def is_available(self) -> bool:
        """Whether mlx is importable."""
        return is_mlx_installed()

    def resolve_device(self, num_gpus: float) -> Any:
        """Return an MLX device: the Metal GPU when requested and available, else CPU."""
        import mlx.core as mx

        if num_gpus > 0:
            try:
                if mx.metal.is_available():
                    return mx.gpu
            except Exception:
                pass
        return mx.cpu

    def resolve_dtype(self, dtype: str) -> Any:
        """Resolve an mlx dtype from a string (e.g. ``"float32"``, ``"bfloat16"``)."""
        import mlx.core as mx

        mapping = {
            "float32": mx.float32,
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported mlx dtype '{dtype}'. Choose from {list(mapping)}.")
        return mapping[dtype]

    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert an MLX array to a numpy array (host memory)."""
        return np.array(array)

    def from_numpy(self, array: np.ndarray, device: Any = None, dtype: Any = None) -> Any:
        """Convert a numpy array to an MLX array. MLX manages device placement implicitly."""
        import mlx.core as mx

        result = mx.array(array)
        if dtype is not None:
            result = result.astype(dtype)
        return result


register_adapter(MlxAdapter())
