# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""TensorFlow framework adapter.

Provides framework-neutral device/dtype resolution and numpy interchange for custom models
written in TensorFlow. TensorFlow is imported lazily; importing this module does not
require it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pixano_inference.utils.package import is_tensorflow_installed

from .base import register_adapter


class TensorFlowAdapter:
    """:class:`~pixano_inference.frameworks.base.FrameworkAdapter` for TensorFlow."""

    name = "tensorflow"

    def is_available(self) -> bool:
        """Whether tensorflow is importable."""
        return is_tensorflow_installed()

    def resolve_device(self, num_gpus: float) -> Any:
        """Return a TensorFlow device string (``"/GPU:0"`` when requested and present)."""
        import tensorflow as tf

        if num_gpus > 0 and tf.config.list_physical_devices("GPU"):
            return "/GPU:0"
        return "/CPU:0"

    def resolve_dtype(self, dtype: str) -> Any:
        """Resolve a tf.DType from a string (e.g. ``"float32"``, ``"bfloat16"``)."""
        import tensorflow as tf

        try:
            return tf.dtypes.as_dtype(dtype)
        except TypeError as exc:
            raise ValueError(f"Unsupported tensorflow dtype '{dtype}'.") from exc

    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert a TensorFlow tensor to a numpy array."""
        return np.asarray(array)

    def from_numpy(self, array: np.ndarray, device: Any = None, dtype: Any = None) -> Any:
        """Convert a numpy array to a TensorFlow tensor on *device* with *dtype*."""
        import tensorflow as tf

        if device is not None:
            with tf.device(device):
                return tf.convert_to_tensor(array, dtype=dtype)
        return tf.convert_to_tensor(array, dtype=dtype)


register_adapter(TensorFlowAdapter())
