# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Optional ML-framework adapters (PyTorch, JAX, TensorFlow, MLX).

The core of pixano-inference depends on numpy only. Model backends reach their framework
through an adapter obtained from :func:`get_adapter`, keeping the framework an optional
extra and letting custom models be written against any supported framework.

Importing this package pulls in no ML framework; adapter modules and their frameworks are
imported lazily on first use.
"""

from .base import FrameworkAdapter, available_frameworks, get_adapter, register_adapter


__all__ = ["FrameworkAdapter", "available_frameworks", "get_adapter", "register_adapter"]
