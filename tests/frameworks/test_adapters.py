# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the framework adapter registry and the torch adapter."""

import numpy as np
import pytest

from pixano_inference.frameworks import available_frameworks, get_adapter
from pixano_inference.schemas.nd_array import NDArrayFloat


def test_registry_knows_all_frameworks():
    for name in ("torch", "jax", "tensorflow", "mlx"):
        adapter = get_adapter(name)
        assert adapter.name == name


def test_unknown_framework_raises():
    with pytest.raises(ValueError):
        get_adapter("theano")


def test_available_frameworks_returns_installed_only():
    available = available_frameworks()
    assert isinstance(available, list)
    # torch is a dev dependency, so it must be reported available here.
    assert "torch" in available


def test_ndarray_wire_type_is_numpy_only():
    """The core wire type exposes numpy conversions and no framework-specific ones."""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    nd = NDArrayFloat.from_numpy(arr)
    np.testing.assert_array_equal(nd.to_numpy(), arr)
    assert not hasattr(nd, "to_torch")
    assert not hasattr(nd, "from_torch")


@pytest.mark.parametrize("num_gpus", [0.0, 1.0])
def test_torch_adapter_roundtrip(num_gpus):
    pytest.importorskip("torch")
    import torch

    adapter = get_adapter("torch")
    assert adapter.is_available() is True

    device = adapter.resolve_device(num_gpus)
    assert isinstance(device, torch.device)
    assert adapter.resolve_dtype("float32") == torch.float32

    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = adapter.from_numpy(arr, dtype="float32")
    assert isinstance(tensor, torch.Tensor)
    np.testing.assert_array_equal(adapter.to_numpy(tensor), arr)


def test_torch_ndarray_bridge():
    pytest.importorskip("torch")
    import torch

    from pixano_inference.frameworks.torch import ndarray_to_tensor, tensor_to_ndarray

    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    nd = NDArrayFloat.from_numpy(arr)
    tensor = ndarray_to_tensor(nd)
    assert isinstance(tensor, torch.Tensor)
    np.testing.assert_array_equal(tensor.cpu().numpy(), arr)

    nd_back = tensor_to_ndarray(tensor)
    np.testing.assert_array_equal(nd_back.to_numpy(), arr)
