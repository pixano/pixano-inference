# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the shared torch helpers."""

import numpy as np
import pytest
import torch
from PIL import Image
from pixano_inference_torch import (
    convert_image_pil_to_tensor,
    ndarray_to_tensor,
    resolve_device,
    resolve_device_from_num_gpus,
    resolve_torch_dtype,
    should_compile,
    tensor_to_ndarray,
    vector_to_tensor,
)

from pixano_inference.ray.config import ModelDeploymentConfig, ResourceConfig
from pixano_inference.schemas.nd_array import NDArrayFloat


def test_resolve_dtype():
    assert resolve_torch_dtype("float32") == torch.float32
    assert resolve_torch_dtype("bfloat16") == torch.bfloat16
    with pytest.raises(ValueError):
        resolve_torch_dtype("int8")


def test_cpu_when_no_gpu_requested():
    assert resolve_device_from_num_gpus(0) == torch.device("cpu")


def test_resolve_device_reads_the_deployment_resources():
    config = ModelDeploymentConfig(
        name="m", capability="detection", model_class="M", resources=ResourceConfig(num_gpus=0)
    )
    assert resolve_device(config) == torch.device("cpu")


@pytest.mark.parametrize(
    ("device", "requested", "expected"),
    [
        (torch.device("cpu"), None, False),
        ("cuda", None, True),
        ("cuda:0", None, True),
        (torch.device("cpu"), True, True),
        ("cuda", False, False),
    ],
)
def test_should_compile(device, requested, expected):
    assert should_compile(device, requested) is expected


def test_pil_to_tensor():
    tensor = convert_image_pil_to_tensor(Image.new("RGB", (4, 2), (255, 0, 0)), torch.device("cpu"))
    assert tensor.shape == (3, 2, 4)
    assert float(tensor[0].max()) == 1.0


def test_ndarray_bridge():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = ndarray_to_tensor(NDArrayFloat.from_numpy(arr))
    assert isinstance(tensor, torch.Tensor)
    np.testing.assert_array_equal(tensor.numpy(), arr)
    np.testing.assert_array_equal(tensor_to_ndarray(tensor).to_numpy(), arr)


def test_vector_to_tensor():
    arr = np.ones(3, dtype=np.float32)
    assert isinstance(vector_to_tensor(NDArrayFloat.from_numpy(arr)), torch.Tensor)
    t = torch.zeros(2)
    assert vector_to_tensor(t) is t
    with pytest.raises(ValueError):
        vector_to_tensor([1.0, 2.0])  # type: ignore[arg-type]
