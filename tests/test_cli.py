# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the `pixano-inference` command-line entry point."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import pixano_inference.ray as ray_module
from pixano_inference.main import app


@pytest.fixture
def captured_config(monkeypatch: pytest.MonkeyPatch):
    """Run the CLI without starting a server, capturing the RayServeConfig it built."""
    captured: dict = {}

    class _FakeServer:
        def __init__(self, config):
            captured["config"] = config

        def register_from_config(self, *args, **kwargs):
            captured["registered"] = args

        def start(self, *args, **kwargs):
            captured["started"] = kwargs

    monkeypatch.setattr(ray_module, "InferenceServer", _FakeServer)
    return captured


def test_num_gpus_reaches_the_ray_config(captured_config):
    result = CliRunner().invoke(app, ["--num-gpus", "2"])
    assert result.exit_code == 0, result.output
    assert captured_config["config"].num_gpus == 2


def test_num_gpus_defaults_to_autodetect(captured_config):
    """None means "let Ray decide", which is what RayServeConfig omits from ray.init()."""
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert captured_config["config"].num_gpus is None


def test_num_gpus_zero_is_accepted(captured_config):
    """0 is meaningful: it forces CPU-only placement rather than auto-detecting."""
    result = CliRunner().invoke(app, ["--num-gpus", "0"])
    assert result.exit_code == 0, result.output
    assert captured_config["config"].num_gpus == 0


def test_negative_num_gpus_is_rejected(captured_config):
    result = CliRunner().invoke(app, ["--num-gpus", "-1"])
    assert result.exit_code != 0
    assert "config" not in captured_config


def test_host_and_port_still_reach_the_config(captured_config):
    result = CliRunner().invoke(app, ["--host", "0.0.0.0", "--port", "9999", "--num-gpus", "1"])
    assert result.exit_code == 0, result.output
    config = captured_config["config"]
    assert (config.host, config.port, config.num_gpus) == ("0.0.0.0", 9999, 1)
    assert captured_config["started"]["host"] == "0.0.0.0"
