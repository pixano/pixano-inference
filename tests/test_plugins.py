# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for entry-point plugin discovery of custom models."""

import pytest

from pixano_inference import plugins
from pixano_inference.models.registry import ModelClassRegistry
from pixano_inference.plugins import ENTRY_POINT_GROUP, ensure_models_loaded, load_plugin_models


def test_entry_point_group_name():
    assert ENTRY_POINT_GROUP == "pixano_inference.models"


def test_discovers_installed_example_plugin():
    """The editable-installed example package is discovered and registered by name."""
    pytest.importorskip("pixano_numpy_detector")
    result = load_plugin_models()
    assert "numpy_detector" in result["loaded"]
    assert result["failed"] == []
    assert ModelClassRegistry.has("NumpyDetector")


def test_ensure_models_loaded_registers_plugins():
    pytest.importorskip("pixano_numpy_detector")
    ensure_models_loaded(force=True)
    assert ModelClassRegistry.has("NumpyDetector")


def test_broken_plugin_is_skipped_not_fatal(monkeypatch):
    """A plugin whose load() raises is logged and skipped, not fatal."""

    class _BadEntryPoint:
        name = "broken"
        value = "does.not.exist"

        def load(self):
            raise ImportError("no such module")

    monkeypatch.setattr(plugins, "entry_points", lambda group: [_BadEntryPoint()])
    result = load_plugin_models()
    assert result["failed"] == ["broken"]
    assert result["loaded"] == []
