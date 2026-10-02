# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the YOLO example plugin (discovery only: no weights, no ultralytics import)."""

from pixano_inference.models.registry import ModelClassRegistry
from pixano_inference.plugins import load_plugin_models


def test_discovered_through_its_entry_point():
    result = load_plugin_models()
    assert {"yolo_detector", "yolo_bytetrack"} <= set(result["loaded"])
    assert result["failed"] == []
    assert ModelClassRegistry.has("YOLOModel")
    assert ModelClassRegistry.has("YOLOByteTrackModel")


def test_each_model_declares_its_capability():
    from pixano_yolo import YOLOByteTrackModel, YOLOModel

    from pixano_inference.models import infer_http_capability

    assert infer_http_capability(YOLOModel) == "detection"
    assert infer_http_capability(YOLOByteTrackModel) == "tracking"


def test_discovery_does_not_import_the_framework():
    """Importing the package (what plugin discovery does at startup) must not load torch/ultralytics."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import pixano_yolo\n"
        "print(','.join(m for m in ('torch', 'ultralytics') if m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"discovery imported: {result.stdout.strip()}"
