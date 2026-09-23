# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""Tests for the CLIP plugin's discovery and param defaults."""

import subprocess
import sys

from pixano_inference.models.registry import ModelClassRegistry
from pixano_inference.plugins import load_plugin_models


def test_discovers_clip_plugin():
    """Importing the plugin registers the model; ``open_clip`` is only needed at ``load_model``."""
    result = load_plugin_models()
    assert "clip" in result["loaded"]
    assert result["failed"] == []
    assert ModelClassRegistry.has("OpenClipEmbeddingModel")


def test_plugin_defaults_resolve_in_fresh_interpreter():
    """A config naming the model with no params still gets the plugin's defaults (cold start)."""
    script = (
        "from pixano_inference.configs import ModelConfig\n"
        "c = ModelConfig(name='clip', model_class='OpenClipEmbeddingModel')\n"
        "dep = c.to_deployment_config()\n"
        "assert dep.model_params['path'] == 'MobileCLIP2-S2', dep.model_params\n"
        "assert dep.model_params['pretrained'] == 'dfndr2b'\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_discovery_does_not_import_the_framework():
    """Importing the package (what plugin discovery does at startup) must not load torch/open_clip."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import pixano_inference_clip\n"
        "print(','.join(m for m in ('torch', 'open_clip') if m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"discovery imported: {result.stdout.strip()}"
