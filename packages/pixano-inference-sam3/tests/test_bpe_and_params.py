# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""The tokenizer vocabulary's location and integrity check, and the typed params (no network)."""

from __future__ import annotations

import pytest
from pixano_inference_sam3 import Sam3DetectionParams
from pixano_inference_sam3._bpe import default_bpe_path, ensure_bpe

from pixano_inference.configs import ModelConfig


def test_default_location_follows_hf_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    assert default_bpe_path() == tmp_path / "sam3" / "bpe_simple_vocab_16e6.txt.gz"


def test_a_vocabulary_with_the_wrong_hash_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "bpe.txt.gz"
    path.write_bytes(b"not the vocabulary")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("a present file is not downloaded"))

    with pytest.raises(ValueError, match="sha256"):
        ensure_bpe(path)


def test_params_defaults_and_resolution_through_the_registry():
    config = ModelConfig(name="sam3", model_class="Sam3DetectionModel", model_params={"return_masks": False})

    assert isinstance(config.model_params, Sam3DetectionParams)
    assert config.model_params.path == "facebook/sam3"
    assert config.model_params.return_masks is False
    assert config.capability == "detection"
