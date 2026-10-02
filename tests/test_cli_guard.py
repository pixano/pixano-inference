# =================================
# Copyright: CEA-LIST/DIASI/SIALV
# Author : pixano@cea.fr
# License: CECILL-C
# =================================

"""The ``pixano-inference`` launcher explains a missing server stack instead of crashing."""

from __future__ import annotations

import sys
import types

import pytest

from pixano_inference import cli


def test_launcher_reports_the_missing_server_stack(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name in {"ray", "uvicorn"} else object())

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    message = capsys.readouterr().err
    assert "missing: ray, uvicorn" in message
    assert 'pip install "pixano-inference[server]"' in message


def test_launcher_runs_the_server_command_when_the_stack_is_installed(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    fake_main = types.ModuleType("pixano_inference.main")
    fake_main.app = lambda: calls.append("app")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pixano_inference.main", fake_main)
    monkeypatch.setattr(cli, "find_spec", lambda name: object())

    cli.main()

    assert calls == ["app"]
