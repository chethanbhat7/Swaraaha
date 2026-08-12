"""Tests for the data setup orchestrator (model.data.setup)."""

import subprocess
import sys

import pytest

import model.data.setup as setup


def _fake_run(captured):
    """Return a subprocess.run replacement that records commands."""

    def run(cmd, timeout=None):
        captured.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    return run


def test_help_prints_usage_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        setup.parse_args(["--help"])
    assert exc_info.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_default_runs_all_steps_with_no_extra_args(monkeypatch):
    captured = []
    monkeypatch.setattr(setup.subprocess, "run", _fake_run(captured))
    monkeypatch.setattr(sys, "argv", ["setup"])
    code = setup.main()
    assert code == 0
    cmds = [c for c in captured]
    assert len(cmds) == 3
    assert cmds[0] == [sys.executable, "-m", "model.data.download"]
    assert cmds[1] == [sys.executable, "-m", "model.data.merge"]
    assert cmds[2] == [sys.executable, "-m", "model.data.prepare"]


def test_force_flag_reaches_merge_and_prepare_not_download(monkeypatch):
    captured = []
    monkeypatch.setattr(setup.subprocess, "run", _fake_run(captured))
    monkeypatch.setattr(sys, "argv", ["setup", "--force"])
    code = setup.main()
    assert code == 0
    cmds = [c for c in captured]
    assert "--force" not in cmds[0]
    assert "--force" in cmds[1]
    assert "--force" in cmds[2]


def test_extra_args_passthrough_reaches_all_steps(monkeypatch):
    captured = []
    monkeypatch.setattr(setup.subprocess, "run", _fake_run(captured))
    monkeypatch.setattr(sys, "argv", ["setup", "--", "--cache-dir", "/tmp/x"])
    code = setup.main()
    assert code == 0
    cmds = [c for c in captured]
    assert all(cmd[-2:] == ["--cache-dir", "/tmp/x"] for cmd in cmds)


def test_force_plus_passthrough_combine(monkeypatch):
    captured = []
    monkeypatch.setattr(setup.subprocess, "run", _fake_run(captured))
    monkeypatch.setattr(sys, "argv", ["setup", "--force", "--", "--max-rows", "100"])
    code = setup.main()
    assert code == 0
    cmds = [c for c in captured]
    assert "--force" not in cmds[0]
    for cmd in cmds[1:]:
        assert "--force" in cmd
        assert cmd[-2:] == ["--max-rows", "100"]


def test_failing_step_returns_exit_code_1(monkeypatch):
    def fail(cmd, timeout=None):
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(setup.subprocess, "run", fail)
    monkeypatch.setattr(sys, "argv", ["setup"])
    code = setup.main()
    assert code == 1
