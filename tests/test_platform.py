"""The loop is POSIX-only, and says so instead of failing on a stdlib import (issue #67).

On native Windows ``pip install -e .`` succeeds and then ``python -m dream_rsi.run``
died at import time with ``ModuleNotFoundError: No module named 'resource'`` — a
message about a stdlib module rather than about the reason. The sandbox that runs
model-written policy code is POSIX by design (process resource limits, a process
group to kill, pipe descriptors handed to the child), and CLAUDE.md is explicit
that it is never run without those limits. So the fix is to refuse clearly, not
to fall back to running policy code unbounded.

These tests run on Linux in CI, so "Windows" is simulated at the two places the
failure showed itself: the import, and the platform check the CLI and the
sandbox consult.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dream_rsi import run, sandbox
from dream_rsi.sandbox import SandboxedPolicy, SandboxError

POLICY = """\
from dream_rsi.policy import BreadthFirstPolicy


class OptimalPolicy(BreadthFirstPolicy):
    pass
"""


def test_the_cli_imports_where_the_resource_module_does_not_exist() -> None:
    """The reported traceback: every module up to ``_sandbox_child`` imported ``resource``.

    ``sys.modules['resource'] = None`` makes ``import resource`` raise exactly as it
    does on Windows, so this fails on the unfixed package for the reported reason
    and passes once nothing needs ``resource`` until a sandbox actually starts.
    """
    hide_resource = "import sys; sys.modules['resource'] = None; import dream_rsi.run"

    completed = subprocess.run(
        [sys.executable, "-c", hide_resource],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr


def test_the_cli_refuses_a_non_posix_platform_in_one_line_and_starts_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #67's first "tests first": non-zero, the requirement named, no policy code run."""
    monkeypatch.setattr(sandbox, "_OS_NAME", "nt")

    def a_cycle_started(**_: object) -> None:
        raise AssertionError("a cycle started on a platform with no sandbox")

    monkeypatch.setattr(run, "run_cycles", a_cycle_started)
    directory = tmp_path / "run"

    code = run.main([str(directory)])

    captured = capsys.readouterr()
    # The status a command refused before it ran anything exits with, not merely
    # "some failure": that is what lets a caller tell "did not start" from a crash.
    assert code == run.EXIT_REFUSED
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1, captured.err
    assert "Linux" in lines[0]
    assert "macOS" in lines[0]
    assert "WSL" in lines[0]
    assert "Traceback" not in captured.err
    assert not directory.exists()


def test_starting_a_sandbox_off_posix_raises_sandbox_error_and_leaves_nothing_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #67's second: library callers never touch the CLI, and get ``SandboxError``.

    Not ``ImportError``, and not a policy run without its limits: the refusal names
    the platform requirement and no scratch directory is created for a sandbox that
    cannot exist.
    """
    monkeypatch.setattr(sandbox, "_OS_NAME", "nt")

    with pytest.raises(SandboxError, match="Linux or macOS"):
        SandboxedPolicy(POLICY, scratch_root=tmp_path)

    assert list(tmp_path.iterdir()) == []
