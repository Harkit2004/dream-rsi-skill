"""Checking on a run without resuming it (issue #73).

``python -m dream_rsi.run DIR`` was the only way to see a run's report, and it also
resumes the run: if the run is unfinished it starts the next cycle, and if another
process is still running it there are two writers. Under a harness the agent has to
ask how a long run is doing from a later tool call, and the question must not be able
to spend money or corrupt anything.

Every run here is a real one (``tests/holder.py``), so what is reported on is what a
run actually leaves on disk, live or dead.
"""

from __future__ import annotations

import signal
from pathlib import Path

import pytest
from holder import cli, contents, holding

from dream_rsi.lock import LOCK_FILENAME
from dream_rsi.run import (
    EXIT_FINISHED,
    EXIT_REFUSED,
    EXIT_RUNNING,
    EXIT_STOPPED,
    SESSION_FILENAME,
)


def _run(directory: Path, cycles: int) -> str:
    """A real run, to completion, and the report it printed."""
    completed = cli("--cycles", str(cycles), str(directory))
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_a_report_on_a_finished_run_is_the_table_the_run_printed_and_changes_nothing(
    tmp_path: Path,
) -> None:
    """Issue #73's first "tests first"."""
    directory = tmp_path / "run"
    printed = _run(directory, 2)
    before = contents(directory)

    reported = cli("--report", str(directory))

    assert reported.returncode == EXIT_FINISHED, reported.stderr
    assert reported.stdout.endswith(printed)
    assert "finished" in reported.stdout.splitlines()[0]
    assert contents(directory) == before


def test_a_report_on_a_directory_another_process_holds_says_it_is_still_running(
    tmp_path: Path,
) -> None:
    """Issue #73's second: the exit code says so, the report says which cycle, nothing changes."""
    directory = tmp_path / "run"
    _run(directory, 1)

    with holding(tmp_path, "--cycles", "2", str(directory)) as holder:
        before = contents(directory)

        reported = cli("--report", str(directory))

        assert reported.returncode == EXIT_RUNNING, reported.stderr
        status = reported.stdout.splitlines()[0]
        assert "running" in status
        assert "cycle 1 in progress" in status
        assert str(holder.pid) in status
        assert contents(directory) == before


def test_a_report_on_a_run_that_died_says_where_it_stopped_and_does_not_resume_it(
    tmp_path: Path,
) -> None:
    """Told apart from "still running" by nothing holding the directory, not by a guess."""
    directory = tmp_path / "run"
    _run(directory, 1)
    with holding(tmp_path, "--cycles", "3", str(directory)) as holder:
        holder.send_signal(signal.SIGKILL)
        holder.wait()
    before = contents(directory)

    reported = cli("--report", str(directory))

    assert reported.returncode == EXIT_STOPPED, reported.stderr
    status = reported.stdout.splitlines()[0]
    assert "stopped" in status
    assert "cycle 1" in status
    assert "1 of 3" in status
    assert contents(directory) == before


def test_a_directory_from_before_locks_and_sessions_is_reported_finished_and_left_as_it_was(
    tmp_path: Path,
) -> None:
    """Nothing says it stopped short, and looking at it must not add a lock file to it."""
    directory = tmp_path / "run"
    _run(directory, 2)
    (directory / SESSION_FILENAME).unlink()
    (directory / LOCK_FILENAME).unlink()
    before = contents(directory)

    reported = cli("--report", str(directory))

    assert reported.returncode == EXIT_FINISHED, reported.stderr
    assert contents(directory) == before


def test_a_refused_session_does_not_overwrite_what_the_last_one_asked_for(
    tmp_path: Path,
) -> None:
    """A resume under a changed configuration is refused before it records anything."""
    directory = tmp_path / "run"
    _run(directory, 2)
    recorded = (directory / SESSION_FILENAME).read_bytes()

    refused = cli("--cycles", "5", "--seed", "9", str(directory))

    assert refused.returncode == EXIT_REFUSED
    assert (directory / SESSION_FILENAME).read_bytes() == recorded


@pytest.mark.parametrize("kind", ["absent", "empty", "unrelated"])
def test_a_report_on_something_that_is_not_a_run_directory_says_so_and_creates_nothing(
    tmp_path: Path, kind: str
) -> None:
    directory = tmp_path / kind
    if kind != "absent":
        directory.mkdir()
    if kind == "unrelated":
        (directory / "notes.txt").write_text("not a run\n", encoding="utf-8")
    before = contents(tmp_path)

    reported = cli("--report", str(directory))

    assert reported.returncode == EXIT_REFUSED
    assert "not a run directory" in reported.stderr
    assert reported.stdout == ""
    assert contents(tmp_path) == before
