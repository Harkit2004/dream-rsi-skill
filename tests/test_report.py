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

import os
import signal
from dataclasses import replace
from pathlib import Path

import pytest
from holder import cli, contents, holding

from dream_rsi import run
from dream_rsi.lock import LOCK_FILENAME
from dream_rsi.run import (
    CYCLES_DIRNAME,
    EXIT_FINISHED,
    EXIT_REFUSED,
    EXIT_RUNNING,
    EXIT_STOPPED,
    MANIFEST_FILENAME,
    POOL_DIRNAME,
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
    # One worker: with two, the second can still be setting up its workspace after the
    # first has announced itself, and the directory would change under the comparison.
    assert cli("--workers", "1", "--cycles", "1", str(directory)).returncode == 0

    with holding(tmp_path, "--workers", "1", "--cycles", "2", str(directory)) as holder:
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


@pytest.mark.parametrize("kind", ["stray-manifest", "empty-cycles"])
def test_a_directory_that_only_shares_a_run_file_name_is_not_a_run_directory(
    tmp_path: Path, kind: str
) -> None:
    """A ``manifest.json`` or a ``cycles/`` that no run wrote says nothing about a run."""
    directory = tmp_path / kind
    directory.mkdir()
    if kind == "stray-manifest":
        (directory / MANIFEST_FILENAME).write_text('{"name": "my-package"}\n', encoding="utf-8")
    else:
        (directory / CYCLES_DIRNAME).mkdir()
    before = contents(tmp_path)

    reported = cli("--report", str(directory))

    assert reported.returncode == EXIT_REFUSED
    assert "not a run directory" in reported.stderr
    assert contents(tmp_path) == before


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root reads any file")
def test_a_lock_the_report_cannot_read_is_a_refusal_and_not_a_finished_run(
    tmp_path: Path,
) -> None:
    """Not knowing whether a process holds the run is not the same as knowing none does."""
    directory = tmp_path / "run"
    _run(directory, 1)
    (directory / LOCK_FILENAME).chmod(0)

    try:
        reported = cli("--report", str(directory))
    finally:
        (directory / LOCK_FILENAME).chmod(0o644)

    assert reported.returncode == EXIT_REFUSED
    assert "status: finished" not in reported.stdout
    assert "Traceback" not in reported.stderr


def test_a_run_that_finishes_while_it_is_being_read_is_reported_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The last cycle lands, and the writer lets go, between reading the records and the lock.

    The records read before the lock was seen free are a snapshot of a run still going;
    once nothing holds it, they are read again, because now nothing can change them.
    """
    directory = tmp_path / "run"
    _run(directory, 2)
    real_load = run.Run.load
    finished = {"yet": False}

    def load(path: Path) -> run.Run:
        loaded = real_load(path)
        if not finished["yet"]:
            # read before cycle 1 landed
            loaded = replace(loaded, cycles=loaded.cycles[:1], policies=loaded.policies[:1])
            finished["yet"] = True  # ... and the writer finished while it was being read
        return loaded

    monkeypatch.setattr(run.Run, "load", staticmethod(load))
    monkeypatch.setattr(run, "probe", lambda path: (not finished["yet"], None))

    code = run.main(["--report", str(directory)])

    assert code == EXIT_FINISHED, capsys.readouterr().out
    assert "status: finished (2 cycle(s))" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["pool-tree", "policy-bytes"])
def test_a_run_directory_with_a_damaged_file_is_refused_in_one_line(
    tmp_path: Path, damage: str
) -> None:
    """A report is read by a program branching on its exit status: no traceback, exit 2."""
    directory = tmp_path / "run"
    _run(directory, 1)
    if damage == "pool-tree":
        [tree] = sorted((directory / POOL_DIRNAME).glob("*.json"))
        tree.write_text("{not json", encoding="utf-8")
    else:
        [policy] = sorted((directory / CYCLES_DIRNAME).glob("*/policy.py"))
        policy.write_bytes(b"\xff\xfe not utf-8\n")

    reported = cli("--report", str(directory))

    assert reported.returncode == EXIT_REFUSED
    assert "Traceback" not in reported.stderr
    assert reported.stderr.strip()


@pytest.mark.parametrize("record", ["", '{"cycles": ', '{"cycles": "three"}', "[]"])
def test_a_session_record_that_cannot_be_read_is_refused_rather_than_taken_for_finished(
    tmp_path: Path, record: str
) -> None:
    """A half-written or damaged request must not make a run that stopped short look done."""
    directory = tmp_path / "run"
    _run(directory, 1)
    (directory / SESSION_FILENAME).write_text(record, encoding="utf-8")

    reported = cli("--report", str(directory))

    assert reported.returncode == EXIT_REFUSED
    assert SESSION_FILENAME in reported.stderr
    assert "Traceback" not in reported.stderr
