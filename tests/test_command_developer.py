"""A policy-development agent that drives a coding-agent CLI (issue #71).

Until now the only developer was ``FakeDeveloper``, which answers from a fixed list,
so every policy improvement a run ever reported was written in advance. Here the
revision comes from a CLI's file instead, and what is tested is everything around
that: what the CLI is given, what happens to what it writes, and — above all — that
a CLI which fails costs the cycle nothing but that revision.

The CLI is ``tests/stand_in_developer.py``. Nothing calls a model.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from dream_rsi.adapters.command_developer import CommandDeveloper
from dream_rsi.adapters.toy_search import ToySearchAgent, ToySearchEvaluator
from dream_rsi.develop import DevelopmentReport, develop
from dream_rsi.dream import DreamConfig, ReplayWorld, select
from dream_rsi.orchestrator import RolloutConfig
from dream_rsi.replay import ReplaySimulator
from dream_rsi.run import (
    DEFAULT_POLICY_SOURCE,
    TOY_PROBLEM,
    TOY_SCRIPT,
    RunConfig,
    run_cycles,
)
from dream_rsi.sandbox import SandboxLimits
from dream_rsi.tree import DiscoveryTree

STAND_IN = str(Path(__file__).with_name("stand_in_developer.py"))
TREES = Path(__file__).parent / "fixtures" / "trees"

# Generous next to what a policy here spends, so nothing is timing the machine.
TEST_LIMITS = SandboxLimits(
    wall_seconds=5.0, cpu_seconds=2, memory_bytes=512 * 1024 * 1024, disk_bytes=1024 * 1024
)

BASELINE = """\
from dream_rsi.policy import GreedyBestFirstPolicy


class OptimalPolicy(GreedyBestFirstPolicy):
    pass
"""

REVISION = """\
from dream_rsi.policy import GreedyBestFirstPolicy


class OptimalPolicy(GreedyBestFirstPolicy):
    def __init__(self, config=None):
        super().__init__({'beta': 0.5})
"""


@pytest.fixture
def log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "stand-in-log"
    monkeypatch.setenv("STAND_IN_LOG", str(directory))
    return directory


def _calls(log: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(log.glob("call_*"))]


def _developer(*flags: str, timeout: float = 60.0, **options: Any) -> CommandDeveloper:
    return CommandDeveloper([sys.executable, STAND_IN, *flags], timeout=timeout, **options)


def _revising(tmp_path: Path, text: str = REVISION) -> CommandDeveloper:
    revision = tmp_path / "revision.py"
    revision.write_text(text, encoding="utf-8")
    return _developer("--revision", str(revision))


def _develop(
    developer: CommandDeveloper, scratch: Path, *, versions: int = 2, attempts: int = 1
) -> DevelopmentReport:
    world = ReplayWorld(
        name="wide_shallow",
        simulator=ReplaySimulator(DiscoveryTree.load(TREES / "wide_shallow" / "tree.json")),
    )
    return develop(
        BASELINE,
        [world],
        developer,
        versions=versions,
        attempts=attempts,
        config=DreamConfig(width=2),
        limits=TEST_LIMITS,
        scratch_root=scratch,
    )


def test_a_revision_the_command_writes_is_replayed_scored_and_differs_from_the_incumbent(
    tmp_path: Path, log: Path
) -> None:
    """Issue #71's first "tests first": once a model writes this, the dreaming half is the paper's."""
    report = _develop(_revising(tmp_path), tmp_path)

    assert not report.rejected
    incumbent, revised = report.versions
    assert revised.source == REVISION
    assert revised.source != incumbent.source
    assert revised.report.score is not None
    assert [replay.world for replay in revised.report.replays] == ["wide_shallow"]

    # What the command was given: the rendered prompt and the policy being revised,
    # in its own scratch directory.
    [call] = _calls(log)
    assert call["policy"] == BASELINE
    assert "Version `v0` scored" in call["prompt"]
    assert "$version" not in call["prompt"]
    assert "revised_policy.py" in call["prompt"]
    assert "prompt.md" in call["instruction"]


@pytest.mark.parametrize("via", ["argument", "stdin"])
def test_the_adapter_adds_nothing_to_the_command_line_but_its_instruction(
    tmp_path: Path, log: Path, via: str
) -> None:
    flags = ["--stdin"] if via == "stdin" else []
    revision = tmp_path / "revision.py"
    revision.write_text(REVISION, encoding="utf-8")
    developer = _developer(*flags, "--revision", str(revision), prompt_via=via)

    _develop(developer, tmp_path)

    [call] = _calls(log)
    given = [*flags, "--revision", str(revision)]
    assert call["argv"] == ([*given, call["instruction"]] if via == "argument" else given)


def test_output_that_is_not_a_policy_is_refused_through_the_existing_path_and_asked_for_again(
    tmp_path: Path, log: Path
) -> None:
    """Issue #71's second: validation stays in ``develop``, and the retry sees the refusal."""
    report = _develop(_developer("--mode", "prose"), tmp_path, attempts=3)

    assert len(report.versions) == 1, "prose was scored as a policy"
    assert len(report.rejected) == 3
    assert all("does not parse" in refusal.reason for refusal in report.rejected)
    assert report.calls == 3
    first, second, third = _calls(log)
    assert "does not parse" not in first["prompt"]
    assert "does not parse" in second["prompt"]
    assert "does not parse" in third["prompt"]


@pytest.mark.parametrize(
    ("flags", "timeout", "named"),
    [
        pytest.param(["--mode", "fail"], 60.0, "boom: the stand-in", id="exits-non-zero"),
        pytest.param(["--mode", "hang"], 1.0, "timed out", id="hangs"),
        # A good module does not redeem a run that ended in failure.
        pytest.param(["--mode", "write-then-fail"], 60.0, "status 4", id="crashes-at-the-end"),
        pytest.param(["--mode", "no-file"], 60.0, "wrote no revised_policy.py", id="no-file"),
        pytest.param(["--mode", "empty"], 60.0, "wrote no revised_policy.py", id="empty-file"),
    ],
)
def test_a_command_that_fails_costs_the_revision_and_not_the_cycle(
    tmp_path: Path, log: Path, flags: list[str], timeout: float, named: str
) -> None:
    """Issue #71's third: rejected like any other unusable output, incumbent still selectable.

    The reason carries what the CLI printed, because that is what the next attempt
    reads and what a person reading the round needs.
    """
    revision = tmp_path / "revision.py"
    revision.write_text(REVISION, encoding="utf-8")
    developer = _developer(*flags, "--revision", str(revision), timeout=timeout)

    report = _develop(developer, tmp_path, attempts=2)

    assert [version.name for version in report.versions] == ["v0"]
    assert len(report.rejected) == 2
    assert all(named in refusal.reason for refusal in report.rejected)
    assert report.calls == 2
    assert select(report.comparison).winner == "v0"
    # The retry is told why the first try was refused.
    assert named in _calls(log)[1]["prompt"]


def test_a_run_whose_developer_always_fails_still_completes_every_cycle(
    tmp_path: Path, log: Path
) -> None:
    run = run_cycles(
        agent=ToySearchAgent(script=TOY_SCRIPT),
        evaluator=ToySearchEvaluator(),
        developer=_developer("--mode", "fail"),
        policy=DEFAULT_POLICY_SOURCE,
        problem=TOY_PROBLEM,
        directory=tmp_path / "run",
        config=RunConfig(
            cycles=2, rollout=RolloutConfig(workers=2, max_rounds=2), versions=2, attempts=1
        ),
    )

    assert len(run.cycles) == 2
    for record in run.cycles:
        assert record.selection.winner == "v0"
        assert record.rejected == 1
        assert record.cost.dreaming.developer_calls == 1


def test_a_revision_is_never_run_by_the_adapter(tmp_path: Path, log: Path) -> None:
    """The revised source only ever runs through the sandbox, as before.

    The revision writes a file the moment its module body runs. An adapter that
    imported it — to check it, to load it — would leave the file behind.
    """
    escaped = tmp_path / "escaped.txt"
    source = (
        "import pathlib\n"
        f"pathlib.Path({str(escaped)!r}).write_text('ran in the harness')\n"
        "\n" + REVISION
    )

    _develop(_revising(tmp_path, source), tmp_path)

    assert not escaped.exists()


def test_an_answer_that_is_a_symlink_to_somewhere_else_is_not_read_through(
    tmp_path: Path, log: Path
) -> None:
    """The CLI has a shell in the scratch directory, and the harness reads what it left.

    Following a link would pull any file the harness can read into the recorded policy
    source — and into the next prompt — so it is a refusal like any other unusable answer.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text(REVISION, encoding="utf-8")  # legal source, so only the link is wrong
    developer = _developer("--mode", "symlink", "--revision", str(secret))

    report = _develop(developer, tmp_path, attempts=1)

    assert [version.name for version in report.versions] == ["v0"]
    assert "outside" in report.rejected[0].reason


def test_an_answer_swapped_for_a_link_after_the_check_is_still_not_read_through(
    tmp_path: Path, log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checking a path and reading it are two steps, and the CLI may have left a process running.

    Something detached from the command's process group survives cleanup and can put a
    link where the file was in between. So the read itself must refuse a link, not
    rely on the check having come first.
    """
    monkeypatch.setattr(
        "dream_rsi.adapters.command_developer.is_inside", lambda root, path: True
    )
    secret = tmp_path / "secret.txt"
    secret.write_text(REVISION, encoding="utf-8")  # legal source, so only the link is wrong
    developer = _developer("--mode", "symlink", "--revision", str(secret))

    report = _develop(developer, tmp_path, attempts=1)

    assert [version.name for version in report.versions] == ["v0"]
    assert "could not read" in report.rejected[0].reason


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root reads any file")
def test_an_answer_the_harness_cannot_read_costs_the_revision_and_not_the_cycle(
    tmp_path: Path, log: Path
) -> None:
    """A file with no read permission exists, so it is not "no file" — but it is no answer."""
    revision = tmp_path / "revision.py"
    revision.write_text(REVISION, encoding="utf-8")
    developer = _developer("--mode", "unreadable", "--revision", str(revision))

    report = _develop(developer, tmp_path, attempts=1)

    assert [version.name for version in report.versions] == ["v0"]
    assert "could not read" in report.rejected[0].reason
    assert select(report.comparison).winner == "v0"


def test_a_revision_over_the_output_limit_is_refused_and_one_at_it_is_not(
    tmp_path: Path, log: Path
) -> None:
    """``max_output_bytes`` bounds what the command may write as well as what it prints.

    The file is read into the harness's memory, and the CLI is not the harness's to trust
    with how large it is.
    """
    revision = tmp_path / "revision.py"
    revision.write_text(REVISION, encoding="utf-8")
    size = revision.stat().st_size

    at_limit = _develop(_developer("--revision", str(revision), max_output_bytes=size), tmp_path)
    over = _develop(
        _developer("--revision", str(revision), max_output_bytes=size - 1),
        tmp_path,
        attempts=1,
    )

    assert not at_limit.rejected
    assert at_limit.versions[1].source == REVISION
    assert [version.name for version in over.versions] == ["v0"]
    assert f"larger than {size - 1} bytes" in over.rejected[0].reason


@contextlib.contextmanager
def _within(seconds: int) -> Iterator[None]:
    """Fail, instead of hanging the suite, if the block is still running after ``seconds``."""

    def _expired(signum: int, frame: object) -> None:
        raise TimeoutError(f"still running after {seconds}s")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize(
    ("mode", "named"),
    [
        pytest.param("fifo", "not a regular file", id="pipe"),
        pytest.param("fifo-held", "not a regular file", id="pipe-held-open"),
        pytest.param("directory", "not a regular file", id="directory"),
    ],
)
def test_an_answer_that_is_not_a_file_is_refused_and_never_waited_on(
    tmp_path: Path, log: Path, mode: str, named: str
) -> None:
    """Opening a pipe for reading waits for a writer; the harness must not."""
    developer = _developer("--mode", mode)

    with _within(60):
        report = _develop(developer, tmp_path, attempts=1)

    assert [version.name for version in report.versions] == ["v0"]
    assert named in report.rejected[0].reason


def test_a_command_is_an_argv_list_because_there_is_no_shell_to_split_a_string() -> None:
    with pytest.raises(TypeError, match="argv list"):
        CommandDeveloper("claude -p")  # type: ignore[arg-type]
