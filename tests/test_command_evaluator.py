"""Scoring a candidate by running the user's own command (issue #69).

Every scorer here is a tiny script written into ``tmp_path`` and run as a real
subprocess, because what is being tested is the contract between a command and the
loop: what it may write, how it may fail, and that none of that can take a rollout
down. Nothing calls a model.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from dream_rsi.adapters.command_evaluator import (
    EVAL_ERROR,
    MALFORMED_SCORE,
    NO_SCORE,
    SCORE_PATH,
    TIMEOUT,
    CommandEvaluator,
)
from dream_rsi.adapters.evaluator import ScoreDirection, TaskEvaluator
from dream_rsi.adapters.fake_agent import FakeAgent
from dream_rsi.orchestrator import FirstEligiblePolicy, RolloutConfig, run_rollout
from dream_rsi.workspace import SnapshotStore

HIGHER = ScoreDirection.HIGHER_IS_BETTER
LOWER = ScoreDirection.LOWER_IS_BETTER

# Generous next to a script that prints and exits, so nothing here times the machine.
QUICK = 60.0


def _scorer(tmp_path: Path, body: str) -> list[str]:
    """An argv running a script whose body is ``body``, with the usual imports in scope."""
    script = tmp_path / "scorer.py"
    script.write_text(
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n" + textwrap.dedent(body),
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


# Scores a candidate by counting its lines, so the score proves the artifact was
# written where the command runs, under the name it was told.
LINE_COUNT = """
    lines = Path("solution.py").read_text().splitlines()
    Path("eval").mkdir(exist_ok=True)
    Path("eval/score.json").write_text(json.dumps({"score": len(lines) / 4, "correct": True}))
    print("scored", len(lines), "lines")
"""


@pytest.mark.parametrize(
    ("direction", "canonical"),
    [
        pytest.param(HIGHER, 1.0, id="higher-is-better"),
        pytest.param(LOWER, -1.0, id="lower-is-better"),
    ],
)
def test_a_scorer_that_writes_a_score_produces_that_score(
    tmp_path: Path, direction: ScoreDirection, canonical: float
) -> None:
    """Under LOWER_IS_BETTER the node's ``s_v`` is the canonical mapping, not the raw number."""
    evaluator = CommandEvaluator(_scorer(tmp_path, LINE_COUNT), direction, timeout=QUICK)

    result = evaluator.evaluate("a\nb\nc\nd\n", _workspace(tmp_path))

    assert isinstance(evaluator, TaskEvaluator)
    assert result.evaluated
    assert result.correct
    assert result.score == 1.0
    assert result.to_node_fields(direction)["score"] == canonical
    # What the command printed is in front of the agent that resumes from this node.
    assert "scored 4 lines" in result.diagnostics


@pytest.mark.parametrize(
    ("body", "error", "diagnostic", "fail_class"),
    [
        pytest.param(
            """
            print("the benchmark blew up", file=sys.stderr)
            sys.exit(3)
            """,
            "status 3",
            "the benchmark blew up",
            EVAL_ERROR,
            id="exits-non-zero",
        ),
        pytest.param(
            """
            Path("eval").mkdir(exist_ok=True)
            Path("eval/error.txt").write_text("compile failed on line 3")
            """,
            "compile failed on line 3",
            "compile failed on line 3",
            EVAL_ERROR,
            id="writes-error-txt",
        ),
        pytest.param(
            """
            Path("eval/score.json").write_text('{"score": 9.0, "correct": true}')
            sys.exit(2)
            """,
            "status 2",
            "status 2",
            EVAL_ERROR,
            id="exits-non-zero-after-writing-a-score",
        ),
        pytest.param(
            "pass", "no eval/score.json", "no eval/score.json", NO_SCORE, id="writes-nothing"
        ),
    ],
)
def test_a_scorer_that_fails_gives_a_failed_result_carrying_the_reason(
    tmp_path: Path, body: str, error: str, diagnostic: str, fail_class: str
) -> None:
    evaluator = CommandEvaluator(_scorer(tmp_path, body), HIGHER, timeout=QUICK)

    result = evaluator.evaluate("candidate\n", _workspace(tmp_path))

    assert not result.evaluated
    assert result.score is None
    assert not result.correct
    assert result.fail_class == fail_class
    assert result.error is not None and error in result.error
    assert diagnostic in result.diagnostics


def test_a_result_left_by_an_earlier_attempt_is_not_read_as_this_ones(tmp_path: Path) -> None:
    """A workspace is resumed from its parent's snapshot, which holds the parent's score.

    A scorer that writes nothing this time must not be credited with what its parent
    scored, or a candidate that broke the scorer would inherit a good result.
    """
    workspace = _workspace(tmp_path)
    (workspace / SCORE_PATH).parent.mkdir()
    (workspace / SCORE_PATH).write_text('{"score": 99.0, "correct": true}')
    evaluator = CommandEvaluator(_scorer(tmp_path, "pass"), HIGHER, timeout=QUICK)

    result = evaluator.evaluate("candidate\n", workspace)

    assert not result.evaluated
    assert result.score is None
    assert result.fail_class == NO_SCORE


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("time.sleep(120)", id="sleeps"),
        pytest.param(
            # A child that keeps the command's output open for as long as it lives:
            # a scorer that waited on a pipe would still be waiting when the timeout
            # had long passed.
            'child = subprocess.Popen(["sleep", "120"])\n'
            'Path("child.pid").write_text(str(child.pid))\n'
            "time.sleep(120)",
            id="leaves-a-child-behind",
        ),
    ],
)
def test_a_scorer_that_hangs_is_stopped_at_the_timeout_and_reported_as_such(
    tmp_path: Path, body: str
) -> None:
    evaluator = CommandEvaluator(_scorer(tmp_path, body), HIGHER, timeout=1.0)
    workspace = _workspace(tmp_path)
    started = time.monotonic()

    result = evaluator.evaluate("candidate\n", workspace)

    assert time.monotonic() - started < 30
    assert not result.evaluated
    assert result.score is None
    assert result.fail_class == TIMEOUT
    assert result.error is not None and "timed out" in result.error
    # Killed with everything it started: a benchmark that outlived its own timeout
    # would go on burning the machine a search is meant to be spending carefully.
    pidfile = workspace / "child.pid"
    if pidfile.exists():
        assert _is_gone(int(pidfile.read_text()))


def _is_gone(pid: int, *, patience: float = 10.0) -> bool:
    """Whether ``pid`` stops existing, allowing for it to be reaped after the kill."""
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize(
    "written",
    [
        pytest.param("not json at all", id="not-json"),
        pytest.param('{"score": NaN, "correct": true}', id="nan-score"),
        pytest.param('{"score": Infinity, "correct": true}', id="infinite-score"),
        pytest.param('{"correct": true}', id="no-score"),
        pytest.param('{"score": 1.0}', id="no-correct"),
        pytest.param('{"score": "1.0", "correct": true}', id="string-score"),
        pytest.param('{"score": true, "correct": true}', id="boolean-score"),
        pytest.param("[1, 2]", id="not-an-object"),
    ],
)
def test_a_malformed_score_file_gives_a_failed_result_not_a_crash(
    tmp_path: Path, written: str
) -> None:
    """The guarantee issue #34 put on the tree: nothing a scorer writes reaches it as a bad number."""
    body = f'Path("eval/score.json").write_text({written!r})'
    evaluator = CommandEvaluator(_scorer(tmp_path, body), HIGHER, timeout=QUICK)

    result = evaluator.evaluate("candidate\n", _workspace(tmp_path))

    assert not result.evaluated
    assert result.score is None
    assert result.fail_class == MALFORMED_SCORE


def test_a_scorer_may_name_its_own_failure_class_and_explain_it(tmp_path: Path) -> None:
    body = """
        Path("eval/score.json").write_text(json.dumps(
            {"score": 0.5, "correct": False, "fail_class": "wrong_answer",
             "diagnostics": "expected 3, got 4"}))
        print("checked 12 cases")
    """
    evaluator = CommandEvaluator(_scorer(tmp_path, body), HIGHER, timeout=QUICK)

    result = evaluator.evaluate("candidate\n", _workspace(tmp_path))

    assert result.score == 0.5
    assert not result.correct
    assert result.fail_class == "wrong_answer"
    assert "expected 3, got 4" in result.diagnostics
    assert "checked 12 cases" in result.diagnostics


def test_a_command_that_cannot_be_started_is_a_failed_result_naming_it(tmp_path: Path) -> None:
    evaluator = CommandEvaluator(["no-such-scorer-binary-anywhere"], HIGHER, timeout=QUICK)

    result = evaluator.evaluate("candidate\n", _workspace(tmp_path))

    assert not result.evaluated
    assert result.error is not None and "no-such-scorer-binary-anywhere" in result.error


def test_a_command_is_an_argv_list_because_there_is_no_shell_to_split_a_string() -> None:
    with pytest.raises(TypeError, match="argv list"):
        CommandEvaluator("python score.py", HIGHER)  # type: ignore[arg-type]


def test_a_candidate_cannot_be_written_outside_the_workspace() -> None:
    with pytest.raises(ValueError, match="artifact_name"):
        CommandEvaluator([sys.executable], HIGHER, artifact_name="../escape.py")


def test_a_rollout_scores_every_node_with_the_command(tmp_path: Path) -> None:
    """The evaluator in the loop it exists for: real workspaces, resumed from snapshots.

    Each node's score is what the scorer computed from *that node's* artifact, so it
    is also the check that the candidate is written into the attempt's own workspace
    and not a shared one.
    """
    evaluator = CommandEvaluator(_scorer(tmp_path, LINE_COUNT), HIGHER, timeout=QUICK)
    workspace = _workspace(tmp_path)
    agent = FakeAgent(script=("a\n", "a\nb\n", "a\nb\nc\n", "a\nb\nc\nd\n"))

    rollout = run_rollout(
        agent=agent,
        evaluator=evaluator,
        policy=FirstEligiblePolicy(),
        problem="count lines",
        workspace=workspace,
        snapshots=SnapshotStore(tmp_path / "store"),
        config=RolloutConfig(workers=2, max_rounds=2),
    )

    attempts = [node for node in rollout.tree.iter_nodes() if node.parent_id is not None]
    assert attempts
    for node in attempts:
        assert node.artifact is not None
        assert node.score == len(node.artifact.splitlines()) / 4
