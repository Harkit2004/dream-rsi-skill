"""The worked example: a real scorer, driven by a coding-agent CLI (issue #76).

The quickstart proves the orchestration layer with scripted roles on both sides.
Nothing showed the loop on a task where a model writes the candidates and a scorer
measures something real, which is what a user of the skill wants to see — and the
fastest way to find out where the adapters' contracts are wrong. ``examples/packing``
is that task: ten points in the unit square, scored by their smallest pairwise
distance.

The scorer and the task file are the real ones. Only the CLI is a stand-in
(``tests/stand_in_cli.py`` and ``tests/stand_in_developer.py``), so CI never calls a
model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dream_rsi import run
from dream_rsi.adapters.command_evaluator import CommandEvaluator
from dream_rsi.adapters.evaluator import ScoreDirection
from dream_rsi.orchestrator import TREE_FILENAME
from dream_rsi.run import CYCLE_TEMPLATE, CYCLES_DIRNAME
from dream_rsi.tree import DiscoveryTree

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "packing"
SCORER = EXAMPLE / "score.py"
TASK = EXAMPLE / "task.py"
TESTS = Path(__file__).parent

HIGHER = ScoreDirection.HIGHER_IS_BETTER


def _program(points: object) -> str:
    """A candidate: a program that prints ``points`` as JSON."""
    return f"import json\nprint(json.dumps({points!r}))\n"


# Three answers of known quality. The grid is what a first attempt writes; the
# staggered rows (3, 2, 3, 2) are a construction a model can find; a pile of ten
# copies of one point is valid and as bad as it gets.
GRID = [[(i % 4) / 3, (i // 4) / 2] for i in range(10)]
STAGGERED = [
    [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
    [0.25, 1 / 3], [0.75, 1 / 3],
    [0.0, 2 / 3], [0.5, 2 / 3], [1.0, 2 / 3],
    [0.25, 1.0], [0.75, 1.0],
]  # fmt: skip
PILE = [[0.5, 0.5]] * 10


def _evaluator() -> CommandEvaluator:
    # A short limit on the candidate, so the test that hangs does not wait out the
    # real thirty seconds; the scorer's own timeout is what is being exercised.
    return CommandEvaluator([sys.executable, str(SCORER), "--timeout", "3"], HIGHER, timeout=60.0)


def _score(tmp_path: Path, source: str):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return _evaluator().evaluate(source, workspace)


def test_the_scorer_ranks_known_answers_in_the_stated_direction(tmp_path: Path) -> None:
    """Issue #76's first "tests first": larger is better, and the numbers are the true ones."""
    pile = _score(tmp_path, _program(PILE))
    grid = _score(tmp_path, _program(GRID))
    staggered = _score(tmp_path, _program(STAGGERED))

    assert pile.evaluated and grid.evaluated and staggered.evaluated
    assert pile.score == pytest.approx(0.0)
    assert grid.score == pytest.approx(1 / 3)
    assert staggered.score == pytest.approx(5 / 12)  # (0.25, 1/3) to (0, 0) is 5/12
    assert pile.score < grid.score < staggered.score
    assert staggered.correct


@pytest.mark.parametrize(
    ("source", "named"),
    [
        pytest.param(_program(GRID[:9]), "10 points", id="too-few-points"),
        pytest.param(_program([[*p] for p in GRID] + [[0.5, 0.5]]), "10 points", id="too-many"),
        pytest.param(_program([[1.5, 0.5]] + GRID[1:]), "unit square", id="outside-the-square"),
        pytest.param("print('not json')\n", "JSON", id="not-json"),
        pytest.param("raise SystemExit(3)\n", "exited with status 3", id="crashes"),
        pytest.param("import time\ntime.sleep(120)\n", "timed out", id="hangs"),
        # Stopped when the output passes the limit, not held in memory until the timeout.
        pytest.param("while True:\n    print('x' * 65536)\n", "more than", id="floods-stdout"),
        pytest.param(
            "import sys\nwhile True:\n    sys.stderr.write('x' * 65536)\n",
            "more than",
            id="floods-stderr",
        ),
        pytest.param(_program([[0.5, "x"]] * 10), "numbers", id="not-numbers"),
    ],
)
def test_an_invalid_answer_is_a_failed_evaluation_that_says_why(
    tmp_path: Path, source: str, named: str
) -> None:
    result = _score(tmp_path, source)

    assert not result.evaluated
    assert result.score is None
    assert result.error is not None and named in result.error


def test_nothing_a_candidate_started_outlives_its_scoring(tmp_path: Path) -> None:
    """A candidate may start processes of its own; scoring it ends them too.

    Run as the user would run the scorer, outside ``CommandEvaluator`` (whose own
    process-group kill would otherwise hide the difference). The child marks that it
    is running before the candidate answers, so the test cannot pass by the child
    never having started.
    """
    started, survived = tmp_path / "started", tmp_path / "survived"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child = (
        f"import pathlib, time; pathlib.Path({str(started)!r}).touch(); "
        f"time.sleep(2); pathlib.Path({str(survived)!r}).touch()"
    )
    (workspace / "solution.py").write_text(
        "import pathlib, subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        f"while not pathlib.Path({str(started)!r}).exists():\n"
        "    time.sleep(0.01)\n" + _program(STAGGERED),
        encoding="utf-8",
    )

    scored = subprocess.run(
        [sys.executable, str(SCORER)], cwd=workspace, capture_output=True, text=True, check=False
    )

    assert scored.returncode == 0, scored.stderr
    assert (workspace / "eval" / "score.json").is_file()
    assert started.exists()
    deadline = time.monotonic() + 4  # well past the child's two seconds
    while time.monotonic() < deadline and not survived.exists():
        time.sleep(0.05)
    assert not survived.exists(), "a process the candidate started was still running"


def _environment(monkeypatch: pytest.MonkeyPatch, *, agent: str = "packing") -> None:
    """The stand-in CLIs, named the way a user names a real one."""
    revision = TESTS / "packing_revision.py"
    monkeypatch.setenv(
        "DREAM_RSI_AGENT_CMD", f"{sys.executable} {TESTS / 'stand_in_cli.py'} --mode {agent}"
    )
    monkeypatch.setenv(
        "DREAM_RSI_DEVELOPER_CMD",
        f"{sys.executable} {TESTS / 'stand_in_developer.py'} --revision {revision}",
    )


def test_the_task_file_runs_a_cycle_over_the_stand_in_cli_and_records_a_tree_the_real_scorer_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #76's second "tests first": ``task.py`` end to end, minus only the model."""
    _environment(monkeypatch)
    directory = tmp_path / "run"

    code = run.main(
        ["--task", str(TASK), "--cycles", "1", "--workers", "2", "--rounds", "2", str(directory)]
    )

    assert code == 0
    cycle = directory / CYCLES_DIRNAME / CYCLE_TEMPLATE.format(0)
    tree = DiscoveryTree.load(cycle / TREE_FILENAME)
    attempts = [node for node in tree.iter_nodes() if node.parent_id is not None]
    assert attempts
    scored = [node for node in attempts if node.score is not None]
    assert scored, "no attempt was scored: the scorer, the agent's program, or the wiring is wrong"
    for node in scored:
        # Ten random points: valid, and scored by the real scorer, not the toy landscape.
        assert 0.0 <= node.score < 0.5
        assert node.diagnostics["direction"] == "higher_is_better"
        assert node.diagnostics["correct"] is True
    record = json.loads((cycle / run.RECORD_FILENAME).read_text(encoding="utf-8"))
    assert record["selection"]["winner"]


def test_without_a_cli_the_example_refuses_to_start_and_names_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """There is no default CLI: a default would be a provider chosen for the user."""
    monkeypatch.delenv("DREAM_RSI_AGENT_CMD", raising=False)
    monkeypatch.delenv("DREAM_RSI_DEVELOPER_CMD", raising=False)

    code = run.main(["--task", str(TASK), "--cycles", "1", str(tmp_path / "run")])

    err = capsys.readouterr().err
    assert code == run.EXIT_REFUSED
    assert "DREAM_RSI_AGENT_CMD" in err
    assert not (tmp_path / "run").exists()


def test_the_example_is_a_directory_a_reader_can_use_on_its_own() -> None:
    """A problem statement, a scorer and a task file, and the scorer runs by itself."""
    assert (EXAMPLE / "problem.md").read_text(encoding="utf-8").strip()
    completed = subprocess.run(
        [sys.executable, str(SCORER), "--help"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
