"""Running the loop on your own task from a shell (issue #68).

An agent inside a harness has a shell, not a Python session, and ``run.main``
hardwired the toy task, so the toy was the only thing it could run — which is what
happened when the skill was tried from OpenCode on a task it fits: the run scored
the toy's ``(width, depth)`` planner and never touched the task. A task is one
Python file defining ``task()``, so that each of the three roles can be a user's
own class without a config format to translate it through.

Nothing here calls a model: the roles a task file supplies are scripted.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from dream_rsi import run
from dream_rsi.orchestrator import TREE_FILENAME
from dream_rsi.run import CYCLE_TEMPLATE, CYCLES_DIRNAME, POLICY_FILENAME
from dream_rsi.tree import DiscoveryTree

PROBLEM = "write the candidate that scores highest under the user's own scorer"

# A policy that is not the default baseline, so a run that ignored the task's
# ``policy`` would deploy something visibly different.
POLICY = """\
from dream_rsi.policy import GreedyBestFirstPolicy


class OptimalPolicy(GreedyBestFirstPolicy):
    pass
"""

# Every score this evaluator gives is above 1000, and no score on the toy landscape
# reaches 32, so a node carrying one of them was scored by *this* evaluator.
BASE_SCORE = 1000.0


def _task_file(
    directory: Path, *, problem: str = PROBLEM, policy: str = POLICY, name: str = "task.py"
) -> Path:
    """A task file with its own agent, evaluator and policy, and the toy's none of them."""
    path = directory / name
    path.write_text(
        textwrap.dedent(
            f"""
            from dream_rsi.adapters.agent import Artifact
            from dream_rsi.adapters.evaluator import EvalResult, ScoreDirection
            from dream_rsi.adapters.fake_developer import FakeDeveloper
            from dream_rsi.task import Task


            class Agent:
                def propose(self, context):
                    return Artifact(content="candidate " + "x" * (context.seed % 5) + "\\n")


            class Evaluator:
                direction = ScoreDirection.HIGHER_IS_BETTER
                baseline_score = None

                def evaluate(self, artifact, workspace):
                    return EvalResult(score={BASE_SCORE} + len(artifact), correct=True)


            def task():
                return Task(
                    agent=Agent(),
                    evaluator=Evaluator(),
                    developer=FakeDeveloper(),
                    problem={problem!r},
                    policy={policy!r},
                )
            """
        ),
        encoding="utf-8",
    )
    return path


def _main(tmp_path: Path, *arguments: str) -> int:
    return run.main(["--cycles", "1", "--rounds", "2", *arguments, str(tmp_path / "run")])


def test_a_task_file_supplies_the_roles_the_run_uses(tmp_path: Path) -> None:
    """Issue #68's first "tests first": the nodes carry *that* evaluator's scores."""
    task = _task_file(tmp_path)

    code = _main(tmp_path, "--task", str(task))

    assert code == 0
    cycle = tmp_path / "run" / CYCLES_DIRNAME / CYCLE_TEMPLATE.format(0)
    tree = DiscoveryTree.load(cycle / TREE_FILENAME)
    scores = [node.score for node in tree.iter_nodes() if node.parent_id is not None]
    assert scores, "the run recorded no attempts"
    assert all(score is not None and score > BASE_SCORE for score in scores)
    assert all(node.artifact.startswith("candidate") for node in tree.iter_nodes() if node.artifact)
    # The task's own starting policy is what cycle 0 deployed, not the default's.
    assert (cycle / POLICY_FILENAME).read_text(encoding="utf-8") == POLICY


@pytest.mark.parametrize(
    ("source", "named"),
    [
        pytest.param("x = 1\n", "task()", id="no-task-function"),
        pytest.param("def task():\n    return 'not a task'\n", "not a dream_rsi.task.Task", id="wrong-type"),
        pytest.param("def task():\n    raise RuntimeError('boom')\n", "boom", id="task-raises"),
        pytest.param("def task(:\n", "SyntaxError", id="not-python"),
        pytest.param("import no_such_module_anywhere\n", "no_such_module_anywhere", id="bad-import"),
        pytest.param(
            "from dream_rsi.adapters.toy_search import ToySearchEvaluator\n"
            "from dream_rsi.adapters.fake_developer import FakeDeveloper\n"
            "from dream_rsi.task import Task\n"
            "def task():\n"
            "    return Task(agent=object(), evaluator=ToySearchEvaluator(),\n"
            "                developer=FakeDeveloper(), problem='p')\n",
            "agent",
            id="agent-cannot-propose",
        ),
    ],
)
def test_a_task_file_that_cannot_supply_a_task_stops_the_command_before_any_cycle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source: str,
    named: str,
) -> None:
    """Issue #68's third "tests first": non-zero, the file and the reason named, nothing run."""
    path = tmp_path / "broken_task.py"
    path.write_text(source, encoding="utf-8")

    code = _main(tmp_path, "--task", str(path))

    err = capsys.readouterr().err
    assert code == run.EXIT_REFUSED
    assert str(path) in err
    assert named in err
    assert "Traceback" not in err
    assert not (tmp_path / "run").exists(), "a cycle started, or the run directory was made"


def test_a_missing_task_file_is_named_and_stops_the_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nowhere" / "task.py"

    code = _main(tmp_path, "--task", str(missing))

    err = capsys.readouterr().err
    assert code == run.EXIT_REFUSED
    assert str(missing) in err
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("change", "named"),
    [
        pytest.param({"problem": "maximise a different thing"}, "problem", id="problem"),
        pytest.param({"policy": POLICY + "# a different start\n"}, "policy", id="policy"),
    ],
)
def test_a_run_directory_is_not_resumed_under_a_different_task(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, str], named: str
) -> None:
    """Issue #68's fourth: a history is one experiment or it is not (issue #45).

    The ``problem`` text is what the manifest already records, and it is what
    tells one task from another, so a second task file under the directory the
    first one ran in is refused, naming the field — and appends nothing.
    """
    first = _task_file(tmp_path, name="first.py")
    second = _task_file(tmp_path, name="second.py", **change)
    assert _main(tmp_path, "--task", str(first)) == 0
    capsys.readouterr()
    records = sorted((tmp_path / "run" / CYCLES_DIRNAME).rglob("cycle.json"))
    before = [json.loads(path.read_text(encoding="utf-8")) for path in records]

    code = run.main(["--cycles", "2", "--rounds", "2", "--task", str(second), str(tmp_path / "run")])

    err = capsys.readouterr().err
    assert code == run.EXIT_REFUSED
    assert named in err
    assert "Traceback" not in err
    after = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "run" / CYCLES_DIRNAME).rglob("cycle.json"))
    ]
    assert after == before
