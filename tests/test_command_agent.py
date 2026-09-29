"""A discovery agent that drives a coding-agent CLI (issue #70).

The CLI here is ``tests/stand_in_cli.py``: a script that reads the prompt, records
what it could see, and writes what a discovery agent would. CI never calls a real
model, and what is being tested is the adapter's contract with a CLI — what the
command is given, what it must leave behind, and what happens when it does not.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from dream_rsi.adapters.agent import AgentContext, Artifact
from dream_rsi.adapters.command_agent import CommandAgent, CommandAgentError
from dream_rsi.adapters.evaluator import EvalResult, ScoreDirection
from dream_rsi.adapters.fake_developer import FakeDeveloper
from dream_rsi.adapters.toy_search import ToySearchEvaluator
from dream_rsi.orchestrator import RolloutConfig, run_rollout
from dream_rsi.run import (
    DEFAULT_POLICY_SOURCE,
    TOY_PROBLEM,
    TOY_REVISIONS,
    RunConfig,
    run_cycles,
)
from dream_rsi.tree import Node
from dream_rsi.workspace import SnapshotStore

STAND_IN = str(Path(__file__).with_name("stand_in_cli.py"))

PROBLEM = "find the shortest program that prints its own length"

# Quick enough that nothing here times the machine, long enough that a stand-in
# that has started is never cut off.
TIMEOUT = 60.0

# A template whose whole body is the paths, so what the stand-in learns is exactly
# what the adapter filled in.
TEMPLATE = """\
NODE=$node_dir
HISTORY=$history_dir
BASELINE=$baseline_dir
PROBLEM=$problem_file
PROGRAM=$eval_program
SEED=$seed
GUIDANCE=$direction_guidance
"""


def _agent(*flags: str, template: str | None = TEMPLATE, **options: Any) -> CommandAgent:
    return CommandAgent(
        [sys.executable, STAND_IN, *flags], template=template, timeout=TIMEOUT, **options
    )


def _context(workspace: Path, *, seed: int | None = 7) -> AgentContext:
    workspace.mkdir(parents=True, exist_ok=True)
    return AgentContext(
        problem=PROBLEM, workspace=workspace, history=(Node(id="n000000"),), seed=seed
    )


def _calls(log: Path) -> list[dict[str, Any]]:
    """What the stand-in recorded on each call, in the order the calls were made."""
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(log.glob("call_*"))]


@pytest.fixture
def log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "stand-in-log"
    monkeypatch.setenv("STAND_IN_LOG", str(directory))
    return directory


class LengthEvaluator:
    """Scores a program by its length, and says so, so the history has a score to show."""

    direction = ScoreDirection.HIGHER_IS_BETTER
    baseline_score = None

    def evaluate(self, artifact: str, workspace: Path) -> EvalResult:
        return EvalResult(score=float(len(artifact)), correct=True, diagnostics="measured length")


class Deepen:
    """Refines the newest leaf every round, so a rollout is one chain of attempts."""

    def select(self, tree: Any, eligible: Any, width: int) -> tuple[str, ...]:
        leaves = [node_id for node_id in eligible if node_id != tree.root_id]
        return (leaves[-1] if leaves else tree.root_id,)


@pytest.mark.parametrize("via", ["argument", "stdin"])
def test_the_command_writes_a_proposal_and_a_program_and_propose_returns_them(
    tmp_path: Path, log: Path, via: str
) -> None:
    flags = ["--stdin"] if via == "stdin" else []

    artifact = _agent(*flags, prompt_via=via).propose(_context(tmp_path / "ws"))

    assert artifact == Artifact(content="program for seed 7\n", proposal="proposal for seed 7\n")
    [call] = _calls(log)
    # The adapter adds nothing to the command line beyond the prompt (AGENTS.md rule 6):
    # as the last argument, or not at all when it goes on stdin.
    assert call["argv"] == ([*flags, call["prompt"]] if via == "argument" else flags)


def test_the_paths_in_the_prompt_are_absolute_however_the_workspace_was_named(
    tmp_path: Path, log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run directory the CLI is given is usually relative (``runs/quickstart``).

    The command runs *in* the workspace, so a relative path in its prompt would be
    read from there — one level down from where it was meant — and name nothing.
    """
    monkeypatch.chdir(tmp_path)

    _agent().propose(_context(Path("relative") / "workspace"))

    [call] = _calls(log)
    for name in ("NODE", "HISTORY", "BASELINE", "PROBLEM"):
        assert Path(call["fields"][name]).is_absolute(), name
    assert call["cwd"] == call["fields"]["NODE"]
    assert call["problem"] == PROBLEM


def test_the_seed_reaches_the_command_as_a_variable_and_as_an_environment_variable(
    tmp_path: Path, log: Path
) -> None:
    _agent().propose(_context(tmp_path / "ws", seed=41))

    [call] = _calls(log)
    assert call["fields"]["SEED"] == "41"
    assert call["seed_env"] == "41"


def test_each_attempt_reads_the_problem_and_every_inherited_attempt_with_its_score(
    tmp_path: Path, log: Path
) -> None:
    """Issue #70's first "tests first", through a real rollout over real snapshots.

    Three attempts in a chain: the third is handed the first two, each with the
    proposal and program it wrote and the score its evaluation recorded — which is
    the evidence §B.1 has the agent read before it proposes anything.
    """
    rollout = run_rollout(
        agent=_agent(),
        evaluator=LengthEvaluator(),
        policy=Deepen(),
        problem=PROBLEM,
        workspace=_workspace(tmp_path),
        snapshots=SnapshotStore(tmp_path / "store"),
        config=RolloutConfig(workers=1, max_rounds=3),
    )

    assert [node.parent_id for node in rollout.tree.iter_nodes()][1:] == [
        "n000000",
        "n000001",
        "n000002",
    ]
    first, second, third = _calls(log)
    assert [len(call["history"]) for call in (first, second, third)] == [0, 1, 2]
    assert third["problem"] == PROBLEM
    assert third["baseline_exists"]
    # The 1st and 2nd attempts, as the 3rd sees them: seeds 0 and 1 wrote these.
    assert third["history"]["attempt_n000001"] == {
        "proposal": "proposal for seed 0\n",
        "program": "program for seed 0\n",
        "score": {
            "score": float(len("program for seed 0\n")),
            "correct": True,
            "fail_class": "ok",
            "diagnostics": "measured length",
        },
        "error": None,
    }
    assert third["history"]["attempt_n000002"]["proposal"] == "proposal for seed 1\n"
    assert third["history"]["attempt_n000002"]["program"] == "program for seed 1\n"
    # Its own directory is where it works: the command's working directory.
    assert third["cwd"] == third["fields"]["NODE"]


def test_a_failed_attempt_is_shown_to_the_next_one_as_an_error(tmp_path: Path, log: Path) -> None:
    """§B.1: read ``eval/score.json``, "and ``error.txt`` if it failed"."""
    workspace = _workspace(tmp_path)
    store = SnapshotStore(tmp_path / "store")
    failing = _agent("--mode", "fail")

    run_rollout(
        agent=failing,
        evaluator=LengthEvaluator(),
        policy=Deepen(),
        problem=PROBLEM,
        workspace=workspace,
        snapshots=store,
        config=RolloutConfig(workers=1, max_rounds=2),
    )

    first, second = _calls(log)
    assert first["history"] == {}
    [attempt] = second["history"].values()
    assert attempt["score"] is None
    assert attempt["program"] is None
    assert "boom" in attempt["error"]


def test_history_is_not_carried_into_the_snapshots_the_tree_keeps(
    tmp_path: Path, log: Path
) -> None:
    """The history is a view laid out for one call, not part of the workspace a child resumes."""
    run_rollout(
        agent=_agent(),
        evaluator=LengthEvaluator(),
        policy=Deepen(),
        problem=PROBLEM,
        workspace=_workspace(tmp_path),
        snapshots=SnapshotStore(tmp_path / "store"),
        config=RolloutConfig(workers=1, max_rounds=3),
    )

    assert list((tmp_path / "store").rglob("attempt_*")) == []


@pytest.mark.parametrize(
    ("mode", "named"),
    [
        pytest.param("fail", "boom: the stand-in was told to fail", id="exits-non-zero"),
        # A good program does not redeem a run that ended in failure: whatever the CLI
        # went on to do after writing it is unknown.
        pytest.param("write-then-fail", "status 4: crashed after writing", id="crashes-at-the-end"),
        pytest.param("no-program", "wrote no solution.py", id="writes-no-program"),
        pytest.param("empty-program", "wrote no solution.py", id="writes-an-empty-program"),
    ],
)
def test_a_command_that_leaves_no_attempt_raises_with_what_it_printed(
    tmp_path: Path, log: Path, mode: str, named: str
) -> None:
    """Issue #70's second: never an empty artifact, and the reason is the CLI's own words."""
    with pytest.raises(CommandAgentError, match=named):
        _agent("--mode", mode).propose(_context(tmp_path / "ws"))


def test_a_command_that_hangs_is_stopped_at_the_timeout(tmp_path: Path, log: Path) -> None:
    hanging = CommandAgent([sys.executable, STAND_IN, "--mode", "hang"], template=TEMPLATE, timeout=1.0)

    with pytest.raises(CommandAgentError, match="timed out"):
        hanging.propose(_context(tmp_path / "ws"))


def test_a_command_that_prints_without_end_is_stopped_before_it_fills_the_disk(
    tmp_path: Path, log: Path
) -> None:
    flooding = CommandAgent(
        [sys.executable, STAND_IN, "--mode", "flood"],
        template=TEMPLATE,
        timeout=20.0,
        max_output_bytes=1024 * 1024,
    )
    started = time.monotonic()

    with pytest.raises(CommandAgentError, match="without end"):
        flooding.propose(_context(tmp_path / "ws"))

    assert time.monotonic() - started < 10


@pytest.mark.parametrize("escaping", [".dream_rsi", "solution.py", "proposal.md"])
def test_a_symlink_a_previous_attempt_left_is_never_written_or_read_through(
    tmp_path: Path, log: Path, escaping: str
) -> None:
    """The workspace was resumed from a snapshot, and the agent that made it is a model.

    A link out of it would send the harness's writes (the history copy, the saved
    proposals) and its removals — or its reads of the program — somewhere else, with the
    harness's permissions. It is refused before the CLI is spent.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside"
    (outside / "history").mkdir(parents=True)
    (outside / "history" / "precious.txt").write_text("keep me\n", encoding="utf-8")
    (outside / "file.txt").write_text("keep me too\n", encoding="utf-8")
    if escaping == ".dream_rsi":
        (workspace / escaping).symlink_to(outside, target_is_directory=True)
    else:
        (workspace / escaping).symlink_to(outside / "file.txt")

    with pytest.raises(CommandAgentError, match="outside the workspace"):
        _agent().propose(_context(workspace))

    assert (outside / "history" / "precious.txt").read_text(encoding="utf-8") == "keep me\n"
    assert (outside / "file.txt").read_text(encoding="utf-8") == "keep me too\n"
    assert _calls(log) == [], "the CLI was run although the workspace could not be trusted"


def test_a_program_the_parent_already_had_is_not_a_new_attempt(tmp_path: Path, log: Path) -> None:
    """A workspace is resumed from its parent's, which holds the parent's program.

    A command that leaves it alone has proposed nothing; returning it would be
    recording a copy of the parent as a fresh, evaluated attempt.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "solution.py").write_text("the parent's program\n", encoding="utf-8")

    with pytest.raises(CommandAgentError, match="left solution.py unchanged"):
        _agent("--mode", "unchanged").propose(_context(workspace))


def test_a_proposal_the_parent_wrote_is_not_taken_for_this_attempts(
    tmp_path: Path, log: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "proposal.md").write_text("the parent's rationale\n", encoding="utf-8")

    artifact = _agent("--mode", "no-proposal").propose(_context(workspace))

    assert artifact.content == "program for seed 7\n"
    assert artifact.proposal == ""


def test_the_default_template_is_the_papers_exploration_prompt_with_its_variables_filled(
    tmp_path: Path, log: Path
) -> None:
    agent = _agent(template=None, direction_guidance="prefer small programs")
    workspace = tmp_path / "ws"

    agent.propose(_context(workspace))

    [call] = _calls(log)
    prompt = call["prompt"]
    assert "You must read every historical proposal" in prompt
    assert "prefer small programs" in prompt
    assert str(workspace.resolve()) in prompt
    # Filled in, not left for the model to puzzle over.
    assert "$node_dir" not in prompt
    assert "$history_dir" not in prompt


def test_a_template_naming_a_variable_nothing_supplies_is_refused_up_front() -> None:
    with pytest.raises(ValueError, match="template"):
        CommandAgent([sys.executable], template="read $no_such_variable")


def test_a_command_is_an_argv_list_because_there_is_no_shell_to_split_a_string() -> None:
    with pytest.raises(TypeError, match="argv list"):
        CommandAgent("claude -p")  # type: ignore[arg-type]


def test_a_command_that_cannot_be_started_raises_naming_it(tmp_path: Path) -> None:
    missing = CommandAgent(["no-such-cli-binary-anywhere"], template=TEMPLATE, timeout=TIMEOUT)

    with pytest.raises(CommandAgentError, match="no-such-cli-binary-anywhere"):
        missing.propose(_context(tmp_path / "ws"))


def test_a_failed_command_is_recorded_as_a_failed_node_by_the_orchestrator(
    tmp_path: Path, log: Path
) -> None:
    """One path for a failure: the exception, and the rollout's own handling of it."""
    rollout = run_rollout(
        agent=_agent("--mode", "fail"),
        evaluator=LengthEvaluator(),
        policy=Deepen(),
        problem=PROBLEM,
        workspace=_workspace(tmp_path),
        snapshots=SnapshotStore(tmp_path / "store"),
        config=RolloutConfig(workers=1, max_rounds=1),
    )

    [node] = [node for node in rollout.tree.iter_nodes() if node.parent_id is not None]
    assert node.score is None
    assert node.artifact is None
    assert "boom" in node.diagnostics["error"]


def test_the_loop_completes_a_cycle_with_the_command_as_its_discovery_agent(
    tmp_path: Path, log: Path
) -> None:
    """Issue #70's third: ``run_cycles`` over the stand-in and the toy evaluator, default prompt."""
    run = run_cycles(
        agent=_agent("--mode", "plan", template=None),
        evaluator=ToySearchEvaluator(),
        developer=FakeDeveloper(script=TOY_REVISIONS),
        policy=DEFAULT_POLICY_SOURCE,
        problem=TOY_PROBLEM,
        directory=tmp_path / "run",
        config=RunConfig(cycles=1, rollout=RolloutConfig(workers=2, max_rounds=2), versions=2),
    )

    [cycle] = run.cycles
    assert cycle.cost.online.agent_calls > 0
    assert cycle.best_score is not None
    assert len(_calls(log)) == cycle.cost.online.agent_calls


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "root-workspace"
    workspace.mkdir()
    return workspace
