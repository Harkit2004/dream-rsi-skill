"""The online rollout loop (issue #4)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dream_rsi.adapters.agent import AgentContext, Artifact
from dream_rsi.adapters.evaluator import EvalResult, ScoreDirection
from dream_rsi.adapters.fake_agent import FakeAgent
from dream_rsi.adapters.toy_evaluator import ToyEvaluator
from dream_rsi.orchestrator import (
    RolloutConfig,
    eligible_nodes,
    main,
    run_rollout,
)
from dream_rsi.tree import DiscoveryTree
from dream_rsi.workspace import SnapshotStore

PROBLEM = "write the shortest program that solves it"


@dataclass
class ScriptedPolicy:
    """Selects, each round, the eligible nodes at the scripted positions.

    Positions rather than node ids so a test says what it means — position 0 is
    always the root — and an exhausted script selects the empty batch.
    """

    script: tuple[tuple[int, ...], ...]
    widths: list[int] = field(default_factory=list)
    _round: int = 0

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        self.widths.append(width)
        if self._round >= len(self.script):
            return ()
        picks = self.script[self._round]
        self._round += 1
        return tuple(eligible[position] for position in picks)


@dataclass
class ReselectPolicy:
    """Opens a branch, refines it, then selects that branch's now-interior node."""

    selected: list[str] = field(default_factory=list)

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        if len(self.selected) < 2:
            # The root in round 0 (nothing else is eligible), then the leaf it
            # produced in round 1 — which round 1 turns into an interior node.
            self.selected.append(eligible[-1])
            return (self.selected[-1],)
        return (self.selected[1],)


@dataclass(frozen=True)
class SeedAgent:
    """Stamps each attempt's seed into its artifact, so attempts can be told apart.

    With ``jitter`` set, even-seeded attempts — the ones submitted first, since
    seeds ascend with submission — take the longest to return. A rollout that
    attaches children as workers finish then records different artifacts from one
    that attaches them in the order the policy selected their parents.
    """

    jitter: float = 0.0

    def propose(self, context: AgentContext) -> Artifact:
        seed = context.seed or 0
        if self.jitter and seed % 2 == 0:
            time.sleep(self.jitter)
        return Artifact(content=f"def solve(values):  # attempt {seed}\n    return sum(values)\n")


@dataclass(frozen=True)
class ExplodingAgent:
    """Raises on odd-seeded attempts and behaves on the rest."""

    inner: FakeAgent = field(default_factory=FakeAgent)

    def propose(self, context: AgentContext) -> Artifact:
        if (context.seed or 0) % 2:
            raise RuntimeError("agent exploded")
        return self.inner.propose(context)


@dataclass(frozen=True)
class TracingAgent:
    """Appends the node it resumed from to ``trace.txt`` in its own workspace.

    So the file an attempt leaves behind says which branch produced it, and a
    workspace that inherited another branch's writes is visible as such.
    """

    def propose(self, context: AgentContext) -> Artifact:
        with (context.workspace / "trace.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"{context.parent.id}\n")
        return Artifact(content="def solve(values):\n    return sum(values)\n")


@dataclass(frozen=True)
class ExplodingEvaluator:
    """Raises on artifacts it is asked to measure after the first."""

    direction: ScoreDirection = ScoreDirection.LOWER_IS_BETTER
    baseline_score: float | None = None
    seen: list[str] = field(default_factory=list)

    def evaluate(self, artifact: str, workspace: Path) -> EvalResult:
        self.seen.append(artifact)
        if len(self.seen) > 1:
            raise RuntimeError("evaluator exploded")
        return EvalResult(score=1.0, correct=True)


def rollout(
    policy, tmp_path, *, agent=None, evaluator=None, snapshots=None, workspace=None, **config
):
    return run_rollout(
        agent=agent if agent is not None else FakeAgent(),
        evaluator=evaluator if evaluator is not None else ToyEvaluator(),
        policy=policy,
        problem=PROBLEM,
        workspace=tmp_path if workspace is None else workspace,
        snapshots=snapshots,
        config=RolloutConfig(**config),
    )


def attempts(result):
    """Every node the rollout recorded, root excluded."""
    return [node for node in result.tree.iter_nodes() if node.parent_id is not None]


def test_each_attempt_is_recorded_as_a_child_of_the_node_the_policy_selected(tmp_path):
    # Round 0 opens a branch from the root; round 1 refines that branch and opens
    # a second one; round 2 stops.
    policy = ScriptedPolicy(script=((0,), (0, 1), ()))

    result = rollout(policy, tmp_path, workers=2)

    root = result.tree.root_id
    first = result.tree.children(root)[0]
    assert [node.parent_id for node in attempts(result)] == [root, root, first.id]
    # Every attempt carries the artifact it produced and the score the evaluator
    # gave it, so the tree stands on its own as a replay world.
    for node in attempts(result):
        assert node.artifact
        assert node.score is not None
        assert node.diagnostics["fail_class"] == "ok"
    assert result.stop_reason == "empty_batch"
    # The policy is offered the paper's W as the feasible batch width (§3).
    assert policy.widths[:1] == [2]


def test_each_round_records_the_width_asked_for_and_the_children_it_produced(tmp_path):
    policy = ScriptedPolicy(script=((0, 0), (0, 1, 2)))

    result = rollout(policy, tmp_path, workers=3, max_rounds=2)

    assert [round_.k for round_ in result.rounds] == [2, 3]
    # selected/produced align pairwise, which is what a replay reconstructing
    # "what was observable when" needs (issue #7).
    revealed: list[str] = []
    for round_ in result.rounds:
        assert len(round_.selected) == len(round_.produced)
        for parent_id, child_id in zip(round_.selected, round_.produced, strict=True):
            assert result.tree.node(child_id).parent_id == parent_id
            # A node cannot be selected before the round that produced it.
            assert parent_id == result.tree.root_id or parent_id in revealed
        revealed.extend(round_.produced)
    assert revealed == [node.id for node in attempts(result)]


def test_worker_count_does_not_change_the_recorded_tree(tmp_path):
    # The batches this policy asks for do not depend on the width it is offered,
    # so W is left controlling only how many attempts run at once — at W=1 each
    # three-node batch runs in three sequential waves (§B.2's ceil(k / W)).
    script = ((0, 0, 0), (0, 1, 2), (1, 2, 3))

    agent = SeedAgent(jitter=0.02)
    serial = rollout(ScriptedPolicy(script=script), tmp_path, agent=agent, workers=1)
    parallel = rollout(ScriptedPolicy(script=script), tmp_path, agent=agent, workers=4)

    assert len(attempts(serial)) == 9
    assert serial.tree == parallel.tree


@pytest.mark.parametrize(
    ("config", "expected_attempts", "expected_stop"),
    [
        ({"max_rounds": 2}, 6, "max_rounds"),
        # 3 per round into a budget of 5: the second round has to stop short
        # rather than overshoot to 6.
        ({"max_rounds": 8, "max_nodes": 5}, 5, "max_nodes"),
        ({"max_rounds": 8, "max_nodes": 0}, 0, "max_nodes"),
        ({"max_rounds": 0}, 0, "max_rounds"),
        ({"max_rounds": 8, "max_seconds": 0.0}, 0, "wall_clock"),
    ],
)
def test_the_budget_is_respected_exactly(tmp_path, config, expected_attempts, expected_stop):
    policy = ScriptedPolicy(script=((0, 0, 0),) * 8)

    result = rollout(policy, tmp_path, workers=3, **config)

    assert len(attempts(result)) == expected_attempts
    assert result.stop_reason == expected_stop


def test_a_round_cut_short_by_the_budget_still_records_the_width_asked_for(tmp_path):
    # Equation 1's batching term divides by the k the policy chose (issue #8), so
    # a round the budget truncated must not look like a narrower decision.
    policy = ScriptedPolicy(script=((0, 0, 0),) * 4)

    result = rollout(policy, tmp_path, workers=3, max_rounds=4, max_nodes=5)

    assert [(round_.k, len(round_.produced)) for round_ in result.rounds] == [(3, 3), (3, 2)]


@pytest.mark.parametrize("failing", ["agent", "evaluator"])
def test_a_failing_attempt_is_recorded_and_the_rollout_continues(tmp_path, failing):
    policy = ScriptedPolicy(script=((0, 0), (0, 0)))
    parts = (
        {"agent": ExplodingAgent()} if failing == "agent" else {"evaluator": ExplodingEvaluator()}
    )

    result = rollout(policy, tmp_path, workers=2, max_rounds=2, **parts)

    recorded = attempts(result)
    assert len(recorded) == 4, "a failed attempt must not abort the rollout"
    failed = [node for node in recorded if node.score is None]
    assert failed, "a failed attempt must still be recorded as a node"
    for node in failed:
        assert node.diagnostics["fail_class"] != "ok"
        assert "exploded" in node.diagnostics["error"]


def test_selecting_a_node_that_is_not_eligible_is_rejected(tmp_path):
    # Round 0 gives the root a child; round 1 gives that child one, which makes
    # it an interior node. A(T) is the root plus the leaves (§3), so reselecting
    # it in round 2 is not a legal action.
    policy = ReselectPolicy()

    with pytest.raises(ValueError, match=r"outside A\(T\)"):
        rollout(policy, tmp_path, max_rounds=3)

    assert len(policy.selected) == 2


def test_eligible_nodes_are_the_root_and_the_leaves(tmp_path):
    result = rollout(ScriptedPolicy(script=((0,), (0, 1))), tmp_path, workers=2, max_rounds=2)

    root = result.tree.root_id
    leaves = [node.id for node in attempts(result) if not result.tree.children(node.id)]
    assert eligible_nodes(result.tree) == (root, *sorted(leaves))


def test_the_rollout_is_reproducible(tmp_path):
    script = ((0, 0), (0, 1, 2))

    first = rollout(ScriptedPolicy(script=script), tmp_path, agent=SeedAgent(), workers=4)
    second = rollout(ScriptedPolicy(script=script), tmp_path, agent=SeedAgent(), workers=4)

    assert first.tree == second.tree
    # Two attempts from the same parent in the same round are distinguishable:
    # they are separate branches, not a duplicate record.
    opened = [node.artifact for node in first.tree.children(first.tree.root_id)]
    assert len(set(opened)) > 1


def test_the_saved_rollout_reloads_as_a_tree_beside_its_round_log(tmp_path):
    result = rollout(ScriptedPolicy(script=((0, 0),)), tmp_path, workers=2, max_rounds=1)

    result.save(tmp_path / "run")

    assert DiscoveryTree.load(tmp_path / "run" / "tree.json") == result.tree
    log = json.loads((tmp_path / "run" / "rounds.json").read_text(encoding="utf-8"))
    assert log["stop_reason"] == result.stop_reason
    assert [round_["produced"] for round_ in log["rounds"]] == [
        list(round_.produced) for round_ in result.rounds
    ]
    assert [round_["k"] for round_ in log["rounds"]] == [2]


def test_a_two_level_tree_replays_its_workspaces_from_disk(tmp_path):
    # The issue's "done when" (#5): after the run, the recorded tree plus the
    # snapshot store are enough to put any node's workspace back — which is what
    # "the parent's saved workspace" means once the rollout is over.
    store = SnapshotStore(tmp_path / "store")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "problem.txt").write_text(PROBLEM, encoding="utf-8")
    # Round 0 opens two branches off the root; round 1 refines each of them.
    policy = ScriptedPolicy(script=((0, 0), (1, 2)))

    result = rollout(
        policy,
        tmp_path,
        agent=TracingAgent(),
        snapshots=store,
        workspace=workspace,
        workers=2,
        max_rounds=2,
    )
    result.save(tmp_path / "run")

    tree = DiscoveryTree.load(tmp_path / "run" / "tree.json")
    reopened = SnapshotStore(tmp_path / "store")
    leaves = [node for node in tree.iter_nodes() if not tree.children(node.id)]
    assert len(leaves) == 2
    traces = []
    for leaf in leaves:
        restored = reopened.materialize(leaf.snapshot_ref, tmp_path / "restored" / leaf.id)
        # The root workspace reached a grandchild, and the writes along the way
        # are exactly this branch's — no sibling's, and in order.
        assert (restored / "problem.txt").read_text(encoding="utf-8") == PROBLEM
        ancestry = [tree.node(leaf.parent_id).parent_id, leaf.parent_id]
        assert (restored / "trace.txt").read_text(encoding="utf-8").split() == ancestry
        traces.append(tuple(ancestry))
    assert len(set(traces)) == 2


def test_module_entrypoint_writes_a_loadable_tree(tmp_path):
    # The issue's "done when": `python -m dream_rsi.orchestrator` on the toy task.
    completed = subprocess.run(
        [sys.executable, "-m", "dream_rsi.orchestrator", str(tmp_path / "run")],
        capture_output=True,
        text=True,
        # src on the path so this passes whether or not the package is installed;
        # the rest of the environment is inherited, because a hosted runner's
        # interpreter needs its own LD_LIBRARY_PATH to start at all.
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    tree = DiscoveryTree.load(tmp_path / "run" / "tree.json")
    assert len(tree) > 1
    assert main([str(tmp_path / "second")]) == 0
