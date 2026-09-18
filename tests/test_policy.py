"""The shared decision interface and the baseline policies (issue #10).

The baselines are what a dreaming round has to compare against and what the
policy-development agent starts from (issues #12, #14), so what matters about
them is that they are three different strategies which the replay objective can
tell apart — and that each one drives an online rollout and a replay of the
tree that rollout recorded with the same decisions, because a policy that can
tell the two apart makes the dreaming signal meaningless (§3).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from dream_rsi.adapters.toy_search import ToySearchAgent, ToySearchEvaluator, plan_source
from dream_rsi.orchestrator import ExplorationPolicy, RolloutConfig, run_rollout
from dream_rsi.policy import (
    BreadthFirstPolicy,
    BudgetAwarePolicy,
    GreedyBestFirstPolicy,
    OptimalPolicy,
)
from dream_rsi.replay import STOP_EMPTY_BATCH, ReplaySimulator
from dream_rsi.scoring import replay_score
from dream_rsi.tree import DiscoveryTree, eligible_nodes

REPO = Path(__file__).resolve().parents[1]
TREES = Path(__file__).parent / "fixtures" / "trees"

NAMES = ("wide_shallow", "narrow_deep", "failing_branch")
BASELINES = (BreadthFirstPolicy, GreedyBestFirstPolicy, BudgetAwarePolicy)

# Spans the toy landscape the way a real rollout would have to: the local
# optimum, the valley, an over-budget plan that evaluates fine and scores 0.0,
# an unreadable artifact that scores nothing at all, and the best admissible
# plan. A policy that only ever sees successes is not being asked anything.
SCRIPT = (
    plan_source(1, 1),
    plan_source(2, 2),
    "# out of ideas, no plan this time\n",
    plan_source(4, 4),
    plan_source(3, 3),
    plan_source(0, 1),
)


def _load(name: str) -> DiscoveryTree:
    return DiscoveryTree.load(TREES / name / "tree.json")


def _replayed(policy: OptimalPolicy, name: str, *, width: int = 3):
    return ReplaySimulator(_load(name)).replay(policy, width=width).result()


class SelectsAnInteriorNode(OptimalPolicy):
    """Deepens once, then reaches back for the node it just left behind.

    The node it asks for exists and was revealed; it is simply no longer a
    frontier, which is what prefix-observability forbids (§3).
    """

    def choose(
        self, tree: DiscoveryTree, live: Sequence[str], width: int
    ) -> Sequence[str]:
        leaves = [node_id for node_id in live if node_id != tree.root_id]
        if not leaves:
            return (tree.root_id,)
        parent = tree.node(leaves[-1]).parent_id
        return (parent,) if parent != tree.root_id else (leaves[-1],)


class NeverStops(OptimalPolicy):
    """Selects the root for ever, whatever the world has already refused it."""

    def choose(
        self, tree: DiscoveryTree, live: Sequence[str], width: int
    ) -> Sequence[str]:
        return (tree.root_id,)


class BoundedQuestion:
    """A ``Question`` that refuses to be driven more than ``limit`` rounds.

    Anything carrying these four members is a world ``solve`` can drive (§B.2's
    ``question``), which a replay run is one of. Wrapping one here turns a
    ``solve`` that fails to terminate into a failing test rather than a hanging
    suite — and holds ``solve`` to the protocol it declares instead of to
    whatever else ``ReplayRun`` happens to expose.
    """

    def __init__(self, run: Any, limit: int) -> None:
        self._run = run
        self._limit = limit
        self._rounds = 0

    @property
    def revealed(self) -> DiscoveryTree:
        return self._run.revealed

    @property
    def complete(self) -> bool:
        return self._run.complete

    def reveal(self, batch: Sequence[str]) -> Any:
        self._rounds += 1
        assert self._rounds <= self._limit, (
            f"solve took more than {self._limit} rounds without terminating"
        )
        return self._run.reveal(batch)

    def result(self) -> Any:
        return self._run.result()


class SelectsANodeThatDoesNotExist(OptimalPolicy):
    """Asks for a node id no tree ever held — the other way a selection is illegal."""

    def choose(
        self, tree: DiscoveryTree, live: Sequence[str], width: int
    ) -> Sequence[str]:
        return ("n999999",)


@pytest.mark.parametrize("baseline", BASELINES)
def test_each_baseline_satisfies_the_shared_decision_interface(
    baseline: type[OptimalPolicy],
) -> None:
    """One object, both entry points: the driver's ``select`` and the paper's ``solve``.

    §3: "both the online and offline phases use this same decision interface",
    and §B.2 names it ``OptimalPolicy.solve(self, question, budget=None)`` —
    which is what the policy-development agent writes against in issue #14. A
    baseline that implements only one of the two cannot be driven by both.
    """
    policy = baseline()

    assert isinstance(policy, ExplorationPolicy)

    result = policy.solve(ReplaySimulator(_load("failing_branch")).start())

    assert result.rounds, "solve returned without taking a single decision round"
    assert result.revealed >= 1


@pytest.mark.parametrize("baseline", BASELINES)
@pytest.mark.parametrize("name", NAMES)
def test_each_baseline_runs_to_completion_on_every_fixture(
    baseline: type[OptimalPolicy], name: str
) -> None:
    """Every baseline terminates on every recorded shape without an invalid selection.

    The three fixtures are the shapes a policy has to survive: eight siblings
    off the root, an unbranched chain, and a tree holding a hard failure with no
    score at all. An illegal selection raises out of the driver, so reaching a
    stop reason at all is the assertion; the rest insists the replay actually
    explored something rather than stopping on round one.
    """
    result = _replayed(baseline(), name)

    assert result.stop_reason is not None
    assert result.revealed >= 1
    assert result.attainment is not None
    assert result.round_count == len(result.rounds)


@pytest.mark.parametrize("baseline", BASELINES)
def test_a_recorded_rollout_replays_to_the_trajectory_that_recorded_it(
    baseline: type[OptimalPolicy], tmp_path: Path
) -> None:
    """The same policy, online and in replay, makes the same decisions (§3).

    This is the invariant the dreaming signal rests on: a replay world hands
    back the recorded children of the nodes a policy selects, so the policy that
    recorded a tree must retrace it. It fails if a policy branches on which mode
    it is in, reads anything replay cannot show it, or misreads a barren reveal
    as an outcome — and it fails if the rollout attaches children in an order
    the replay does not reproduce.
    """
    rollout = run_rollout(
        agent=ToySearchAgent(script=SCRIPT),
        evaluator=ToySearchEvaluator(),
        policy=baseline(),
        problem="pick the plan (width, depth) with the best throughput within the cost budget",
        workspace=tmp_path,
        config=RolloutConfig(workers=3, max_rounds=5, max_nodes=None, seed=0),
    )

    # A one-round or one-attempt recording would compare two trivial
    # trajectories. The two batching baselines also put several selections in a
    # round here, so the comparison covers lining a batch up as well as a
    # singleton.
    assert len(rollout.rounds) > 1 and len(rollout.tree) - 1 > 2, (
        "too short a rollout to test retracing a trajectory"
    )

    replayed = (
        ReplaySimulator(rollout.tree)
        .replay(baseline(), width=3, max_rounds=len(rollout.rounds))
        .result()
    )

    assert [round_.selected for round_ in replayed.rounds] == [
        round_.selected for round_ in rollout.rounds
    ]
    assert [round_.revealed for round_ in replayed.rounds] == [
        round_.produced for round_ in rollout.rounds
    ]


def test_a_policy_that_selects_a_node_outside_the_frontier_is_refused() -> None:
    """An illegal selection is an error, not a silently dropped id.

    A policy is model-written code (issue #13). One that reaches back into the
    revealed prefix for a node that is no longer a frontier has broken
    prefix-observability, and dropping the id would score that version on a
    traversal it was not entitled to. Both entry points are held to it, and
    neither leaves the run half-advanced.

    The error also has to name the node, for both a node that exists and one
    that never did: a policy's own bookkeeping failing first with a bare
    ``KeyError`` tells whoever reads the dreaming log nothing about what the
    version did wrong.
    """
    world = ReplaySimulator(_load("narrow_deep"))

    driven = world.start()
    with pytest.raises(ValueError, match=r"outside A\(T\).*n000001"):
        SelectsAnInteriorNode().solve(driven)

    assert driven.result().revealed == 2, "the refused round advanced the run anyway"

    with pytest.raises(ValueError, match=r"outside A\(T\).*n000001"):
        world.replay(SelectsAnInteriorNode())

    with pytest.raises(ValueError, match=r"outside A\(T\).*n999999"):
        world.replay(SelectsANodeThatDoesNotExist())


def test_the_baselines_decide_differently_on_the_same_revealed_prefix() -> None:
    """Three strategies, one prefix: what each of them does with it.

    The prefix holds a shallow frontier scoring 5.0 and, one level down a branch
    that scored 9.0, a frontier that regressed to 3.0. Each baseline's stated
    rule picks a different node out of it, and the three ways of getting this
    wrong are all live: a "breadth-first" policy that actually deepens, a
    "greedy" one that ranks a frontier by its own latest score rather than its
    branch's successful anchor (§B.2) — which would drop the branch holding the
    best result so far over one regression — and a portfolio that never spends a
    slot on exploration.
    """
    tree = DiscoveryTree.with_root()
    shallow = tree.add_child(tree.root_id, score=5.0)
    strong = tree.add_child(tree.root_id, score=9.0)
    regressed = tree.add_child(strong.id, score=3.0)
    eligible = eligible_nodes(tree)

    assert eligible == (tree.root_id, shallow.id, regressed.id)
    assert BreadthFirstPolicy().select(tree, eligible, 2) == (shallow.id,)
    assert GreedyBestFirstPolicy().select(tree, eligible, 2) == (regressed.id,)
    assert BudgetAwarePolicy().select(tree, eligible, 2) == (tree.root_id, regressed.id)


def test_the_replay_objective_tells_the_baselines_apart() -> None:
    """Issue #10's "done when": the baselines score differently under Equation 1.

    Three strategies that all score the same are one strategy written three
    ways, and a dreaming round comparing them would be measuring nothing. If
    this fails because the fixtures have no real structure to exploit, issue #6
    is what needs revisiting, not this test.
    """
    scores = {
        name: {
            baseline.__name__: replay_score(
                (result := _replayed(baseline(), name)).attainment,
                revealed=result.revealed,
                rounds=result.round_count,
            )
            for baseline in BASELINES
        }
        for name in NAMES
    }

    for name, by_policy in scores.items():
        assert len(set(by_policy.values())) > 1, f"every baseline scored the same on {name}"

    for first in BASELINES:
        for second in BASELINES:
            if first is second:
                continue
            assert any(
                scores[name][first.__name__] != scores[name][second.__name__] for name in NAMES
            ), f"{first.__name__} and {second.__name__} score identically on every fixture"


def test_the_budget_aware_baseline_spends_what_beta_tells_it_to() -> None:
    """``beta`` is the one knob the paper's policies expose (§B.2), and it bites.

    "High beta means more width, deeper patience, and weaker pruning. Low beta
    means fewer probes, earlier stagnation stops, and stronger pruning." A
    policy whose thresholds are hardcoded would replay the same world the same
    way under every beta, and the offline beta sweep would report a trade-off
    that does not exist.
    """
    frugal = _replayed(BudgetAwarePolicy(config={"beta": 0.25}), "wide_shallow")
    patient = _replayed(BudgetAwarePolicy(config={"beta": 2.0}), "wide_shallow")

    assert frugal.revealed < patient.revealed
    assert frugal.stop_reason == STOP_EMPTY_BATCH, "the frugal policy was stopped, it did not stop"


def test_solve_stops_once_the_budget_is_spent() -> None:
    """``solve``'s ``budget`` caps what the policy is allowed to reveal (§B.2).

    Replay calls with ``budget=None`` and the policy must terminate on its own
    there; online it is handed a cap, and a policy that ignores it would spend a
    real generation budget it was not given. A batch is cut to what is left, too:
    the last round of a budgeted run is where a policy asking for full width
    would overshoot.
    """
    run = ReplaySimulator(_load("wide_shallow")).start()

    result = BreadthFirstPolicy().solve(run, budget=3)

    assert result.revealed == 3

    batched = ReplaySimulator(_load("wide_shallow")).start()
    batched.max_parallelism = 4

    truncated = BreadthFirstPolicy().solve(batched, budget=3)

    # Four branches asked for in the first round, three of them paid for.
    assert truncated.revealed == 3
    assert truncated.rounds[0].selected == (batched.revealed.root_id,) * 3


def test_solve_returns_when_a_policy_stops_getting_anywhere() -> None:
    """A policy that never stops does not hang the run that is scoring it.

    §B.2 puts termination on the policy — "Always terminate when no batch is
    selected" — and a policy is model-written code, so some version will not.
    Replay's driver caps its rounds at ``K₂``; ``solve`` has no such number, so
    it stops a policy that has gone longer without revealing anything than it
    has frontiers left to close. A dreaming round over ``M`` versions and every
    recorded tree (issue #12) cannot afford one of them to spin.
    """
    run = ReplaySimulator(_load("narrow_deep")).start()

    result = NeverStops().solve(BoundedQuestion(run, limit=12))

    # The one child the root has recorded, and then nothing: the rounds after it
    # asked the world for a branch it does not hold.
    assert result.revealed == 1
    assert result.stop_reason is None, "solve is not the driver; it sets no stop reason"


def test_solve_batches_to_the_width_the_question_offers() -> None:
    """``solve`` reads ``W`` off the question it is driving (§B.2).

    A ``solve`` that always went one node at a time would take the whole of
    Equation 1's cost term and none of its parallelism refund, and a batching
    policy driven through it would look serial — so the offline comparison would
    be measuring the entry point rather than the policy.
    """
    run = ReplaySimulator(_load("wide_shallow")).start()
    # §B.2's ``question.max_parallelism``: the worker count a world advertises.
    run.max_parallelism = 3

    result = BreadthFirstPolicy().solve(run)

    assert max(len(round_.selected) for round_ in result.rounds) == 3


def test_importing_the_policy_module_reaches_no_agent_and_no_evaluator() -> None:
    """Policies run inside the frozen world, so their module stays off the adapters.

    ``tests/test_replay.py`` asserts this of the simulator; a policy is the
    other half of the replay path. If importing it pulled an adapter in, a
    model-written policy could call a discovery agent mid-replay and be scored
    on an outcome nothing recorded.
    """
    completed = subprocess.run(
        [sys.executable, "-c", "import dream_rsi.policy, sys; print('\\n'.join(sys.modules))"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    adapters = sorted(
        name for name in completed.stdout.split() if name.startswith("dream_rsi.adapters")
    )
    assert not adapters, f"importing dream_rsi.policy pulled in {', '.join(adapters)}"
