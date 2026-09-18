"""The frozen replay world and its prefix-observable reveal (issue #7)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from dream_rsi.replay import (
    STOP_ALL_REVEALED,
    STOP_EMPTY_BATCH,
    STOP_MAX_ROUNDS,
    ReplaySimulator,
)
from dream_rsi.tree import DiscoveryTree

REPO = Path(__file__).resolve().parents[1]
TREES = Path(__file__).parent / "fixtures" / "trees"

NAMES = ("wide_shallow", "narrow_deep", "failing_branch")


def _load(name: str) -> DiscoveryTree:
    return DiscoveryTree.load(TREES / name / "tree.json")


def _rounds(name: str) -> list[dict]:
    return json.loads((TREES / name / "rounds.json").read_text(encoding="utf-8"))["rounds"]


def _ids(tree: DiscoveryTree) -> set[str]:
    return {node.id for node in tree.iter_nodes()}


@dataclass
class OpenBranches:
    """Keeps opening branches off the root; stops once a round revealed nothing."""

    per_round: int = 2
    _size: int = 0

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        if len(tree) == self._size:
            return ()
        self._size = len(tree)
        return (tree.root_id,) * self.per_round


@dataclass
class ExtendDeepest:
    """Refines the last eligible leaf; stops once a round revealed nothing."""

    _size: int = 0

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        if len(tree) == self._size:
            return ()
        self._size = len(tree)
        return (eligible[-1],)


@dataclass
class RecordedRounds:
    """Selects exactly what the recording's round log selected, then stops."""

    rounds: list[dict]
    _index: int = 0

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        if self._index >= len(self.rounds):
            return ()
        selected = tuple(self.rounds[self._index]["selected"])
        self._index += 1
        return selected


@dataclass
class GrabbyPolicy:
    """Writes into the tree it is handed, which must not be the recorded world."""

    rounds: int = 2
    _index: int = 0

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        tree.add_child(tree.root_id, artifact="never recorded", score=1e9)
        if self._index >= self.rounds:
            return ()
        self._index += 1
        return (tree.root_id,)


@dataclass
class AlwaysRoot:
    """Selects the root forever, exhausted branches or not."""

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        return (tree.root_id,)


def test_selecting_an_unrevealed_node_raises() -> None:
    """A policy may only select from A(T) over the *revealed* tree (§3).

    A recorded node the policy has not reached yet is not selectable, however
    well the policy knows the world's id scheme: reaching it requires revealing
    its parent first.
    """
    tree = _load("failing_branch")
    run = ReplaySimulator(tree).start()
    unrevealed = tree.children(tree.root_id)[0].id

    with pytest.raises(ValueError, match=unrevealed):
        run.reveal((unrevealed,))

    # The rejected batch changed nothing: the root still opens that same branch.
    assert run.reveal((tree.root_id,)).revealed == (unrevealed,)


def test_selecting_a_revealed_interior_node_raises() -> None:
    """A(T) is the root and the revealed *leaves* — not every revealed node.

    Re-selecting an interior node is how a replay would walk off the prefix:
    the recording may hold a second child there, and revealing it without
    revealing the branch it hangs under is not a prefix of any trajectory.
    """
    tree = _load("failing_branch")
    run = ReplaySimulator(tree).start()

    opened = run.reveal((tree.root_id,)).revealed[0]
    run.reveal((opened,))

    with pytest.raises(ValueError, match=opened):
        run.reveal((opened,))


def test_a_leaf_with_no_recorded_continuation_reveals_nothing_and_the_run_continues() -> None:
    """Walking off the recorded tree returns the empty set, not a fallback.

    The dead branch in ``failing_branch`` was never continued, so selecting it
    reveals nothing — the honest signal that this branch is unexplored — and the
    replay carries on with the branches that were.
    """
    tree = _load("failing_branch")
    run = ReplaySimulator(tree).start()

    opened = run.reveal((tree.root_id, tree.root_id, tree.root_id)).revealed
    dead = next(node_id for node_id in opened if not tree.children(node_id))
    live = next(node_id for node_id in opened if tree.children(node_id))

    assert run.reveal((dead,)).revealed == (None,)
    assert _ids(run.revealed) == {tree.root_id, *opened}, "a dead end revealed something"

    assert run.reveal((live,)).revealed == (tree.children(live)[0].id,)


@pytest.mark.parametrize("name", NAMES)
def test_two_policies_reveal_different_subsets_and_neither_mutates_the_tree(name: str) -> None:
    """The point of a replay world: one recording, many trajectories over it."""
    tree = _load(name)
    world = ReplaySimulator(tree)

    breadth = world.replay(OpenBranches())
    depth = world.replay(ExtendDeepest())

    assert _ids(breadth.revealed) != _ids(depth.revealed)
    for run in (breadth, depth):
        assert _ids(run.revealed) <= _ids(tree), "revealed a node the recording never held"
        assert tree.root_id in _ids(run.revealed)
    assert tree == _load(name), "replay wrote into the recorded tree"


def test_a_policy_cannot_write_into_the_recorded_world() -> None:
    """What the policy is handed each round is a copy, not the frozen world.

    A policy is LLM-written code (issue #13); if it could append to the tree it
    observes, a later round — or the next policy over the same world — would
    read outcomes nothing ever executed.
    """
    tree = _load("wide_shallow")
    before = _load("wide_shallow")
    world = ReplaySimulator(tree)

    run = world.replay(GrabbyPolicy())

    assert tree == before
    assert _ids(run.revealed) <= _ids(before)
    assert _ids(world.replay(ExtendDeepest()).revealed) <= _ids(before)


@pytest.mark.parametrize("name", NAMES)
def test_replaying_the_recorded_rollout_reveals_every_node_and_no_more(name: str) -> None:
    """The recording is a fixed point: its own decisions reveal exactly its tree.

    Round for round, the nodes revealed are the ones that round produced — so
    reveal follows each branch in recorded parent-child order and opens the
    root's branches in creation order — and the replay ends having revealed the
    whole tree and nothing beyond it.
    """
    tree = _load(name)
    rounds = _rounds(name)

    run = ReplaySimulator(tree).replay(RecordedRounds(rounds))

    assert [round_.selected for round_ in run.rounds] == [
        tuple(recorded["selected"]) for recorded in rounds
    ]
    assert [round_.revealed for round_ in run.rounds] == [
        tuple(recorded["produced"]) for recorded in rounds
    ]
    assert _ids(run.revealed) == _ids(tree)
    assert run.complete
    assert run.stop_reason == STOP_ALL_REVEALED


def test_a_policy_that_never_stops_is_cut_off_at_the_round_limit() -> None:
    """K₂ bounds a replay whose batches are nonempty but reveal nothing (§3)."""
    tree = _load("narrow_deep")

    run = ReplaySimulator(tree).replay(AlwaysRoot(), max_rounds=3)

    assert run.stop_reason == STOP_MAX_ROUNDS
    assert len(run.rounds) == 3
    # The root's one recorded branch, opened once and never re-opened.
    assert len(run.revealed) == 2


def test_a_policy_that_selects_nothing_ends_the_replay() -> None:
    """The paper's other termination rule, and the one that leaves k* honest."""
    # Only the root's branches, so the replay gives up with nodes still unseen.
    run = ReplaySimulator(_load("failing_branch")).replay(OpenBranches(per_round=1))

    assert run.stop_reason == STOP_EMPTY_BATCH
    assert not run.complete
    # Every round it ran was a nonempty batch, which is what Equation 1 counts.
    assert all(round_.selected for round_ in run.rounds)


def test_replay_imports_no_agent_and_no_evaluator() -> None:
    """Nothing on the replay path can reach a discovery agent or an evaluator.

    Replay retrieves recorded outcomes and generates none, so importing
    ``dream_rsi.replay`` must not pull in an adapter — directly or through a
    module that does.
    """
    completed = subprocess.run(
        [sys.executable, "-c", "import dream_rsi.replay, sys; print('\\n'.join(sys.modules))"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    adapters = sorted(
        name for name in completed.stdout.split() if name.startswith("dream_rsi.adapters")
    )
    assert not adapters, f"importing dream_rsi.replay pulled in {', '.join(adapters)}"
