"""The frozen replay world: a recorded discovery tree a new policy can navigate.

A completed rollout's tree becomes a replay simulator (Dream-RSI §2, §3). An
alternative exploration policy traverses it, revealing different subsets of the
recorded branches in a different order and with different batching, and every
outcome it sees was executed once, online, and stored. Replay "returns recorded
children of the selected nodes deterministically rather than generating new
candidates" (§3): nothing here calls a discovery agent or an evaluator, and this
module imports neither — ``tests/test_replay.py`` asserts that.

Reveal is strictly prefix-observable. The policy observes ``T^{m,k} ⊆ T``, the
subtree revealed after ``k`` rounds, starting from ``{r}``, and each round it
selects a batch from ``A(T^{m,k})`` — the root, plus the leaves of what it has
already revealed. Selecting a node returns:

* for the root, the earliest-created recorded child not yet revealed, opening
  one more branch;
* for any other node, its recorded continuation;
* the empty set when the recording holds no continuation there.

That empty set is the point. A policy that walks off the recorded tree gets
nothing back, which is the honest signal that the branch was never explored;
there is no nearest-neighbour fallback and no interpolation, because either
would score a replay on outcomes that never happened.

A replay ends when the policy selects an empty batch, when the round limit
``K₂`` is reached, or when every recorded node has been revealed (§3). The
trajectory it leaves behind — the rounds, what each selected and what each
revealed — is what Equation 1 is computed over (issue #8) and what the
trajectory log records (issue #9).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dream_rsi.tree import DiscoveryTree, Node, eligible_nodes

__all__ = [
    "STOP_ALL_REVEALED",
    "STOP_EMPTY_BATCH",
    "STOP_MAX_ROUNDS",
    "ReplayPolicy",
    "ReplayRound",
    "ReplayRun",
    "ReplaySimulator",
]

# Why a replay stopped — the paper's three termination rules for the offline
# phase (§3: "terminates when the policy selects C = ∅, the round limit k = K₂
# is reached, or T^{m,k} = T_i").
STOP_EMPTY_BATCH = "empty_batch"
STOP_MAX_ROUNDS = "max_rounds"
STOP_ALL_REVEALED = "all_revealed"


class ReplayPolicy(Protocol):
    """The shared decision interface, as replay calls it (§3).

    Structurally identical to :class:`~dream_rsi.orchestrator.ExplorationPolicy`
    — "both the online and offline phases use this same decision interface", so
    a policy written for one drives the other — but declared here rather than
    imported, because the orchestrator reaches an agent and an evaluator and
    nothing on the replay path may.
    """

    def select(
        self, tree: DiscoveryTree, eligible: Sequence[str], width: int
    ) -> Sequence[str]:
        """Choose the batch to reveal next, or nothing to end the replay.

        ``tree`` is the revealed subtree ``T^{m,k}``, carrying the stored
        observations and scores of everything revealed so far and nothing else.
        ``eligible`` is ``A(T^{m,k})``, the root followed by the revealed
        leaves; ``width`` is ``W``. Selecting a node twice in one batch opens
        two continuations from it, as online.
        """
        ...


@dataclass(frozen=True)
class ReplayRound:
    """One completed replay decision round: what was selected, what that revealed.

    ``selected`` and ``revealed`` align pairwise: ``revealed[i]`` is the node id
    that ``selected[i]`` revealed, or ``None`` where the recording held no
    continuation there. Only nonempty batches become rounds (§3: "each nonempty
    batch counts as one round"), so the number of these is Equation 1's
    ``k^{m,★}``.
    """

    index: int
    selected: tuple[str, ...]
    revealed: tuple[str | None, ...]


class ReplaySimulator:
    """A frozen replay world built from one recorded discovery tree (§2, §3).

    The world is immutable and reusable: it takes its own copy of the node table
    it is given, hands a policy a fresh tree of the revealed nodes rather than
    anything it holds, and starts every replay over it again from ``{r}``. So
    the candidate policy versions of one dreaming round (issue #12) can share a
    world without seeing each other's traversals, and a policy — which is
    model-written code (issue #13) — cannot write an outcome into the record
    that nothing ever executed.
    """

    def __init__(self, tree: DiscoveryTree) -> None:
        self._nodes: dict[str, Node] = {node.id: node for node in tree.iter_nodes()}
        self._root_id = tree.root_id
        children: dict[str, list[str]] = {}
        # Ascending id order is creation order (see tree.py), so "the
        # earliest-created child" is this list's first unrevealed entry.
        for node in tree.iter_nodes():
            if node.parent_id is not None:
                children.setdefault(node.parent_id, []).append(node.id)
        self._children: dict[str, tuple[str, ...]] = {
            parent_id: tuple(ids) for parent_id, ids in children.items()
        }

    @property
    def root_id(self) -> str:
        return self._root_id

    def __len__(self) -> int:
        """How many nodes the recording holds, the root included."""
        return len(self._nodes)

    def _node(self, node_id: str) -> Node:
        """The recorded node, which is a frozen dataclass and safe to hand out."""
        return self._nodes[node_id]

    def _recorded_children(self, node_id: str) -> tuple[str, ...]:
        """This node's recorded children, earliest-created first."""
        return self._children.get(node_id, ())

    def start(self) -> ReplayRun:
        """Begin a replay at ``T^{m,0} = {r}``, to be stepped by hand."""
        return ReplayRun(self)

    def replay(
        self,
        policy: ReplayPolicy,
        *,
        width: int = 1,
        max_rounds: int | None = None,
    ) -> ReplayRun:
        """Run ``policy`` over this world until one of the termination rules fires.

        ``width`` is ``W``, the parallel worker count offered to the policy. As
        online, it is not enforced as a cap: the orchestrator already takes a
        batch as a sequence that may repeat a node and may exceed ``W`` (see
        ``orchestrator._check_batch``), and replay has to accept the batches a
        recorded rollout actually issued.

        ``max_rounds`` is ``K₂``.

        PAPER-GAP: §3 caps a replay at ``K₂`` decision rounds without saying
        what ``K₂`` is. The cap is load-bearing rather than cosmetic — a
        nonempty batch that reveals nothing still counts as a round, so a policy
        that keeps selecting an exhausted leaf would otherwise never terminate.
        We default it to the number of recorded attempts: the smallest cap that
        still lets any policy reveal the whole tree even at one node per round,
        so the default never truncates a replay that is making progress.
        Revisit if the authors' implementation lands (see references/method.md).
        """
        run = ReplayRun(self)
        rounds_left = len(self._nodes) - 1 if max_rounds is None else max_rounds
        reason = STOP_MAX_ROUNDS
        while True:
            if run.complete:
                reason = STOP_ALL_REVEALED
                break
            if rounds_left <= 0:
                break
            revealed = run.revealed
            batch = tuple(policy.select(revealed, eligible_nodes(revealed), width))
            if not batch:
                reason = STOP_EMPTY_BATCH
                break
            run.reveal(batch)
            rounds_left -= 1
        run._stop(reason)
        return run


class ReplayRun:
    """One policy's traversal of a frozen world: the revealed subtree and its rounds.

    Stepping it by hand — :meth:`reveal` one batch at a time — and driving it
    with :meth:`ReplaySimulator.replay` see the same state; the trajectory the
    run accumulates is the same either way.
    """

    def __init__(self, world: ReplaySimulator) -> None:
        self._world = world
        self._revealed: list[str] = [world.root_id]
        self._rounds: list[ReplayRound] = []
        self._stop_reason: str | None = None

    @property
    def revealed(self) -> DiscoveryTree:
        """``T^{m,k}``: a fresh tree of what has been revealed, safe to hand out."""
        return DiscoveryTree(self._world._node(node_id) for node_id in self._revealed)

    @property
    def eligible(self) -> tuple[str, ...]:
        """``A(T^{m,k})``: the root, then the revealed leaves."""
        return eligible_nodes(self.revealed)

    @property
    def rounds(self) -> tuple[ReplayRound, ...]:
        return tuple(self._rounds)

    @property
    def complete(self) -> bool:
        """Whether ``T^{m,k} = T``, the paper's third termination rule."""
        return len(self._revealed) == len(self._world)

    @property
    def stop_reason(self) -> str | None:
        """Why this replay ended, or ``None`` while it is still running."""
        return self._stop_reason

    def reveal(self, batch: Sequence[str]) -> ReplayRound:
        """Take one decision round: reveal what ``batch`` retrieves from the world.

        The batch is taken in order, so two selections of the same node open two
        continuations. An empty batch is the policy's stop action and is not a
        round — passing one here is rejected rather than logged, because an
        empty round would inflate Equation 1's ``k^{m,★}`` with a decision that
        revealed nothing.
        """
        batch = tuple(batch)
        if not batch:
            raise ValueError("an empty batch ends a replay; it is not a round")
        self._check(batch)

        revealed: list[str | None] = []
        for node_id in batch:
            child = self._next_child(node_id)
            if child is not None:
                self._revealed.append(child)
            revealed.append(child)

        round_ = ReplayRound(index=len(self._rounds), selected=batch, revealed=tuple(revealed))
        self._rounds.append(round_)
        return round_

    def _check(self, batch: Sequence[str]) -> None:
        """Reject anything outside ``A(T^{m,k})`` — the prefix-observability rule."""
        allowed = set(self.eligible)
        outside = [node_id for node_id in batch if node_id not in allowed]
        if outside:
            raise ValueError(
                f"policy selected node(s) outside A(T): {', '.join(sorted(set(outside)))}; "
                f"eligible are {', '.join(self.eligible)}"
            )

    def _next_child(self, node_id: str) -> str | None:
        """``Child(v; T, T^{m,k})``: the continuation this selection retrieves (§3).

        PAPER-GAP: §3 gives the root the earliest-created unrevealed child and
        every other node "its unique recorded child", because online each
        selected node yields one attempt. Our recorded trees can hold more than
        one child anywhere, since a batch may select the same leaf twice (see
        ``orchestrator._check_batch``). We apply the root's rule everywhere:
        earliest-created child not yet revealed. It is the only reading that
        generalises without contradicting the paper — the two rules coincide
        wherever a node has at most one child — and it keeps a recorded rollout
        replayable by its own decisions. Note a consequence: a node's later
        children become unreachable once it stops being a leaf, so a world with
        a branching non-root node can never be fully revealed. Revisit if the
        authors' implementation lands (see references/method.md).
        """
        revealed = set(self._revealed)
        for child in self._world._recorded_children(node_id):
            if child not in revealed:
                return child
        return None

    def _stop(self, reason: str) -> None:
        self._stop_reason = reason
