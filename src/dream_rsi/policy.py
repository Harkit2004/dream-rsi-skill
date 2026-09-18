"""The shared decision interface, and the baselines a dreaming round starts from.

Dream-RSI's exploration policy is **executable Python**, not a prompt and not a
parameter vector: the appendix's improvement prompt tells the
policy-development agent to "implement ``OptimalPolicy.solve(self, question,
budget=None)``" and edit nothing else (§B.2). Code is what makes the policy
something an LLM can rewrite (issue #14), and what makes one version comparable
to another by replaying both over the same recorded worlds (issue #12).

One object drives both phases. §3: "both the online and offline phases use this
same decision interface" — so :class:`OptimalPolicy` exposes:

* :meth:`OptimalPolicy.select`, the per-round decision the online rollout
  (``orchestrator.run_rollout``) and the replay simulator
  (``replay.ReplaySimulator.replay``) both call, with the same three arguments
  in both cases: the tree observed so far, ``A(T)``, and ``W``;
* :meth:`OptimalPolicy.solve`, the paper's own entry point, which drives a
  :class:`Question` — a world handle that reveals what it is asked for — to
  termination by calling ``select`` in a loop.

Neither is told which phase it is in, and there is nothing to branch on: a
policy sees a tree of revealed outcomes and an eligible set either way. That is
the property the dreaming signal rests on, and
``tests/test_policy.py`` asserts the consequence — a policy replaying the tree
its own rollout recorded retraces it decision for decision.

The baselines here are deliberately three different strategies rather than three
spellings of one: :class:`BreadthFirstPolicy` refines ``W`` branches level by
level, :class:`GreedyBestFirstPolicy` deepens the single most promising frontier,
and :class:`BudgetAwarePolicy` mixes the two and stops paying when the returns
stop. Equation 1 ranks them differently (``tests/test_policy.py``), which is what
gives issue #12 something to compare and issue #14 somewhere to start.

Policies run inside the frozen replay world, so this module imports the tree and
nothing else: no agent, no evaluator, no simulator. Its decisions read only the
revealed prefix — "Never use unrevealed scores, a true optimum, hardcoded winning
cell ids" (§B.2) — and none of them samples, so a replay of one world by one
baseline is reproducible whatever seed it is handed (working rule 5).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from dream_rsi.tree import DiscoveryTree, eligible_nodes

__all__ = [
    "DEFAULT_BETA",
    "BreadthFirstPolicy",
    "BudgetAwarePolicy",
    "GreedyBestFirstPolicy",
    "OptimalPolicy",
    "Question",
]

# PAPER-GAP: §B.2 has every policy read "exactly one scalar" in ``__init__`` —
# ``beta = float(self.config.get("beta", <sensible_default>))`` — and describes
# what the ends of its range mean ("high beta means more width, deeper patience,
# and weaker pruning") without giving a scale or a default. We treat it as a
# multiplier on a policy's own thresholds, so 1.0 means "as written" and the
# offline sweep brackets it either side. Revisit if the authors' implementation
# lands (see references/method.md).
DEFAULT_BETA = 1.0


class Question(Protocol):
    """A world a policy can drive: it reveals what it is asked for and logs it.

    ``replay.ReplayRun`` is one — ``ReplaySimulator.start()`` hands back a fresh
    one — and it is what :meth:`OptimalPolicy.solve` is called with in replay.
    Declared structurally, and here rather than imported, so that a policy
    module stays off the simulator's import path and a future online handle can
    satisfy the same shape.

    ``max_parallelism`` (§B.2: ``question.max_parallelism``) and ``seed`` are
    read off the handle when it has them; a handle without either is driven one
    node at a time with an unseeded-but-fixed generator.
    """

    @property
    def revealed(self) -> DiscoveryTree:
        """``T^{m,k}``: the outcomes revealed so far, and nothing else."""
        ...

    @property
    def complete(self) -> bool:
        """Whether everything this world holds has been revealed (``T^{m,k} = T``)."""
        ...

    def reveal(self, batch: Sequence[str]) -> Any:
        """Take one decision round: reveal what ``batch`` retrieves."""
        ...

    def result(self) -> Any:
        """The traversal so far, as a record — §B.2's ``finalize_result``."""
        ...


class OptimalPolicy(ABC):
    """The interface every candidate policy implements (§3, §B.2).

    A subclass implements :meth:`choose`, the strategy: given the revealed tree,
    the eligible nodes that are still worth selecting, and ``W``, return the
    batch to open next, or nothing to stop. Everything else — closing a frontier
    the world has no continuation for, the per-rollout state reset, the
    ``solve`` loop — is here, because all three baselines need it and a
    model-written version should not have to reinvent it to be legal.

    A policy carries state only for the length of one rollout (§3: replay
    "resets the policy's per-rollout state" before each policy-world pair).
    Replay calls :meth:`reset` before the first decision; an online caller
    reusing one instance for a second rollout has to call it too.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self.beta = float(self.config.get("beta", DEFAULT_BETA))
        self.reset()

    def reset(self, rng: random.Random | None = None) -> None:
        """Forget one rollout's state before starting another (§3).

        ``rng`` is the generator replay seeds from the run's seed and hands to a
        policy that declares this method. The baselines here ignore it — §B.2:
        "Never sample randomly" — and keeping the parameter is what lets a
        model-written version that does sample stay reproducible instead of
        reaching for the global stream.
        """
        self._closed: set[str] = set()
        self._selected: dict[str, int] = {}

    def select(
        self, tree: DiscoveryTree, eligible: Sequence[str], width: int
    ) -> Sequence[str]:
        """One decision round, as both drivers call it (§3).

        The eligible set is filtered before :meth:`choose` sees it: a node this
        policy already selected without anything coming back has no continuation
        left, so selecting it again would spend a round to reveal nothing.
        §B.2's skeleton keeps exactly this set (``update_closed(closed, prefix,
        question)``).

        Note what closure here is *not*: a judgement about a branch's quality. A
        branch that scored badly, or failed hard, stays selectable — "Shallow
        weak scores are not enough to discard a branch: deeper attempts can
        recover", and "a later successful result reopens the branch" (§B.2).
        Only the world running out of recorded continuations closes anything,
        which nothing can reopen.
        """
        counts = _child_counts(tree)
        self._close_barren(counts)
        live = tuple(node_id for node_id in eligible if node_id not in self._closed)
        batch = tuple(self.choose(tree, live, width))
        # Counted rather than looked up per node, so a batch naming a node this
        # tree has never held is left for the driver to reject with a message
        # that says so, instead of failing here with a bare ``KeyError``.
        self._selected = {node_id: counts.get(node_id, 0) for node_id in batch}
        return batch

    @abstractmethod
    def choose(
        self, tree: DiscoveryTree, live: Sequence[str], width: int
    ) -> Sequence[str]:
        """The strategy: which of ``live`` to open next, or ``()`` to stop.

        ``live`` is ``A(T)`` — the root first, then the revealed leaves in id
        order — minus the nodes that have already been shown to have no
        continuation. Selecting the root opens a further branch; selecting it
        twice in one batch opens two, which is how a rollout reaches full width
        in one round (see ``orchestrator._check_batch``).

        §B.2's batch rules read against this encoding rather than against the
        paper's grid of cells: "no duplicate ids" forbids probing one cell twice,
        and a batch "may contain several roots", which here is the root selected
        several times, because every branch of ours opens from the one root.
        Likewise a batch holding the root and one of its children is not the
        forbidden "parent and its child together": the root selection opens a
        different branch, and no two selections in a batch ever retrieve the same
        recorded node.
        """

    def solve(self, question: Question, budget: int | None = None) -> Any:
        """Drive ``question`` to termination — the paper's entry point (§B.2).

        The loop is §B.2's skeleton: reset, then each round read the revealed
        prefix, select a batch, stop if it is empty, and reveal it. It also
        stops once the world holds nothing further (§3's ``T^{m,k} = T``), and
        once a policy has gone longer without revealing anything than it has
        frontiers left to close — see the comment on that in the loop.
        "Replay calls with ``budget=None``. Always terminate when no batch is
        selected" puts termination on the policy, and replay's own driver,
        ``ReplaySimulator.replay``, additionally caps the rounds at ``K₂``.

        PAPER-GAP: §B.2 calls ``_budget_done(question, budget)`` without saying
        what the budget counts. We count revealed nodes — Equation 1's ``N``,
        and online the generation-evaluation calls a rollout pays for, since
        every online selection reveals one — and truncate a batch to what is
        left so a round cannot overshoot the cap it was given. Revisit if the
        authors' implementation lands (see references/method.md).
        """
        # A handle that carries no worker count is driven one node at a time,
        # which is what ``ReplaySimulator.replay`` also defaults ``width`` to;
        # one is also the floor, because §3's W is at least 1.
        width = max(1, int(getattr(question, "max_parallelism", 1)))
        self.reset(random.Random(getattr(question, "seed", 0)))

        barren = 0
        size = 0
        while not question.complete:
            tree = question.revealed
            if len(tree) > size:
                size, barren = len(tree), 0
            else:
                barren += 1
            eligible = eligible_nodes(tree)
            # A round that reveals nothing has to close at least one frontier —
            # that is what :meth:`select` does with it — so a policy still
            # getting somewhere cannot go more consecutive rounds without a
            # reveal than there are nodes to close. One that can is selecting
            # what the world has already refused, and no further round will
            # change that. The same reasoning is why ``ReplaySimulator.replay``
            # caps its rounds; ``solve`` has no ``K₂`` to cap with, and
            # unbounded model-written code (issue #13) is not a hang worth
            # inheriting.
            if barren > len(eligible):
                break
            spent = len(tree) - 1
            if budget is not None and spent >= budget:
                break
            # Offered as a narrower width rather than cut afterwards: a batch
            # trimmed after the fact would leave :meth:`select` holding
            # frontiers nothing asked about, and the next round would close them
            # as if the world had refused them.
            room = width if budget is None else min(width, budget - spent)
            batch = tuple(self.select(tree, eligible, room))
            if not batch:
                break
            if budget is not None:
                # A backstop only: a policy may return more than it was offered.
                batch = batch[: budget - spent]
                self._keep_selected(batch)
            question.reveal(batch)
        return question.result()

    def _keep_selected(self, batch: Sequence[str]) -> None:
        """Forget the selections a cut took out of the batch before it was revealed.

        What was cut was never asked about, so the world having no continuation
        for it is not something :meth:`_close_barren` may conclude next round —
        it would close frontiers on evidence that does not exist.
        """
        kept = set(batch)
        self._selected = {
            node_id: children for node_id, children in self._selected.items() if node_id in kept
        }

    def _close_barren(self, counts: Mapping[str, int]) -> None:
        """Close last round's selections that the world had no continuation for.

        Read off the tree rather than off the driver's report, because that is
        all a policy is given in either phase: a selected node that gained no
        child revealed nothing. Online every selection produces a child, so
        nothing closes there — which is the honest difference between a live
        task and a recording that has run out, not a mode the policy branches
        on.
        """
        for node_id, children_before in self._selected.items():
            if counts.get(node_id, 0) == children_before:
                self._closed.add(node_id)
        self._selected = {}


class BreadthFirstPolicy(OptimalPolicy):
    """Opens ``W`` branches and refines them level by level.

    The paper's fixed-exploration shape (§4: "10 parallel workspaces with up to
    11 refinement steps"): spend the whole width on the shallowest frontier, and
    only go back to the root for new branches once the current ones are
    exhausted. It is the policy that reads nothing into the scores it sees,
    which is what makes it the honest floor for issue #12 — it pays for every
    node it opens and cannot be led anywhere by a misleading landscape.
    """

    def choose(
        self, tree: DiscoveryTree, live: Sequence[str], width: int
    ) -> Sequence[str]:
        """The shallowest live leaves, or a full batch of new branches."""
        leaves = [node_id for node_id in live if node_id != tree.root_id]
        if leaves:
            shallowest = min(_depth(tree, node_id) for node_id in leaves)
            return tuple(
                node_id for node_id in leaves if _depth(tree, node_id) == shallowest
            )[:width]
        if tree.root_id in live:
            return (tree.root_id,) * width
        return ()


class GreedyBestFirstPolicy(OptimalPolicy):
    """Refines the single best-anchored frontier, whichever branch it is on.

    Ranks a frontier by its *successful anchor* — "the best historical score
    from such a successful evaluation" on the path that leads to it (§B.2) —
    rather than by its own latest score, so one regression or one implementation
    failure does not throw away the branch that produced the best result so far.
    A frontier it passes over stays live, so the choice is a real one: a branch
    whose refinement regresses loses the next round to a sibling it was beating.

    It spends the width it is offered on opening branches, which is the only
    exploration it does, and refines one node at a time thereafter. That is what
    makes it the counterweight to :class:`BreadthFirstPolicy` under Equation 1,
    which charges ``N`` for what a policy opens and refunds
    ``β₂·N/max{1, k}`` for opening it in parallel: a policy that refines serially
    takes the whole cost and little of the refund, and earns it back only by
    opening fewer, better nodes.
    """

    def choose(
        self, tree: DiscoveryTree, live: Sequence[str], width: int
    ) -> Sequence[str]:
        """The best-anchored live frontier, else a batch of new branches."""
        ranked = _by_anchor(tree, live)
        if ranked:
            return (ranked[0],)
        # PAPER-GAP: §B.2 requires a batch to mix exploitation with "new roots or
        # underexplored branches" without saying how an unopened root ranks
        # against a frontier whose branch has no successful anchor at all. We
        # prefer the root: an unopened branch is untried, while a branch that has
        # only ever failed has evidence against it — and "a local implementation
        # failure does not by itself prove that its parent direction is poor"
        # keeps the failed frontier eligible rather than closing it. Revisit if
        # the authors' implementation lands (see references/method.md).
        if tree.root_id in live:
            # A full batch, so that the next round has several frontiers to rank
            # rather than the one it just opened: best-first with a frontier of
            # one is depth-first wearing a ranking that never fires.
            return (tree.root_id,) * width
        unanchored = [node_id for node_id in live if node_id != tree.root_id]
        return (unanchored[0],) if unanchored else ()


class BudgetAwarePolicy(OptimalPolicy):
    """Mixes exploration with exploitation, and stops when the returns stop.

    §B.2's "dynamic portfolio": each round takes one exploration slot — a new
    branch off the root — and fills the rest of the width with the best-anchored
    frontiers, so it batches instead of going serial and widens instead of
    committing to its first branch. Unlike the other two it also *declines to
    spend*: it stops once it has opened its allowance of nodes, or once
    revealing more has stopped improving the best score it has seen, which is
    what Equation 1's cost term rewards.

    Both thresholds come off ``beta`` through one schedule (§B.2: "Route every
    behavioral threshold through one ``_schedule(beta) -> dict``"), fixed for
    the whole rollout: "Never change beta from observations inside ``solve()``".
    """

    def reset(self, rng: random.Random | None = None) -> None:
        """Also forget what this rollout had found and how long ago it improved."""
        super().reset(rng)
        self._best: float | None = None
        # Unset until the first decision, so a policy handed a world someone
        # else has already revealed part of does not read that as a round of its
        # own that improved nothing.
        self._size: int | None = None
        self._stagnant = 0

    def _schedule(self) -> dict[str, int]:
        """``beta`` as the two numbers this policy actually decides with.

        PAPER-GAP: part of the beta gap above — §B.2 fixes the *direction* of
        each threshold in beta and gives no values. We scale an allowance of 8
        revealed nodes and a patience of 2 unimproved rounds, which is a handful
        of attempts and a short wait against the fixtures in
        ``tests/fixtures/trees/``; the sweep either side of ``beta = 1`` is what
        a reported result shows the sensitivity of. Revisit if the authors'
        implementation lands (see references/method.md).
        """
        return {
            "allowance": max(1, round(8 * self.beta)),
            "patience": max(1, round(2 * self.beta)),
        }

    def choose(
        self, tree: DiscoveryTree, live: Sequence[str], width: int
    ) -> Sequence[str]:
        """One exploration slot plus the best-anchored frontiers, while it is worth it."""
        schedule = self._schedule()
        self._track(tree)

        room = schedule["allowance"] - (len(tree) - 1)
        if room <= 0 or self._stagnant >= schedule["patience"]:
            return ()

        slots = min(width, room)
        batch = [tree.root_id] if tree.root_id in live and slots > 0 else []
        batch.extend(_by_anchor(tree, live)[: slots - len(batch)])
        return tuple(batch)

    def _track(self, tree: DiscoveryTree) -> None:
        """Count the rounds that revealed something without improving on the best.

        A round that revealed nothing is not stagnation: the frontier it
        selected is closed, so the next round decides on a strictly smaller set
        and the policy is not waiting on anything.
        """
        best = _best_score(tree)
        if best is not None and (self._best is None or best > self._best):
            self._best = best
            self._stagnant = 0
        elif self._size is not None and len(tree) > self._size:
            self._stagnant += 1
        self._size = len(tree)


def _child_counts(tree: DiscoveryTree) -> dict[str, int]:
    """How many children each node of ``tree`` has, counted in one pass."""
    counts: dict[str, int] = {}
    for node in tree.iter_nodes():
        if node.parent_id is not None:
            counts[node.parent_id] = counts.get(node.parent_id, 0) + 1
    return counts


def _depth(tree: DiscoveryTree, node_id: str) -> int:
    """How far ``node_id`` sits below the root."""
    depth = 0
    parent = tree.node(node_id).parent_id
    while parent is not None:
        depth += 1
        parent = tree.node(parent).parent_id
    return depth


def _anchor(tree: DiscoveryTree, node_id: str) -> float | None:
    """The best score on the root → ``node_id`` path: the branch's successful anchor.

    ``None`` where nothing on the path was scored — a branch whose every attempt
    failed hard, which is not the same as a branch that scored badly.
    """
    best: float | None = None
    current: str | None = node_id
    while current is not None:
        node = tree.node(current)
        if node.score is not None and (best is None or node.score > best):
            best = node.score
        current = node.parent_id
    return best


def _by_anchor(tree: DiscoveryTree, live: Sequence[str]) -> tuple[str, ...]:
    """The live frontiers that have an anchor, best first, ties by lowest id.

    The root is not one: it is where a new branch is opened, not a frontier with
    a history. Ties break on the id, which is creation order (see ``tree.py``),
    so a batch is a function of the revealed prefix alone and a replay of it is
    reproducible (working rule 5).
    """
    anchored = [
        (anchor, node_id)
        for node_id in live
        if node_id != tree.root_id and (anchor := _anchor(tree, node_id)) is not None
    ]
    return tuple(node_id for _, node_id in sorted(anchored, key=lambda pair: (-pair[0], pair[1])))


def _best_score(tree: DiscoveryTree) -> float | None:
    """``max_v s_v`` over the revealed tree, or ``None`` if nothing scored."""
    scores = [node.score for node in tree.iter_nodes() if node.score is not None]
    return max(scores) if scores else None
