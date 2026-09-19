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

Alongside them are the four observation signals §B.1 offers a policy from
``see.policy.observation_signal`` — :func:`branch_promising`,
:func:`branch_failed_hard`, :func:`probe_improved_vs_parent` and
:func:`probe_improved_vs_baseline`. They are the vocabulary a policy states its
reasons in, and the vocabulary the model rewriting it (issue #14) is pointed at,
so each reads the revealed tree and refuses a node outside it.
:class:`GreedyBestFirstPolicy` decides in them.

Policies run inside the frozen replay world, so this module imports the tree and
nothing else: no agent, no evaluator, no simulator. Its decisions read only the
revealed prefix — "Never use unrevealed scores, a true optimum, hardcoded winning
cell ids" (§B.2) — and none of them samples, so a replay of one world by one
baseline is reproducible whatever seed it is handed (working rule 5).
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dream_rsi.tree import DiscoveryTree, Node, eligible_nodes

__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_FAILURE_STREAK",
    "DEFAULT_MARGIN",
    "DEFAULT_PATIENCE",
    "BreadthFirstPolicy",
    "BudgetAwarePolicy",
    "GreedyBestFirstPolicy",
    "GridPlan",
    "GridPlanningContext",
    "OptimalPolicy",
    "Question",
    "branch_failed_hard",
    "branch_promising",
    "probe_improved_vs_baseline",
    "probe_improved_vs_parent",
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


@dataclass(frozen=True)
class GridPlan:
    """How wide and how deep the next rollout's grid may be (§B.2, issue #21).

    ``GridPlan(branch_count=W, refine_count=R)`` "creates branches ``0..W-1`` and
    attempts ``0..R``; ``R`` is the number of refinements allowed after each
    root" — so a branch is at most ``R + 1`` attempts deep, the first of them
    being the one that opened it. Both "accept arbitrary integers, not a fixed
    set of presets"; it is the runner that validates them against the caps in
    the :class:`GridPlanningContext` it planned from
    (``orchestrator.run_rollout``), and that enforces the grid as "the hard
    bound: controller thresholds may use less, but can never create branches or
    attempts beyond the effective plan".

    ``reason`` is §B.2's "short, factual ``reason`` in every plan": the evidence
    the width-versus-depth choice was made on, so a run that widened can say why.
    """

    branch_count: int
    refine_count: int
    reason: str


@dataclass(frozen=True)
class GridPlanningContext:
    """The prefix-safe facts a grid is planned from (§B.2, issue #21).

    ``hard_max_branch_count`` and ``hard_max_refine_count`` are the caps the
    runner validates a plan against, and ``max_workers`` is ``W``, the worker cap
    the grid will actually be explored under.

    PAPER-GAP: §B.2's context also carries ``history`` — "completed earlier live
    manifests, including prior planned/effective grids, actual opened
    width/depth, probe work, decision rounds, scores, and beta" — and replay's
    structural support fields. Neither is here: the driver
    (``dream_rsi.run``) keeps its cycle records in its own directory and does not
    yet summarise them for a policy, and replay creates no grid to plan, so a
    plan made here is made from the caps alone. A policy therefore sees the
    "history is empty or insufficient" case §B.2 describes and answers it with
    "an explicit conservative bootstrap plan derived from the context's
    fallback/hard-cap fields". Revisit when the driver feeds its cycles in, and
    if the authors' implementation lands (see references/method.md).
    """

    hard_max_branch_count: int
    hard_max_refine_count: int
    max_workers: int


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
    Replay calls :meth:`reset` before the first decision, and :meth:`select`
    resets itself when what it is shown is a rollout starting, so one instance
    can drive a second rollout either way. A caller holding a seed still calls
    :meth:`reset` with it, because a self-reset has no seed to pass on — which
    is why the online driver should own this too (issue #36).

    One method a subclass may add is deliberately absent here:
    ``plan_grid(self, context: GridPlanningContext) -> GridPlan``, §B.2's
    cross-cycle width-and-depth decision, which ``orchestrator.run_rollout``
    calls once before it opens the grid. §B.2 has it inherited from a template
    and overridden — "do not inherit the template stub" — while here it is
    optional (issue #21), so it is not defined at all: a stub returning nothing
    is exactly the thing the runner cannot tell from a policy that plans, and a
    base class answering for every subclass would put the three baselines below
    on a grid none of them chose. A policy that plans one defines it; a policy
    that does not is run ungridded, and neither is told which it is.
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
        if len(tree) == 1 and (self._selected or self._closed):
            # A tree holding only the root is a rollout starting (§3's
            # ``T^{m,0} = {r}``), so anything still held belongs to one that has
            # finished. Replay resets us itself; the online driver does not
            # (issue #36), and carrying a closure across would read the root as
            # a frontier the world refused, filter out the only legal action,
            # and end that rollout without a single attempt.
            self.reset()
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
        """The best-anchored live frontier that is still promising, else new branches."""
        ranked = _by_anchor(tree, live)
        if ranked:
            # §B.2 ranks a frontier on its whole trajectory — "score trend,
            # regressions, failure/repair sequence" — and not on the anchor
            # alone, so a branch that has stopped gaining loses the round to one
            # that has not. Deprioritised, never closed: it stays live and takes
            # the round back as soon as its rivals stall too, which is what
            # keeps a stalled branch from being written off permanently. Beta
            # sets how long it is given, since "high beta means ... deeper
            # patience, and weaker pruning" (§B.2).
            patience = max(1, round(DEFAULT_PATIENCE * self.beta))
            promising = tuple(
                node_id for node_id in ranked if branch_promising(tree, node_id, patience=patience)
            )
            return ((promising or ranked)[0],)
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


# --------------------------------------------------------------------------
# Observation signals (§B.1's ``see.policy.observation_signal``, issue #11)
#
# The four names the paper's exploration prompt offers a policy: two about a
# branch's trajectory, two about one probe's gain. They are what the policy —
# and the model rewriting it (issue #14) — reasons in, so they read the revealed
# tree ``T^{m,k}`` and nothing else. A signal asked about a node the policy has
# not been shown refuses rather than answering: §B.2's prefix-only rule bars
# "unrevealed scores", and a signal that quietly reached past the prefix would
# inflate a dreamed score with attainment the replay never paid ``N`` for.
#
# PAPER-GAP: §B.1 names all four and defines none of them — no formula, and no
# threshold for what counts as "promising" or as "failed hard". We read them off
# the branch trajectory §B.2 requires a policy to reconstruct (successful
# anchor, score trend, failure/repair sequence), and every threshold is a
# parameter with a default stated on the function that takes it. The defaults
# are deliberately counts of attempts rather than score levels, because §B.2
# forbids "absolute score targets" and a task's scores have no fixed scale.
# Revisit if the authors' implementation lands (see references/method.md).
# --------------------------------------------------------------------------

# How much better than what it is compared against a probe must measure before
# it counts as an improvement. Zero is "strictly better at all".
DEFAULT_MARGIN = 0.0

# How many of a branch's most recent attempts are weighed for a gain before it
# stops being promising. Two, so that one regression or one repairable failure
# at the tip does not erase the branch's anchor (§B.2).
DEFAULT_PATIENCE = 2

# How long a run of attempts that measured nothing has to be before the branch
# reads as hard-failed. Two, because §B.2 is explicit that one such error is not
# enough: "Do not infer algorithmic failure from one such error."
DEFAULT_FAILURE_STREAK = 2


def branch_promising(
    tree: DiscoveryTree, node_id: str, *, patience: int = DEFAULT_PATIENCE
) -> bool:
    """Is the branch ending at ``node_id`` still worth refining? (§B.2.)

    True when at least one of the branch's last ``patience`` attempts gained on
    the attempt it resumed from. That carries the *successful anchor* §B.1 asks
    for as well — an attempt that gained necessarily measured something, so a
    branch on which nothing ever evaluated is never promising.

    The two halves are the two things §B.2 warns against conflating. A
    repairable latest failure "must not erase its historical successful anchor",
    so a single regression or failure at the tip leaves the branch promising;
    but a branch that is "repeatedly unpromising after sufficient valid
    evidence" stops being so, and ``patience`` is how much evidence is enough.

    The root is not a branch and is never promising: nothing has been attempted
    down it yet, which is a reason to open it rather than a trajectory to read.
    """
    _window("patience", patience)
    attempts = _attempts(tree, node_id)
    return any(probe_improved_vs_parent(tree, node.id) for node in attempts[-patience:])


def branch_failed_hard(
    tree: DiscoveryTree, node_id: str, *, streak: int = DEFAULT_FAILURE_STREAK
) -> bool:
    """Does the branch ending at ``node_id`` look hard-unrecoverable? (§B.2.)

    True when its last ``streak`` attempts each produced no measurement at all —
    ``s_v`` unset, an evaluation that returned no score rather than a bad one.

    It reads the *current failure episode* only, at the tip: "a later successful
    result reopens the branch and cancels closure based only on an earlier
    failure". And it takes a run rather than a single attempt, because §B.2 will
    not have algorithmic failure inferred from one error. Like every signal here
    it stays a signal — §B.2: "``n_valid == 0`` and ``branch_failed_hard(obs)``
    are signals, not unconditional closure" — so a policy weighs it against the
    branch's anchor and remaining depth rather than closing on it.
    """
    _window("streak", streak)
    attempts = _attempts(tree, node_id)
    if len(attempts) < streak:
        return False
    return all(node.score is None for node in attempts[-streak:])


def probe_improved_vs_parent(
    tree: DiscoveryTree, node_id: str, *, margin: float = DEFAULT_MARGIN
) -> bool:
    """Did this attempt gain on the one it resumed from? (§B.1's ``delta_vs_parent``.)

    True when ``node_id`` carries a score exceeding its parent's by more than
    ``margin``. Both are canonical ``s_v``, larger-is-better (§3), so a
    lower-is-better task compares correctly here without a special case — the
    direction was applied once, where the node was recorded (see
    ``adapters.evaluator.ScoreDirection``).

    An attempt with no measurement gained on nothing, whatever its parent did.

    PAPER-GAP: part of the gap above — the paper gives no ``delta_vs_parent``
    for a probe whose parent has no score, which is every branch's first attempt
    (the root is the initial workspace and carries no ``s_v``) and every repair
    of a failed one. We count measuring where the parent could not as a gain:
    it is the "prior repair outcome" §B.2 ranks on, and the alternative would
    report a branch's opening attempt as an improvement on nothing. The root
    itself is not a probe and has no parent to gain on. Revisit if the authors'
    implementation lands (see references/method.md).
    """
    _margin(margin)
    node = _revealed(tree, node_id)
    if node.parent_id is None:
        return False
    return _gained(node.score, _revealed(tree, node.parent_id).score, margin)


def probe_improved_vs_baseline(
    tree: DiscoveryTree,
    node_id: str,
    baseline: float | None,
    *,
    margin: float = DEFAULT_MARGIN,
) -> bool:
    """Did this attempt beat the task's reference score? (§B.1's ``delta_vs_baseline``.)

    ``baseline`` is the task's own reference — ``question.baseline_score``, which
    this codebase states on the evaluator as
    ``adapters.evaluator.TaskEvaluator.baseline_score`` (issue #2) — **in
    canonical ``s_v`` units**, so a caller on a lower-is-better task converts it
    with ``ScoreDirection.to_canonical`` first, exactly as the recorded node
    scores it is compared against were converted. Passing it in rather than
    reading it off a world is what keeps this module clear of the evaluator
    adapter, which nothing on the replay path may import.

    PAPER-GAP: part of the gap above — the paper's ``baseline_score`` is always
    present, while a task here may state none (``None``). A task that names no
    reference gives a probe nothing to fall short of, so any measurement
    improves on it; a probe with no measurement improves on nothing either way.
    The root is the initial workspace rather than an attempt, so it is not a
    probe here any more than it is in :func:`probe_improved_vs_parent`.
    Revisit if the authors' implementation lands (see references/method.md).
    """
    _margin(margin)
    node = _revealed(tree, node_id)
    if node.parent_id is None:
        return False
    return _gained(node.score, baseline, margin)


def _gained(score: float | None, reference: float | None, margin: float) -> bool:
    """Is ``score`` a measurement that beats ``reference`` by more than ``margin``?"""
    if score is None:
        return False
    if reference is None:
        return True
    return score > reference + margin


def _revealed(tree: DiscoveryTree, node_id: str) -> Node:
    """``node_id`` as the policy has been shown it, or a refusal naming it.

    A signal reads ``T^{m,k}`` and nothing else. Answering ``False`` for a node
    outside it would be indistinguishable from a real answer, which is how a
    policy would come to reason about a branch it had not paid to reveal.
    """
    try:
        return tree.node(node_id)
    except KeyError:
        raise ValueError(
            f"node {node_id!r} is not in the revealed tree: a signal reads the "
            f"revealed prefix only, never a node the policy has not been shown"
        ) from None


def _attempts(tree: DiscoveryTree, node_id: str) -> tuple[Node, ...]:
    """The branch's attempts, root first, the root itself excluded.

    §B.2's "ordered prefix trajectory" for one frontier: the recorded attempts
    that lead to it, oldest first, so the tip is last.
    """
    path: list[Node] = []
    current: str | None = node_id
    while current is not None:
        node = _revealed(tree, current)
        if node.parent_id is not None:
            path.append(node)
        current = node.parent_id
    path.reverse()
    return tuple(path)


def _window(name: str, value: int) -> None:
    """Reject a window that would weigh something other than what it names.

    ``attempts[-0:]`` is the whole branch rather than none of it, so a window of
    zero would silently give a signal every attempt a policy ever made on that
    branch instead of the none it asked for.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a count of attempts >= 1, got {value!r}")


def _margin(value: float) -> None:
    """Reject a margin that would invert what "improved" means."""
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not numeric or not math.isfinite(value) or value < 0:
        raise ValueError(f"margin must be a finite number >= 0, got {value!r}")


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
