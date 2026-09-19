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
trajectory it leaves behind — the rounds, what each selected, what each
revealed and the observations those reveals exposed — is what Equation 1 is
computed over (issue #8) and what :class:`SimResult` serialises so a finished
dreaming run can be inspected afterwards (issue #9).

Replay is reproducible (working rule 5): the same world, policy and seed
produce byte-identical trajectories. Nothing here reads the clock or iterates a
set, and the one place randomness can enter — the policy's own decisions —
takes its generator from the seed the replay was started with.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dream_rsi.tree import DiscoveryTree, Node, eligible_nodes

__all__ = [
    "DEFAULT_SEED",
    "SIM_RESULT_SCHEMA_VERSION",
    "STOP_ALL_REVEALED",
    "STOP_EMPTY_BATCH",
    "STOP_MAX_ROUNDS",
    "CurvePoint",
    "ReplayPolicy",
    "ReplayRound",
    "ReplayRun",
    "ReplaySimulator",
    "SimResult",
    "TrajectoryError",
]

# Bumped whenever a stored trajectory's layout changes in a way an older reader
# would misread, exactly as ``tree.SCHEMA_VERSION`` is; the two are versioned
# independently, because a trajectory and the tree it was taken over change for
# different reasons.
SIM_RESULT_SCHEMA_VERSION = 1

# PAPER-GAP: §3 says replay "resets the policy's per-rollout state" before each
# policy-world pair but never says how a policy's randomness is seeded, or
# whether the paper's policies use any. Working rule 5 requires that a replay
# be reproducible from stated inputs, so we pass every replay an explicit seed
# and fix a default rather than leave one run in a million unreproducible: a
# caller who never thinks about seeding still gets the same trajectory twice.
# Sweeping a policy over several seeds is then the caller's loop, and the seed
# that produced a trajectory is recorded in its ``SimResult``. Revisit if the
# authors' implementation lands (see references/method.md).
DEFAULT_SEED = 0

# Why a replay stopped — the paper's three termination rules for the offline
# phase (§3: "terminates when the policy selects C = ∅, the round limit k = K₂
# is reached, or T^{m,k} = T_i").
STOP_EMPTY_BATCH = "empty_batch"
STOP_MAX_ROUNDS = "max_rounds"
STOP_ALL_REVEALED = "all_revealed"


class TrajectoryError(ValueError):
    """A trajectory cannot be read, or cannot hold what a replay is putting in it.

    One catchable type for "this log does not work", as ``tree.TreeError`` is
    for a tree, so a dreaming run that walks an archive can tell an unreadable
    log from a bug in its own inspection code. Both ends of the trajectory
    raise it, and both are held to the same contract: a replay that wrote a
    score its own loader refuses would produce a log that cannot be read back,
    which is worthless exactly when someone comes to inspect it.
    """


def _object(what: str, value: Any) -> dict[str, Any]:
    """Reject a stored trajectory that is not shaped like one.

    A log read back off disk is decoded, not trusted: an archived dreaming run
    that decodes into the wrong shape would be re-scored as if it had been
    replayed, which is worse than failing to load.
    """
    if not isinstance(value, dict):
        raise TrajectoryError(f"{what} must be an object, got {type(value).__name__}")
    return value


def _sequence(what: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise TrajectoryError(f"{what} must be a list, got {type(value).__name__}")
    return value


def _integer(what: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrajectoryError(f"{what} must be an integer, got {value!r}")
    return value


def _string(what: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TrajectoryError(f"{what} must be a string, got {value!r}")
    return value


def _number(what: str, value: Any) -> float | None:
    """A score, or nothing — and never a NaN or an infinity.

    ``json.loads`` decodes the bare ``NaN`` and ``Infinity`` tokens Python
    writes, so a stored trajectory can carry either. Neither is a score a
    replay could have produced: a NaN stays the running best forever, because
    every comparison against it is false, and ``scoring.replay_score`` rejects
    an infinite attainment outright.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrajectoryError(f"{what} must be a number or null, got {value!r}")
    if not math.isfinite(value):
        raise TrajectoryError(f"{what} must be a finite number, got {value!r}")
    return value


def _score(node: Node) -> float | None:
    """``s_v`` as a trajectory can hold it, or a refusal naming the node.

    The same check the loader applies, at the other end: ``Node`` takes any
    ``int`` or ``float`` as a score, which admits a bool and a NaN, and a
    trajectory built from one could not be read back by :meth:`SimResult.from_dict`.
    A recording that cannot be logged fails here, while the node that carries
    the bad score can still be named.
    """
    return _number(f"score of node {node.id}", node.score)


def _observations(node: Node) -> tuple[str, ...]:
    """What a reveal exposes, as a trajectory can hold it (§3).

    ``Node`` checks that ``observations`` is a list and nothing about what is
    in it, while a round's observations are read back as strings. Held to the
    loader's contract here for the same reason as :func:`_score`.
    """
    return tuple(_string(f"observation of node {node.id}", text) for text in node.observations)


def _detached(node: Node) -> Node:
    """A copy of ``node`` sharing nothing writable with it.

    ``Node`` is frozen, but ``diagnostics`` is a plain mapping and its values
    may nest, so holding the same node is enough for a policy to write into the
    recording — the one thing a frozen world must make impossible. The
    round-trip through the node's own serialisation copies the mutable parts,
    because ``to_dict`` goes through ``dataclasses.asdict``.
    """
    return Node.from_dict(node.to_dict())


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
        leaves; ``width`` is ``W``. A batch is retrieved in order, so repeating
        a node walks the unrevealed children the recording holds under it: two
        selections of the root open two branches, as online. Repeating any
        other node is a batch the online rollout refuses outright (see
        ``orchestrator._check_batch``), so a policy should not issue one; over
        a tree this repo recorded it gains nothing anyway, since a non-root
        node there has at most one recorded child.
        """
        ...

    # Optional: a policy that carries per-rollout state, randomness above all,
    # also defines
    #
    #     def reset(self, rng: random.Random) -> None: ...
    #
    # which replay calls once before the first decision with a generator seeded
    # from the run's seed (§3: "replay resets the policy's per-rollout state").
    # It is not declared on this Protocol because most policies are stateless
    # and requiring the method would stop them matching it. A policy that draws
    # its randomness from anywhere else — the global ``random`` module, the
    # clock, ``id()`` — breaks working rule 5, and
    # ``tests/test_determinism.py`` is what catches that.


@dataclass(frozen=True)
class ReplayRound:
    """One completed replay decision round: what was selected, what that revealed.

    ``selected``, ``revealed`` and ``observations`` align pairwise:
    ``revealed[i]`` is the node id that ``selected[i]`` revealed, or ``None``
    where the recording held no continuation there, and ``observations[i]`` is
    that node's stored observations — §3: "the newly revealed nodes expose
    their stored observations before the policy makes its next decision" — or
    empty where nothing was revealed. Only nonempty batches become rounds (§3:
    "each nonempty batch counts as one round"), so the number of these is
    Equation 1's ``k^{m,★}``.
    """

    index: int
    selected: tuple[str, ...]
    revealed: tuple[str | None, ...]
    observations: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "selected": list(self.selected),
            "revealed": list(self.revealed),
            "observations": [list(group) for group in self.observations],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ReplayRound:
        payload = _object("round", payload)
        round_ = cls(
            index=_integer("round index", payload.get("index")),
            selected=tuple(
                _string("round selected", raw)
                for raw in _sequence("round selected", payload.get("selected"))
            ),
            revealed=tuple(
                None if raw is None else _string("round revealed", raw)
                for raw in _sequence("round revealed", payload.get("revealed"))
            ),
            observations=tuple(
                tuple(
                    _string("round observation", raw)
                    for raw in _sequence("round observations", group)
                )
                for group in _sequence("round observations", payload.get("observations"))
            ),
        )
        widths = {len(round_.selected), len(round_.revealed), len(round_.observations)}
        if len(widths) != 1:
            # The three columns are read by position everywhere. A stored round
            # whose columns differ in length silently drops the odd one out of
            # every zip that walks it, and describes a decision round that cost
            # less than the one that ran.
            raise TrajectoryError(
                f"round {round_.index} does not line up: {len(round_.selected)} selected, "
                f"{len(round_.revealed)} revealed, {len(round_.observations)} observations"
            )
        return round_


@dataclass(frozen=True)
class CurvePoint:
    """One reveal, and the best score the replay had found once it happened.

    The appendix's policy skeleton records a curve point on every revealed node
    (``probe_batch(..., on_reveal=lambda _: _record_curve(res, question))``), so
    a replay reports discovery quality against cumulative attempts rather than
    one final number — the shape the paper's discovery trajectories are plotted
    in (§4, Figure 4).

    ``revealed`` is the running ``N``, the count of revealed non-root nodes
    including this one; ``score`` is this node's ``s_v``, ``None`` where its
    evaluation failed; ``best_score`` is the running ``max_v s_v``, ``None``
    until something scores.
    """

    round_index: int
    node_id: str
    revealed: int
    score: float | None
    best_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "node_id": self.node_id,
            "revealed": self.revealed,
            "score": self.score,
            "best_score": self.best_score,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> CurvePoint:
        payload = _object("curve point", payload)
        return cls(
            round_index=_integer("curve point round_index", payload.get("round_index")),
            node_id=_string("curve point node_id", payload.get("node_id")),
            revealed=_integer("curve point revealed", payload.get("revealed")),
            score=_number("curve point score", payload.get("score")),
            best_score=_number("curve point best_score", payload.get("best_score")),
        )


@dataclass(frozen=True)
class SimResult:
    """The serialisable record of one replay: what it did, what it saw, what it found.

    The paper's replay infrastructure keeps a ``SimResult`` per policy-world
    pair, and the policy-development agent "examines the replay trajectories
    and scores" of a version to produce the next one (§3). So this has to
    survive being written to disk: a dreaming round (issue #12) evaluates ``M``
    versions over every tree in the history, and the logs are what the run is
    inspected from afterwards.

    :meth:`to_json` is the canonical form. It is byte-identical across runs of
    the same world, policy and seed, which is the guarantee working rule 5
    states and ``tests/test_determinism.py`` asserts.
    """

    seed: int
    world_size: int
    stop_reason: str | None
    attainment: float | None
    rounds: tuple[ReplayRound, ...]
    curve: tuple[CurvePoint, ...]

    @property
    def revealed(self) -> int:
        """``N_i^m``: revealed non-root nodes, Equation 1's execution cost term."""
        return len(self.curve)

    @property
    def round_count(self) -> int:
        """``k_i^{m,★}``: completed rounds at termination."""
        return len(self.rounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SIM_RESULT_SCHEMA_VERSION,
            "seed": self.seed,
            "world_size": self.world_size,
            "stop_reason": self.stop_reason,
            "attainment": self.attainment,
            "rounds": [round_.to_dict() for round_ in self.rounds],
            "curve": [point.to_dict() for point in self.curve],
        }

    def to_json(self) -> str:
        """The canonical serialisation: sorted keys, so the bytes are the record.

        ``allow_nan=False`` because Python writes a non-finite score as a bare
        ``NaN`` or ``Infinity`` token that no other JSON reader accepts. A
        trajectory that cannot be read back by something other than this module
        is not the portable archive a dreaming run is inspected from, so a
        world carrying such a score fails here rather than on whoever reads it.
        """
        try:
            return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            # ValueError for a value JSON has no token for, TypeError for one
            # it cannot serialise at all. Both mean the same thing to a caller.
            raise TrajectoryError(f"this trajectory cannot be written as JSON: {exc}") from exc

    @classmethod
    def from_dict(cls, payload: Any) -> SimResult:
        payload = _object("result", payload)
        version = payload.get("schema_version")
        if version != SIM_RESULT_SCHEMA_VERSION:
            raise TrajectoryError(
                f"unreadable trajectory schema version {version!r}; "
                f"this code reads {SIM_RESULT_SCHEMA_VERSION}"
            )
        stop_reason = payload.get("stop_reason")
        if stop_reason is not None and not isinstance(stop_reason, str):
            raise TrajectoryError(f"stop_reason must be a string or null, got {stop_reason!r}")
        return cls(
            seed=_integer("seed", payload.get("seed")),
            world_size=_integer("world_size", payload.get("world_size")),
            stop_reason=stop_reason,
            attainment=_number("attainment", payload.get("attainment")),
            rounds=tuple(
                ReplayRound.from_dict(raw) for raw in _sequence("rounds", payload.get("rounds"))
            ),
            curve=tuple(
                CurvePoint.from_dict(raw) for raw in _sequence("curve", payload.get("curve"))
            ),
        )


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
        self._nodes: dict[str, Node] = {node.id: _detached(node) for node in tree.iter_nodes()}
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
        """The recorded node, detached so a reader cannot write back through it."""
        return _detached(self._nodes[node_id])

    def _recorded_children(self, node_id: str) -> tuple[str, ...]:
        """This node's recorded children, earliest-created first."""
        return self._children.get(node_id, ())

    def start(self, *, seed: int = DEFAULT_SEED) -> ReplayRun:
        """Begin a replay at ``T^{m,0} = {r}``, to be stepped by hand.

        Stepping by hand, the caller makes the decisions, so seeding them is
        the caller's job — but the trajectory still records which seed produced
        it, so ``seed`` passes through rather than defaulting under a caller
        who seeded their own policy.
        """
        return ReplayRun(self, seed=seed)

    def replay(
        self,
        policy: ReplayPolicy,
        *,
        width: int = 1,
        max_rounds: int | None = None,
        seed: int = DEFAULT_SEED,
    ) -> ReplayRun:
        """Run ``policy`` over this world until one of the termination rules fires.

        ``width`` is ``W``, the parallel worker count offered to the policy. As
        online, it is not enforced as a cap: the orchestrator already takes a
        batch as a sequence that may repeat the root and may exceed ``W`` (see
        ``orchestrator._check_batch``), and replay has to accept the batches a
        recorded rollout actually issued.

        ``max_rounds`` is ``K₂``.

        ``seed`` fixes the policy's randomness. Before the first decision the
        policy's per-rollout state is reset (§3) by calling its ``reset(rng)``
        if it defines one, with a generator seeded from ``seed`` and used by
        nothing else — so two replays of this world by this policy under the
        same seed produce the same trajectory, and a policy that ignores the
        generator produces the same trajectory under every seed.

        PAPER-GAP: §3 caps a replay at ``K₂`` decision rounds without saying
        what ``K₂`` is. The cap is load-bearing rather than cosmetic — a
        nonempty batch that reveals nothing still counts as a round, so a policy
        that keeps selecting an exhausted leaf would otherwise never terminate.
        We default it to the number of recorded attempts: the smallest cap that
        still lets any policy reveal the whole tree even at one node per round,
        so the default never truncates a replay that is making progress.
        Revisit if the authors' implementation lands (see references/method.md).
        """
        run = ReplayRun(self, seed=seed)
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset(random.Random(seed))
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

    def __init__(self, world: ReplaySimulator, *, seed: int = DEFAULT_SEED) -> None:
        self._world = world
        self._seed = _integer("seed", seed)
        self._revealed: list[str] = [world.root_id]
        self._rounds: list[ReplayRound] = []
        self._curve: list[CurvePoint] = []
        # Equation 1 maximises over a subtree that always contains the root
        # (§3), and a replay starts with the root revealed. The root is the
        # initial workspace state and normally carries no ``s_v``, but the
        # schema permits one, and where it has one a policy has attained it
        # before making a single decision.
        self._best: float | None = _score(world._node(world.root_id))
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
    def seed(self) -> int:
        """The seed this run's randomness was drawn from."""
        return self._seed

    @property
    def rounds(self) -> tuple[ReplayRound, ...]:
        return tuple(self._rounds)

    @property
    def curve(self) -> tuple[CurvePoint, ...]:
        """The attainment curve: one point per revealed node, in reveal order."""
        return tuple(self._curve)

    def result(self) -> SimResult:
        """This traversal as a serialisable record (issue #9).

        Readable mid-run as well as after one, in which case ``stop_reason`` is
        ``None`` and the trajectory is what has happened so far.
        """
        return SimResult(
            seed=self._seed,
            world_size=len(self._world),
            stop_reason=self._stop_reason,
            attainment=self._best,
            rounds=self.rounds,
            curve=self.curve,
        )

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

        Eligibility is checked once, against ``A(T^{m,k})`` as the round began,
        and the batch is then retrieved in order — so repeating a node walks
        the unrevealed children the recording holds under it: two selections of
        the root open two of its branches, and two of a leaf reveal a second
        recorded child where one exists, which a rollout of ours never records
        (see ``orchestrator._check_batch``). An empty batch is the policy's
        stop action and is not a round — passing one here is rejected rather
        than logged, because an empty round would inflate Equation 1's
        ``k^{m,★}`` with a decision that revealed nothing.
        """
        batch = tuple(batch)
        if not batch:
            raise ValueError("an empty batch ends a replay; it is not a round")
        self._check(batch, self.eligible)

        index = len(self._rounds)
        seen = set(self._revealed)
        revealed: list[str | None] = []
        observations: list[tuple[str, ...]] = []
        # Staged, not committed: a node whose score or observations a trajectory
        # cannot hold refuses the whole round rather than half of it. Committing
        # as we go would leave earlier children revealed with no ``ReplayRound``
        # describing them, and ``_next_child`` would skip the failed one as
        # already revealed on the next attempt.
        pending: list[tuple[Node, float | None]] = []
        for node_id in batch:
            child = self._next_child(node_id, seen)
            if child is None:
                observations.append(())
            else:
                seen.add(child)
                node = self._world._node(child)
                observations.append(_observations(node))
                pending.append((node, _score(node)))
            revealed.append(child)

        for node, score in pending:
            self._revealed.append(node.id)
            self._record(index, node, score)

        round_ = ReplayRound(
            index=index,
            selected=batch,
            revealed=tuple(revealed),
            observations=tuple(observations),
        )
        self._rounds.append(round_)
        return round_

    def _record(self, round_index: int, node: Node, score: float | None) -> None:
        """Extend the attainment curve with one newly revealed node.

        The running best ignores an unscored node rather than treating it as a
        zero: a hard failure is an outcome the policy paid for and learned
        from, not a score of nothing, and on a lower-is-better task converted
        to canonical units a zero would beat every real result.
        """
        if score is not None and (self._best is None or score > self._best):
            self._best = score
        self._curve.append(
            CurvePoint(
                round_index=round_index,
                node_id=node.id,
                revealed=len(self._curve) + 1,
                score=score,
                best_score=self._best,
            )
        )

    def _check(self, batch: Sequence[str], eligible: Sequence[str]) -> None:
        """Reject anything outside ``A(T^{m,k})`` — the prefix-observability rule."""
        allowed = set(eligible)
        outside = [node_id for node_id in batch if node_id not in allowed]
        if outside:
            raise ValueError(
                f"policy selected node(s) outside A(T): {', '.join(sorted(set(outside)))}; "
                f"eligible are {', '.join(eligible)}"
            )

    def _next_child(self, node_id: str, revealed: set[str]) -> str | None:
        """``Child(v; T, T^{m,k})``: the continuation this selection retrieves (§3).

        PAPER-GAP: §3 gives the root the earliest-created unrevealed child and
        every other node "its unique recorded child", because online each
        selected node yields one attempt. We apply the root's rule everywhere:
        earliest-created child not yet revealed. It is the only reading that
        generalises without contradicting the paper — the two rules coincide
        wherever a node has at most one child — and a rollout of ours records
        nothing else, since only the root may be selected twice in a batch (see
        ``orchestrator._check_batch``, and issue #31). Where a tree recorded
        elsewhere does branch off a non-root node, its extra children are
        reachable only from the batch that first reveals that node, which walks
        them by naming it again: once a round ends, the node is no longer a
        leaf and has left ``A(T)`` for good, so whatever it still holds can
        never be revealed.
        Revisit if the authors' implementation lands (see references/method.md).
        """
        for child in self._world._recorded_children(node_id):
            if child not in revealed:
                return child
        return None

    def _stop(self, reason: str) -> None:
        self._stop_reason = reason
