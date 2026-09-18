"""Replay determinism and the trajectory log a replay leaves behind (issue #9).

AGENTS.md rule 5 lives here. Everything downstream — comparing ``M`` policy
versions over one history, selecting a winner that is provably no worse than
the incumbent, believing any reported improvement — rests on a replay of the
same world by the same policy under the same seed producing the same
trajectory. If it drifts, the policy-development agent is optimising against
noise.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dream_rsi.replay import (
    SIM_RESULT_SCHEMA_VERSION,
    ReplayRound,
    ReplaySimulator,
    SimResult,
    TrajectoryError,
)
from dream_rsi.tree import DiscoveryTree

REPO = Path(__file__).resolve().parents[1]
TREES = Path(__file__).parent / "fixtures" / "trees"

NAMES = ("wide_shallow", "narrow_deep", "failing_branch")


def _load(name: str) -> DiscoveryTree:
    return DiscoveryTree.load(TREES / name / "tree.json")


@dataclass
class RandomWalk:
    """A policy whose every decision is a coin flip, so a lost seed shows up.

    It takes its randomness from the generator replay hands it and nowhere
    else: with the seed unwired, ``_rng`` stays ``None`` and the replay fails
    loudly rather than quietly sampling from the global stream.
    """

    budget: int = 6
    _rng: random.Random | None = field(default=None, init=False)
    _left: int = field(default=0, init=False)

    def reset(self, rng: random.Random) -> None:
        self._rng = rng
        self._left = self.budget

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        assert self._rng is not None, "replay never handed the policy a seeded generator"
        if self._left <= 0:
            return ()
        self._left -= 1
        size = self._rng.randint(1, width)
        return tuple(self._rng.choice(eligible) for _ in range(size))


@dataclass
class OpenThenProbe:
    """Opens three branches off the root, then extends one and re-probes another.

    Deterministic, and shaped so one replay of ``failing_branch`` covers every
    case a trajectory log has to get right: a reveal that scores, one that
    failed hard and scores nothing, a selection with no recorded continuation
    left, and a round holding both kinds at once.
    """

    _round: int = 0

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        self._round += 1
        if self._round > 4:
            return ()
        if self._round <= 3:
            return (tree.root_id,)
        return (eligible[1], eligible[-1])


@dataclass
class OpenOnce:
    """Opens one branch off the root, then stops."""

    _done: bool = False

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        if self._done:
            return ()
        self._done = True
        return (tree.root_id,)


@dataclass
class StopImmediately:
    """Selects nothing, ending the replay with the root alone revealed."""

    def select(self, tree: DiscoveryTree, eligible: tuple[str, ...], width: int) -> tuple[str, ...]:
        return ()


def _replay(name: str, *, seed: int = 11) -> SimResult:
    return ReplaySimulator(_load(name)).replay(RandomWalk(), width=3, seed=seed).result()


@pytest.mark.parametrize("name", NAMES)
def test_the_determinism_guarantee_same_tree_policy_and_seed_replay_identically(name: str) -> None:
    """**The determinism test (AGENTS.md rule 5). Do not delete it.**

    Same recorded world, same policy, same seed, twice over: the serialised
    trajectories must be identical byte for byte. It fails if anything in the
    replay path samples from an unseeded generator, iterates a set or a dict
    whose order is not fixed, or writes a wall-clock reading into the log.
    """
    first = _replay(name).to_json()
    second = _replay(name).to_json()

    assert first.encode("utf-8") == second.encode("utf-8")
    # A policy that never got to make a decision would pass the line above
    # while proving nothing, so insist the replay actually traversed something.
    assert json.loads(first)["rounds"], "the replay recorded no rounds"


@pytest.mark.parametrize("name", NAMES)
def test_a_different_seed_gives_a_randomised_policy_a_different_trajectory(name: str) -> None:
    """The other half of the guarantee: the seed is wired through, not ignored.

    A replay that traverses the same path under every seed is reproducible for
    the trivial reason that the policy's randomness came from somewhere replay
    does not control — which is exactly the drift the guarantee is meant to
    exclude. Compared on the rounds rather than the serialised result, because
    that carries the seed and would differ whether or not it changed anything.
    """
    trajectories = {_replay(name, seed=seed).rounds for seed in range(8)}

    assert len(trajectories) > 1


@pytest.mark.parametrize("name", NAMES)
def test_replay_is_reproducible_across_interpreter_hash_seeds(name: str) -> None:
    """Two interpreters that hash strings differently replay the same trajectory.

    Within one process a set of node ids iterates the same way every time, so
    the determinism test above cannot see a set that leaked into the reveal
    order. Across processes with different ``PYTHONHASHSEED`` values it can.
    """
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_determinism import _replay\n"
        "print(_replay(sys.argv[1]).to_json())\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "4242"):
        completed = subprocess.run(
            [sys.executable, "-c", script, name],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONHASHSEED": hash_seed},
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.add(completed.stdout)

    assert len(outputs) == 1


@pytest.mark.parametrize("name", NAMES)
def test_sim_result_round_trips_through_serialisation(name: str) -> None:
    """A dreaming run's log can be written out and read back unchanged.

    Issue #12 keeps these around to feed the policy-development agent, so a
    result that only survives in memory is not a record of anything.
    """
    result = _replay(name)

    restored = SimResult.from_dict(json.loads(result.to_json()))

    assert restored == result
    assert restored.to_json() == result.to_json()


@pytest.mark.parametrize(
    "damage",
    [
        {"schema_version": SIM_RESULT_SCHEMA_VERSION + 1},
        {"schema_version": None},
        {"seed": "eleven"},
        {"rounds": {}},
        {"curve": [{"round_index": 0, "node_id": 3, "revealed": 1}]},
    ],
)
def test_a_trajectory_this_code_cannot_read_is_refused_rather_than_guessed(
    damage: dict[str, object],
) -> None:
    """A stored log that is not what it claims fails to load, loudly.

    These are archives: a dreaming run's logs outlive the code that wrote them.
    Decoding one into the wrong shape would let a policy version be re-scored
    against a trajectory it never took, which is worse than not loading it.
    """
    payload = {**_replay("narrow_deep").to_dict(), **damage}

    with pytest.raises(TrajectoryError):
        SimResult.from_dict(payload)


def test_a_round_whose_columns_do_not_line_up_is_refused() -> None:
    """``selected``, ``revealed`` and ``observations`` align pairwise or the log is junk.

    Every consumer reads a round by position — which selection revealed what,
    and what that exposed. A stored round with more selections than reveals
    silently drops the last one from every ``zip`` that walks it, and reports a
    decision round that cost less than it did.
    """
    payload = _replay("wide_shallow").to_dict()
    payload["rounds"][0]["selected"] = [*payload["rounds"][0]["selected"], "n000000"]

    with pytest.raises(TrajectoryError):
        SimResult.from_dict(payload)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf, True])
def test_a_score_a_trajectory_cannot_hold_is_refused_where_it_enters(score: float) -> None:
    """The producer is held to the contract the reader enforces, not a looser one.

    A replay that writes a score its own loader refuses produces a log that
    cannot be read back — the archive is worthless exactly when someone comes
    to inspect it. NaN and infinity qualify: a NaN stays the running best
    forever because every comparison against it is false, an infinite
    attainment is rejected outright by ``scoring.replay_score``, and Python
    writes both as bare tokens no other JSON reader accepts. So does ``True``,
    which is an ``int`` to ``isinstance`` and a JSON boolean on the way out.

    Both places a node's score enters a trajectory are covered: the revealed
    node, and the root a run starts from.
    """
    revealed = DiscoveryTree.with_root()
    revealed.add_child(revealed.root_id, score=score)

    with pytest.raises(TrajectoryError):
        ReplaySimulator(revealed).replay(OpenOnce())

    with pytest.raises(TrajectoryError):
        ReplaySimulator(DiscoveryTree.with_root(score=score)).replay(StopImmediately())


def test_an_observation_a_trajectory_cannot_hold_is_refused_where_it_enters() -> None:
    """The same contract, for the other thing a reveal copies out of the world.

    ``Node`` checks that ``observations`` is a list and nothing about what is
    in it, while a round's observations are read back as strings. A tree
    carrying anything else would replay into a log that could not be decoded.
    """
    tree = DiscoveryTree.with_root()
    tree.add_child(tree.root_id, score=1.0, observations=(17,))

    with pytest.raises(TrajectoryError):
        ReplaySimulator(tree).replay(OpenOnce())


def test_a_hand_stepped_run_records_the_seed_it_was_started_with() -> None:
    """Stepping by hand, the caller owns the randomness — and the log says which.

    ``start`` is the entry point for driving a replay a round at a time, so a
    caller who seeds their own decisions and feeds the batches in has a
    trajectory whose seed is theirs. Recording the default instead would put a
    seed on the archive that never produced it.
    """
    run = ReplaySimulator(_load("wide_shallow")).start(seed=4242)
    run.reveal((run.revealed.root_id,))

    result = run.result()

    assert result.seed == 4242
    assert result.round_count == 1
    assert result.revealed == 1


def test_a_batch_that_cannot_be_logged_leaves_the_run_untouched() -> None:
    """A refused reveal is not a half-taken round.

    The check runs per node, so a batch whose second selection carries
    something a trajectory cannot hold would otherwise leave the first child
    revealed with no ``ReplayRound`` describing it — and ``_next_child`` would
    treat the failed one as already revealed on the next attempt. The round is
    all or nothing.
    """
    tree = DiscoveryTree.with_root()
    tree.add_child(tree.root_id, score=1.0, observations=("fine",))
    tree.add_child(tree.root_id, score=2.0, observations=(17,))

    run = ReplaySimulator(tree).start()
    before = run.revealed

    with pytest.raises(TrajectoryError):
        run.reveal((run.revealed.root_id, run.revealed.root_id))

    assert run.revealed == before
    assert run.rounds == ()
    assert run.curve == ()


def test_a_seed_a_trajectory_cannot_hold_is_refused_where_it_enters() -> None:
    """The seed is held to the contract it is read back under, like every other field.

    ``True`` is an ``int`` to ``isinstance`` and a JSON boolean on the way out,
    which the loader refuses — so a run started with one would produce an
    archive that cannot be read.
    """
    with pytest.raises(TrajectoryError):
        ReplaySimulator(_load("narrow_deep")).start(seed=True)


def test_a_stored_trajectory_is_read_and_written_as_standard_json() -> None:
    """The reader's end of that same contract, and the writer's last line of defence.

    ``SimResult`` is public: a caller can build one by hand, and ``to_json``
    still may not emit the bare ``Infinity`` token that would make the archive
    unreadable by anything but Python.
    """
    payload = _replay("narrow_deep").to_dict()
    payload["curve"][0]["best_score"] = math.inf

    with pytest.raises(TrajectoryError):
        SimResult.from_dict(payload)

    # Two ways a hand-built result fails to encode — a value JSON has no token
    # for, and a value whose type it cannot serialise at all. The encoder
    # raises ValueError for one and TypeError for the other; both mean the same
    # thing to a caller, so both arrive as TrajectoryError.
    unwritable = [
        SimResult(
            seed=0, world_size=1, stop_reason=None, attainment=math.inf, rounds=(), curve=()
        ),
        SimResult(
            seed=0,
            world_size=2,
            stop_reason=None,
            attainment=None,
            rounds=(ReplayRound(index=0, selected=(object(),), revealed=(None,), observations=((),)),),
            curve=(),
        ),
    ]

    for result in unwritable:
        with pytest.raises(TrajectoryError):
            result.to_json()


def test_a_scored_root_counts_toward_attainment() -> None:
    """Equation 1 maximises over a subtree that always contains the root (§3).

    The root is the initial workspace state and normally carries no ``s_v``,
    but the schema lets it carry one, and a replay starts with it revealed. A
    policy that stops immediately has still attained whatever the root scored;
    reporting ``None`` would send that replay into ``replay_score`` as ``-inf``,
    below every rival, for a world where something was in fact attained.
    """
    tree = DiscoveryTree.with_root(score=9.0)
    tree.add_child(tree.root_id, score=4.0)

    stopped = ReplaySimulator(tree).replay(StopImmediately()).result()

    assert stopped.round_count == 0
    assert stopped.revealed == 0
    assert stopped.attainment == 9.0

    walked = ReplaySimulator(tree).replay(OpenOnce()).result()

    # The root is not an attempt, so it is no curve point and no revealed node —
    # but it is still the best thing the replay has seen.
    assert walked.revealed == 1
    assert [point.score for point in walked.curve] == [4.0]
    assert [point.best_score for point in walked.curve] == [9.0]
    assert walked.attainment == 9.0


def test_the_trajectory_records_what_each_round_selected_revealed_and_exposed() -> None:
    """Per-round observations are the newly revealed nodes' own, in reveal order.

    §3: "the newly revealed nodes expose their stored observations before the
    policy makes its next decision". The log is what the development agent
    reads a trajectory off, so a round that drops or misaligns them describes a
    decision the policy never faced.
    """
    tree = _load("failing_branch")

    result = ReplaySimulator(tree).replay(OpenThenProbe()).result()

    assert [round_.index for round_ in result.rounds] == list(range(len(result.rounds)))
    for round_ in result.rounds:
        assert len(round_.observations) == len(round_.revealed) == len(round_.selected)
        for node_id, observations in zip(round_.revealed, round_.observations):
            expected = () if node_id is None else tree.node(node_id).observations
            assert observations == expected

    revealed = [node_id for round_ in result.rounds for node_id in round_.revealed]
    assert any(node_id is None for node_id in revealed), "no barren selection to align around"
    assert any(node_id is not None for node_id in revealed), "the replay revealed nothing at all"
    assert any(tree.node(node_id).observations for node_id in revealed if node_id), (
        "every revealed node's observations were empty, so alignment proves nothing"
    )


def test_the_curve_tracks_the_best_score_found_against_attempts_spent() -> None:
    """One point per revealed node, carrying the running best — the paper's curve.

    The appendix's policy skeleton records a curve point on every reveal, which
    is how a replay reports quality against cumulative generations rather than
    a single final number. A curve whose best score drops, or that skips a
    revealed node, is not that.
    """
    tree = _load("failing_branch")

    result = ReplaySimulator(tree).replay(OpenThenProbe()).result()

    revealed = [node_id for round_ in result.rounds for node_id in round_.revealed if node_id]
    assert None in [tree.node(node_id).score for node_id in revealed], (
        "no hard-failing node revealed: the running best never had to skip one"
    )
    assert [point.node_id for point in result.curve] == revealed
    assert [point.revealed for point in result.curve] == list(range(1, len(revealed) + 1))
    assert [point.score for point in result.curve] == [tree.node(n).score for n in revealed]

    best = [point.best_score for point in result.curve]
    assert best == sorted(best, key=lambda s: (s is not None, s))
    assert result.attainment == best[-1]


@pytest.mark.parametrize("name", NAMES)
def test_sim_result_reports_equation_1s_inputs(name: str) -> None:
    """``N_i^m`` and ``k_i^{m,★}`` come off the log, so a score is recomputable.

    Equation 1 is a pure function of these three numbers (``scoring.py``), and
    the log is what a finished dreaming run is inspected from. If they disagree
    with the run that produced them, an archived result cannot be re-scored
    under different β without replaying everything again.
    """
    run = ReplaySimulator(_load(name)).replay(RandomWalk(), width=3, seed=11)
    result = run.result()

    assert result.revealed == len(run.revealed) - 1
    assert result.round_count == len(run.rounds)
    assert result.stop_reason == run.stop_reason
    scores = [node.score for node in run.revealed.iter_nodes() if node.score is not None]
    assert result.attainment == (max(scores) if scores else None)
