"""The dreaming harness: M versions over every tree in the history (issue #12).

Step 3 of the loop (§3): "Each version is evaluated separately on every
historical tree", and a version's evaluation score is "its average replay score
across the fixed history". What matters about the harness is therefore that it
covers the whole grid, aggregates it the way the paper says, survives a
candidate that raises — model-written policy code is what it will be handed
(issue #13) — and reports the same numbers however much of the grid it ran at
once, because a dreaming round that depends on scheduling is a dreaming round
the development agent (issue #14) is optimising against noise.

Step 5 is here too (issue #15): :func:`~dream_rsi.dream.select` reads one of
those grids and names ``π_{t+1}``. What matters about that is the floor — §3:
"Because the candidate set includes the current policy, this selection satisfies
``V^{m⋆} ≥ V^0``" — and that the floor is the incumbent's score on the history
in front of it rather than one it earned on an earlier pool.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from dream_rsi.dream import (
    DEFAULT_VERSIONS,
    DreamConfig,
    DreamReport,
    PolicyCandidate,
    ReplayWorld,
    dream,
    select,
)
from dream_rsi.policy import (
    BreadthFirstPolicy,
    BudgetAwarePolicy,
    GreedyBestFirstPolicy,
    OptimalPolicy,
)
from dream_rsi.replay import ReplaySimulator
from dream_rsi.scoring import DEFAULT_WEIGHTS, replay_score
from dream_rsi.tree import DiscoveryTree

REPO = Path(__file__).resolve().parents[1]
TREES = Path(__file__).parent / "fixtures" / "trees"

NAMES = ("wide_shallow", "narrow_deep", "failing_branch")

# The three baselines at the betas that make them decide differently — eight
# versions, which is the documented default M. Issue #12's "done when" is that M
# baselines over the three fixtures produce a stable ranking; only
# ``BreadthFirstPolicy`` reads nothing off beta, so it contributes one version.
BASELINE_VERSIONS = (
    (BreadthFirstPolicy, 1.0),
    (GreedyBestFirstPolicy, 0.5),
    (GreedyBestFirstPolicy, 1.0),
    (GreedyBestFirstPolicy, 2.0),
    (BudgetAwarePolicy, 0.25),
    (BudgetAwarePolicy, 0.5),
    (BudgetAwarePolicy, 1.0),
    (BudgetAwarePolicy, 2.0),
)


def _fixture_worlds() -> tuple[ReplayWorld, ...]:
    return tuple(
        ReplayWorld(
            name=name,
            simulator=ReplaySimulator(DiscoveryTree.load(TREES / name / "tree.json")),
        )
        for name in NAMES
    )


def _candidate(policy: type[OptimalPolicy], beta: float) -> PolicyCandidate:
    return PolicyCandidate(
        name=f"{policy.__name__}@{beta}",
        factory=lambda: policy(config={"beta": beta}),
    )


def _baselines() -> tuple[PolicyCandidate, ...]:
    return tuple(_candidate(policy, beta) for policy, beta in BASELINE_VERSIONS)


class RaisesOnItsSecondDecision(OptimalPolicy):
    """Selects the root, then raises — the crash a model-written policy brings.

    Per-rollout, because :meth:`OptimalPolicy.reset` clears the count: it
    survives a world that has nothing left to reveal after one round and raises
    on any world with more in it, which is how one candidate fails on part of
    the history and not the rest.
    """

    def reset(self, rng: Any = None) -> None:
        super().reset(rng)
        self._decisions = 0

    def choose(self, tree: DiscoveryTree, live: Sequence[str], width: int) -> Sequence[str]:
        self._decisions += 1
        if self._decisions > 1:
            raise RuntimeError("boom")
        return (tree.root_id,)


def _tree(children: int) -> DiscoveryTree:
    tree = DiscoveryTree.with_root()
    for index in range(children):
        tree.add_child(tree.root_id, score=float(index))
    return tree


def _partial_worlds() -> tuple[ReplayWorld, ...]:
    """One world a policy can finish in a single round, and one it cannot."""
    return (
        ReplayWorld(name="tiny", simulator=ReplaySimulator(_tree(1))),
        ReplayWorld(name="roomy", simulator=ReplaySimulator(_tree(3))),
    )


def test_every_version_is_replayed_on_every_world_in_the_history() -> None:
    """The grid is covered, in the order it was given (§3).

    A harness that stopped at the first world, or skipped a candidate, would
    still produce a report and a ranking — of a history it never replayed.
    """
    report = dream(_baselines(), _fixture_worlds(), config=DreamConfig(width=3))

    assert tuple(version.name for version in report.versions) == tuple(
        candidate.name for candidate in _baselines()
    )
    for version in report.versions:
        assert tuple(replay.world for replay in version.replays) == NAMES
        assert all(replay.result is not None for replay in version.replays)


def test_a_version_scores_the_average_of_its_replay_scores() -> None:
    """``V^m = (1/t) Σ_i V_i^m`` — §3's evaluation score, not a min or a sum.

    The aggregate is what selection reads (issue #15), so getting it wrong picks
    a different policy: a sum rewards a version merely for the history being
    long, and a min ranks every version by its worst world alone.
    """
    world = _fixture_worlds()
    config = DreamConfig(width=3)
    report = dream(_baselines()[:2], world, config=config)

    for version in report.versions:
        per_world = [
            replay_score(
                replay.result.attainment,
                revealed=replay.result.revealed,
                rounds=replay.result.round_count,
                weights=DEFAULT_WEIGHTS,
            )
            for replay in version.replays
        ]
        assert [replay.score for replay in version.replays] == per_world
        # Otherwise a mean is indistinguishable from a min or a max here, and the
        # test would pass against either.
        assert len(set(per_world)) > 1, "this version scored the same on every world"
        assert version.score == pytest.approx(sum(per_world) / len(per_world))


def test_the_report_is_identical_however_much_of_the_grid_ran_at_once() -> None:
    """Parallelising (version, world) pairs may not move a single number (§3).

    The pairs are independent, so the harness is free to run them together — but
    a harness that collected them as they completed would report a different
    ranking on a different day, and AGENTS.md rule 5 is what says it may not.
    A crashing candidate is in the sweep because a failure has to serialise the
    same way too.
    """
    candidates = (*_baselines()[:3], PolicyCandidate("crasher", RaisesOnItsSecondDecision))
    reports = [
        dream(candidates, _fixture_worlds(), config=DreamConfig(width=3, workers=workers)).to_json()
        for workers in (1, 4, 1, 4)
    ]

    assert len(set(reports)) == 1, "the report depends on how the grid was scheduled"
    assert json.loads(reports[0])["versions"][0]["score"] is not None


def test_a_version_that_raises_is_recorded_as_failed_without_sinking_the_sweep() -> None:
    """A crash is one version's result, not the round's (issue #12's "tests first").

    Model-written code raises, and the harness is what stands between that and a
    lost dreaming round. The crashing version keeps the worlds it did finish,
    scores no aggregate — a mean over the worlds it survived would flatter
    exactly the version that fell over on the hard ones — and ranks below every
    version that ran.
    """
    healthy = _candidate(BreadthFirstPolicy, 1.0)
    crasher = PolicyCandidate("crasher", RaisesOnItsSecondDecision)
    report = dream((crasher, healthy), _partial_worlds(), config=DreamConfig(width=1))

    failed, ran = report.versions
    assert ran.score is not None and all(replay.error is None for replay in ran.replays)

    assert failed.score is None, "a version that crashed scored an aggregate anyway"
    tiny, roomy = failed.replays
    assert tiny.error is None and tiny.score is not None
    assert roomy.result is None
    assert roomy.error is not None and "RuntimeError: boom" in roomy.error

    assert report.ranking == (healthy.name, "crasher")


def test_the_text_report_shows_every_version_every_world_and_why_one_failed() -> None:
    """The report a human reads (issue #12's scope).

    A report that quietly dropped the failed version, or named it without its
    error, is the one that sends someone looking for a version that is not
    there.
    """
    healthy = _candidate(GreedyBestFirstPolicy, 1.0)
    crasher = PolicyCandidate("crasher", RaisesOnItsSecondDecision)
    report = dream((healthy, crasher), _partial_worlds(), config=DreamConfig(width=1))
    text = report.to_text()

    for name in (healthy.name, "crasher", "tiny", "roomy"):
        assert name in text
    assert f"{report.versions[0].score:.4f}" in text
    assert "RuntimeError: boom" in text


def test_m_baselines_over_the_three_fixtures_produce_a_stable_ranking() -> None:
    """Issue #12's "done when", at the documented default M.

    Stable means two things and needs both: the same order every run and at
    every parallelism setting, and an order that is actually saying something —
    a harness whose versions all tie ranks them stably and compares nothing.
    """
    candidates = _baselines()
    assert len(candidates) == DEFAULT_VERSIONS

    rankings = {
        dream(candidates, _fixture_worlds(), config=DreamConfig(width=3, workers=workers)).ranking
        for workers in (1, 1, 4, 4)
    }

    assert len(rankings) == 1, f"the ranking moved between runs: {rankings}"
    scores = {
        version.score
        for version in dream(candidates, _fixture_worlds(), config=DreamConfig(width=3)).versions
    }
    assert len(scores) > 1, "every version scored the same: the ranking ranks nothing"


@pytest.mark.parametrize(
    ("candidates", "worlds", "match"),
    [
        pytest.param((), "fixtures", r"at least one policy version", id="no-versions"),
        pytest.param("baselines", (), r"at least one replay world", id="no-worlds"),
        pytest.param("duplicate-versions", "fixtures", r"version name.*twice", id="same-version"),
        pytest.param("baselines", "duplicate-worlds", r"world name.*twice", id="same-world"),
    ],
)
def test_a_sweep_that_could_not_be_read_afterwards_is_refused(
    candidates: Any, worlds: Any, match: str
) -> None:
    """The grid has to be a grid, and its rows and columns have to be tellable apart.

    ``M ≥ 1`` and a history of ``t ≥ 1`` trees are §3's own conditions — the
    average replay score over no worlds is not a number — and a report with two
    rows called the same thing sends whoever reads it to the wrong version.
    """
    if candidates == "baselines":
        candidates = _baselines()[:2]
    elif candidates == "duplicate-versions":
        candidates = (_candidate(BreadthFirstPolicy, 1.0), _candidate(BreadthFirstPolicy, 1.0))
    if worlds == "fixtures":
        worlds = _fixture_worlds()
    elif worlds == "duplicate-worlds":
        worlds = _partial_worlds()[:1] * 2

    with pytest.raises(ValueError, match=match):
        dream(candidates, worlds)


def test_importing_the_dream_module_reaches_no_agent_and_no_evaluator() -> None:
    """Dreaming is replay, so it stays off the adapters (§3).

    ``tests/test_replay.py`` and ``tests/test_policy.py`` assert this of the
    simulator and the policies; the harness is what drives both. If importing it
    pulled an adapter in, a dreaming round could call a discovery agent and
    score a version on an outcome nothing ever executed.
    """
    completed = subprocess.run(
        [sys.executable, "-c", "import dream_rsi.dream, sys; print('\\n'.join(sys.modules))"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    adapters = sorted(
        name for name in completed.stdout.split() if name.startswith("dream_rsi.adapters")
    )
    assert not adapters, f"importing dream_rsi.dream pulled in {', '.join(adapters)}"


class OpensBranches(OptimalPolicy):
    """Opens ``branches`` root branches, one per round, then stops.

    A version whose replay score is a function of one number, so a selection can
    be set up to have the answer the test is about rather than whichever answer
    the fixtures happen to give: opening more branches costs more nodes and
    finds whatever those branches hold.
    """

    def __init__(self, branches: int) -> None:
        self._branches = branches
        super().__init__()

    def reset(self, rng: Any = None) -> None:
        super().reset(rng)
        self._opened = 0

    def choose(self, tree: DiscoveryTree, live: Sequence[str], width: int) -> Sequence[str]:
        if self._opened >= self._branches or tree.root_id not in live:
            return ()
        self._opened += 1
        return (tree.root_id,)


def _graded_world(name: str, scores: Sequence[float]) -> ReplayWorld:
    """A world of one-node branches off the root, in the order they are opened."""
    tree = DiscoveryTree.with_root()
    for score in scores:
        tree.add_child(tree.root_id, score=score)
    return ReplayWorld(name=name, simulator=ReplaySimulator(tree))


def _early() -> ReplayWorld:
    """Pays the first branch opened: opening more only costs (Equation 1)."""
    return _graded_world("early_payoff", (1.0, 0.0, 0.0))


def _late() -> ReplayWorld:
    """Pays the last branch: a version that stops early never sees the 5.0."""
    return _graded_world("late_payoff", (0.0, 0.0, 5.0))


def _opener(name: str, branches: int) -> PolicyCandidate:
    return PolicyCandidate(name=name, factory=lambda: OpensBranches(branches))


def test_the_incumbent_is_retained_when_no_version_beats_it() -> None:
    """§3's floor, in the case that makes the loop monotone (issue #15).

    On a world that pays the first branch, every revision that opens more is
    charged for it and scores below ``π_t^0``. A selection that took the best
    revision regardless — or the last version developed, which is the one the
    agent worked hardest on — would deploy a worse policy and the guarantee
    ``V^{m⋆} ≥ V^0`` would not hold.
    """
    report = dream((_opener("v0", 1), _opener("v1", 2), _opener("v2", 3)), (_early(),))
    selection = select(report)

    assert [version.score for version in report.versions] == sorted(
        (version.score for version in report.versions), reverse=True
    ), "this world was supposed to rank the incumbent first"
    assert selection.winner == "v0"
    assert selection.retained
    assert selection.score == report.versions[0].score
    assert selection.margin == 0.0


def test_the_best_version_is_selected_when_one_beats_the_incumbent() -> None:
    """Retention is the floor, not the answer (§3: ``π_{t+1} = π_t^{m⋆}``).

    Adding a world whose payoff is on the third branch makes patience worth more
    than it costs, and the version that opens three wins by a wide margin. A
    selection that never replaced the incumbent — the easy way to satisfy the
    floor — would pass every retention test and improve nothing.
    """
    versions = (_opener("v0", 1), _opener("v1", 2), _opener("v2", 3))
    report = dream(versions, (_early(), _late()))
    selection = select(report)

    scores = {version.name: version.score for version in report.versions}
    assert selection.winner == "v2"
    assert not selection.retained
    assert selection.score == scores["v2"]
    assert selection.incumbent == "v0"
    assert selection.incumbent_score == scores["v0"]
    assert selection.margin == pytest.approx(scores["v2"] - scores["v0"])
    assert selection.margin > 0


def test_a_tie_is_broken_the_same_way_every_run_and_never_costs_the_incumbent() -> None:
    """PAPER-GAP territory: §3's ``argmax`` gives no tie-break (issue #15).

    Two things have to hold for the rule to be usable. Repeated selections over
    the same grid name the same version — a rule that fell back on set or dict
    order would drift between runs and redeploy a different policy each time —
    and a version that merely *matches* ``V^0`` does not displace the incumbent,
    since swapping in an equal policy spends an online rollout to learn nothing.
    """
    tied_with_incumbent = dream(
        (_opener("v0", 1), _opener("v1", 1), _opener("v2", 3)), (_early(),)
    )
    assert tied_with_incumbent.versions[0].score == tied_with_incumbent.versions[1].score

    tied_winners = dream((_opener("v0", 1), _opener("v1", 3), _opener("v2", 3)), (_late(),))
    assert tied_winners.versions[1].score == tied_winners.versions[2].score
    assert tied_winners.versions[1].score > tied_winners.versions[0].score

    assert {select(tied_with_incumbent).winner for _ in range(4)} == {"v0"}
    assert select(tied_with_incumbent).retained
    # The floor is the incumbent's wherever in the round it was developed, so a
    # version that ties it does not take the deployment by being written first.
    assert select(tied_with_incumbent, incumbent="v1").winner == "v1"
    # Tied revisions: the one developed first, which is the one the later ones
    # were revised from.
    assert {select(tied_winners).winner for _ in range(4)} == {"v1"}


def test_the_incumbent_is_compared_on_the_pool_in_front_of_it() -> None:
    """The floor is ``V^0`` on ``ℋ_t``, not the ``V^0`` of an earlier round.

    The same two versions, selected over a one-world pool and then over that
    pool grown by one world: the incumbent wins the first and loses the second.
    A selection that carried a winner or a score forward between rounds would
    retain the incumbent both times, and the monotonicity guarantee would be
    against a history nobody is replaying any more.
    """
    versions = (_opener("v0", 1), _opener("v1", 3))
    on_one = select(dream(versions, (_early(),)))
    on_both = select(dream(versions, (_early(), _late())))

    assert on_one.winner == "v0" and on_one.retained
    assert on_both.winner == "v1" and not on_both.retained
    assert on_both.incumbent_score != on_one.incumbent_score


def test_a_version_that_scored_no_aggregate_cannot_be_selected() -> None:
    """A crash is not a score (issue #12's ``V^m`` of a version that failed).

    The incumbent here scores below zero — every branch of this world is worth
    nothing and opening one costs — so a selection that read a missing aggregate
    as a zero, or as an improvement on "no score yet", would deploy the version
    that fell over in place of the one that ran.
    """
    report = dream(
        (_opener("v0", 1), PolicyCandidate("crasher", RaisesOnItsSecondDecision)), (_late(),)
    )

    assert report.versions[0].score is not None and report.versions[0].score < 0
    assert report.versions[1].score is None

    selection = select(report)
    assert selection.winner == "v0"
    assert selection.retained


def test_an_incumbent_that_failed_does_not_block_a_version_that_ran() -> None:
    """The floor is a number or it is nothing (PAPER-GAP, issue #15).

    §3 assumes every version has a ``V^m``. Where ``π_t^0`` itself raised on part
    of the history there is no ``V^0`` to be no worse than, and retaining a
    policy that crashes rather than deploying one that completed the pool would
    keep the loop stuck on it for every round to come.
    """
    report = dream(
        (PolicyCandidate("crasher", RaisesOnItsSecondDecision), _opener("v1", 3)), (_late(),)
    )
    selection = select(report)

    assert selection.incumbent == "crasher"
    assert selection.incumbent_score is None
    assert selection.winner == "v1"
    assert not selection.retained
    assert selection.margin is None


def test_the_rationale_records_which_version_won_and_by_how_much() -> None:
    """Issue #15's "done when": the reason is in the run log, both ways.

    A line for whoever reads the round and a payload for the run report (issue
    #18). A log that records only the winner's name leaves nobody able to tell
    an improvement from a retention, which is the one thing a reader of this
    step wants to know.
    """
    versions = (_opener("v0", 1), _opener("v1", 3))
    replaced = select(dream(versions, (_early(), _late())))
    retained = select(dream(versions, (_early(),)))

    for selection, expected in ((replaced, "v1"), (retained, "v0")):
        rationale = selection.rationale
        assert expected in rationale and selection.incumbent in rationale
        assert f"{selection.score:.4f}" in rationale
        payload = selection.to_dict()
        assert payload["winner"] == selection.winner
        assert payload["retained"] is selection.retained
        assert payload["rationale"] == rationale
        # The run report is JSON, and Python's ``-inf`` is not a token it holds.
        json.dumps(payload, allow_nan=False)

    assert "retained" in retained.rationale
    assert "retained" not in replaced.rationale


@pytest.mark.parametrize(
    ("odd_row", "incumbent", "match"),
    [
        pytest.param("pool", "nobody", r"nobody.*not one of the versions", id="absent"),
        pytest.param("pool", None, r"scored on .*early_payoff.*not on this round", id="pool"),
        pytest.param("conditions", None, r"scored under W=1 .*not this round's", id="conditions"),
    ],
)
def test_a_selection_that_could_not_hold_the_floor_is_refused(
    odd_row: str, incumbent: str | None, match: str
) -> None:
    """The three ways the floor silently stops being one (issue #15).

    Naming an incumbent the round never scored leaves nothing to be no worse
    than. The other two are the failure the issue calls out — ``develop`` builds
    its comparison out of per-version reports, so a row from another round is an
    assembly away — in its two forms: §3 evaluates the versions "on the same
    ``t`` replay worlds", and Equation 1's ``V_i^m`` is a function of ``W`` and
    the ``β`` as well. A row from an earlier, smaller pool or from a sweep run
    at another width is not a rival score, and comparing one against it would
    leave the guarantee quietly about nothing.
    """
    pool = (_early(), _late())
    config = DreamConfig(width=3)
    if odd_row == "pool":
        odd = dream((_opener("v0", 1),), pool[:1], config=config).versions[0]
    else:
        odd = dream((_opener("v0", 1),), pool, config=DreamConfig(width=1)).versions[0]
    fresh = dream((_opener("v1", 3),), pool, config=config).versions[0]
    report = DreamReport(
        versions=(odd, fresh),
        worlds=("early_payoff", "late_payoff"),
        config=config,
    )

    with pytest.raises(ValueError, match=match):
        select(report, incumbent=incumbent)


def test_versions_replayed_at_different_parallelism_settings_are_still_rivals() -> None:
    """``workers`` is not one of the conditions a comparison is held to.

    How much of a grid ran at once cannot move a number in it — the harness
    asserts that above — so rows swept at different parallelism settings are
    rivals like any others. A selection that held them to the whole
    ``DreamConfig`` would refuse a round whose incumbent happened to be scored
    on a quieter machine than its revisions, which is a property of the machine
    and of nothing in §3.
    """
    pool = (_early(), _late())
    rows = tuple(
        dream(
            (_opener(name, branches),), pool, config=DreamConfig(width=3, workers=workers)
        ).versions[0]
        for name, branches, workers in (("v0", 1, 1), ("v1", 3, 4))
    )
    report = DreamReport(
        versions=rows,
        worlds=("early_payoff", "late_payoff"),
        config=DreamConfig(width=3),
    )

    assert select(report).winner == "v1"
