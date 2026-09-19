"""The dreaming harness: M versions over every tree in the history (issue #12).

Step 3 of the loop (§3): "Each version is evaluated separately on every
historical tree", and a version's evaluation score is "its average replay score
across the fixed history". What matters about the harness is therefore that it
covers the whole grid, aggregates it the way the paper says, survives a
candidate that raises — model-written policy code is what it will be handed
(issue #13) — and reports the same numbers however much of the grid it ran at
once, because a dreaming round that depends on scheduling is a dreaming round
the development agent (issue #14) is optimising against noise.
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
    PolicyCandidate,
    ReplayWorld,
    dream,
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
