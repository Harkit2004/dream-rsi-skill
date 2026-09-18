"""Equation 1, the replay objective."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dream_rsi.adapters.evaluator import ScoreDirection
from dream_rsi.scoring import (
    DEFAULT_WEIGHTS,
    ReplayWeights,
    replay_score,
    sweep,
)

REPO = Path(__file__).resolve().parents[1]

# Round numbers, so the expectations below can be read off Equation 1 by hand.
HAND = ReplayWeights(cost=0.1, parallelism=0.05)


@pytest.mark.parametrize(
    ("attainment", "revealed", "rounds", "expected"),
    [
        # max_v s_v − β₁·N + β₂·N/max{1,k}
        (1.0, 4, 2, 1.0 - 0.4 + 0.1),
        (1.0, 4, 4, 1.0 - 0.4 + 0.05),
        (-2.0, 3, 1, -2.0 - 0.3 + 0.15),
        # An empty replay: no reveals, no rounds, so only attainment survives.
        (0.5, 0, 0, 0.5),
    ],
)
def test_hand_computed_scores(
    attainment: float, revealed: int, rounds: int, expected: float
) -> None:
    """The three terms carry their stated signs and the stated denominator."""
    scored = replay_score(attainment, revealed=revealed, rounds=rounds, weights=HAND)
    assert scored == pytest.approx(expected)


def test_higher_attainment_scores_higher() -> None:
    """Discovery quality enters positively: the better find wins, all else equal."""
    worse = replay_score(1.0, revealed=3, rounds=2)
    better = replay_score(2.0, revealed=3, rounds=2)
    assert better > worse


def test_revealing_more_nodes_scores_lower() -> None:
    """Execution cost enters negatively, and outweighs the parallelism refund.

    Under the default weights β₂ < β₁, so a node is never worth revealing for
    the bonus alone — even revealed at maximum parallelism, in a single round.
    """
    assert replay_score(1.0, revealed=2, rounds=2) > replay_score(1.0, revealed=5, rounds=2)
    assert replay_score(1.0, revealed=2, rounds=1) > replay_score(1.0, revealed=5, rounds=1)


def test_batching_the_same_reveals_into_fewer_rounds_scores_higher() -> None:
    """The bonus rewards attempts per round: same N, fewer rounds scores higher.

    The denominator is ``k^{m,★}``, the number of completed rounds (§3), not the
    width of a batch, so this is the direction the term runs in: six nodes
    revealed two-at-a-time beats the same six revealed one-at-a-time.
    """
    parallel = replay_score(1.0, revealed=6, rounds=3)
    sequential = replay_score(1.0, revealed=6, rounds=6)
    assert parallel > sequential


def test_zero_rounds_does_not_divide_by_zero() -> None:
    """The ``max{1, k}`` guard: k = 0 is finite, and scores as k = 1 does."""
    scored = replay_score(1.0, revealed=2, rounds=0, weights=HAND)
    assert math.isfinite(scored)
    assert scored == pytest.approx(replay_score(1.0, revealed=2, rounds=1, weights=HAND))


def test_lower_is_better_task_scores_its_better_metric_higher() -> None:
    """Attainment is canonical ``s_v``, so a lower-is-better task is respected.

    The direction conversion belongs to the evaluator contract (issue #2); what
    Equation 1 must not do is re-interpret the number it is handed. A 1.5 ms
    candidate has to outscore a 2.5 ms one on a task measured in milliseconds.
    """
    direction = ScoreDirection.LOWER_IS_BETTER
    faster = replay_score(direction.to_canonical(1.5), revealed=3, rounds=2)
    slower = replay_score(direction.to_canonical(2.5), revealed=3, rounds=2)
    assert faster > slower


@pytest.mark.parametrize("attained", [-100.0, 0.0, 5.0])
def test_attaining_nothing_scores_below_attaining_anything(attained: float) -> None:
    """A replay that revealed no scored node has no discovery quality.

    It must lose to one that found something, whatever that something was and
    however many nodes it took — a lower-is-better task's canonical scores are
    all negative, so treating "nothing attained" as a score of zero would rank
    a policy that explored nothing above every policy that worked.
    """
    nothing = replay_score(None, revealed=0, rounds=0)
    something = replay_score(attained, revealed=20, rounds=20)
    assert something > nothing


@pytest.mark.parametrize(
    ("cost", "parallelism"),
    [(-0.1, 0.0), (0.0, -0.1)],
)
def test_weights_reject_negative_coefficients(cost: float, parallelism: float) -> None:
    """§3 fixes β₁, β₂ ≥ 0; a negative one inverts the term it scales."""
    with pytest.raises(ValueError):
        ReplayWeights(cost=cost, parallelism=parallelism)


@pytest.mark.parametrize(
    ("attainment", "revealed", "rounds"),
    [
        (1.0, -1, 0),
        (1.0, 0, -1),
        (math.nan, 1, 1),
        (math.inf, 1, 1),
    ],
)
def test_rejects_counts_and_attainment_that_cannot_be_scored(
    attainment: float, revealed: int, rounds: int
) -> None:
    """Nonsense in is not a number out.

    A NaN attainment would compare false against every rival and so neither win
    nor lose a version selection, and a negative count would turn the cost term
    into a reward.
    """
    with pytest.raises(ValueError):
        replay_score(attainment, revealed=revealed, rounds=rounds)


def test_sweep_scores_every_grid_point_under_its_own_weights() -> None:
    """The sweep is Equation 1 over a β grid, with the axes not transposed."""
    costs = (0.0, 0.1)
    parallelisms = (0.0, 0.05, 0.2)

    points = sweep(1.0, revealed=4, rounds=2, cost_values=costs, parallelism_values=parallelisms)

    assert [weights for weights, _ in points] == [
        ReplayWeights(cost=cost, parallelism=parallelism)
        for cost in costs
        for parallelism in parallelisms
    ]
    for weights, scored in points:
        assert scored == pytest.approx(
            replay_score(1.0, revealed=4, rounds=2, weights=weights)
        )


def test_sweep_defaults_to_a_grid_around_the_default_weights() -> None:
    """A reported result can show its β sensitivity without picking a grid.

    The default grid has to actually vary both coefficients and include the
    defaults, or the sensitivity it shows is no sensitivity at all.
    """
    points = sweep(1.0, revealed=4, rounds=2)
    weights = [w for w, _ in points]

    assert DEFAULT_WEIGHTS in weights
    assert len({w.cost for w in weights}) > 1
    assert len({w.parallelism for w in weights}) > 1
    assert len({scored for _, scored in points}) > 1


def test_scoring_imports_neither_the_simulator_nor_an_adapter() -> None:
    """Equation 1 stays a pure function of numbers (issue #8 scope).

    Importing the replay simulator or an adapter is how it would stop being
    one: the next step after ``import`` is a convenience that takes a
    ``ReplayRun``, reaches into it, and puts I/O behind the scoring call.
    """
    completed = subprocess.run(
        [sys.executable, "-c", "import dream_rsi.scoring, sys; print('\\n'.join(sys.modules))"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = set(completed.stdout.split())
    assert not [name for name in loaded if name.startswith("dream_rsi.adapters")]
    assert "dream_rsi.replay" not in loaded
