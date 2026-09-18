"""The toy discovery task the fixtures are recorded from (issue #6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dream_rsi.adapters.agent import AgentContext
from dream_rsi.adapters.toy_search import (
    BUDGET,
    GRID,
    LANDSCAPE,
    MALFORMED,
    OUT_OF_RANGE,
    ToySearchAgent,
    ToySearchEvaluator,
    plan_source,
)
from dream_rsi.tree import Node

WORKSPACE = Path("unused")


def _cells() -> list[tuple[int, int]]:
    return [(width, depth) for width in range(GRID) for depth in range(GRID)]


def _neighbours(width: int, depth: int) -> list[tuple[int, int]]:
    candidates = [(width - 1, depth), (width + 1, depth), (width, depth - 1), (width, depth + 1)]
    return [(w, d) for w, d in candidates if 0 <= w < GRID and 0 <= d < GRID]


def test_landscape_is_rugged_rather_than_a_gradient() -> None:
    """A monotone landscape makes every replay test pass vacuously.

    Two properties keep it honest: a plan that beats all its neighbours without
    being the best plan (a hill-climber gets stuck on it), and a raw peak that
    the budget rules out (a hill-climber that ignores correctness walks into a
    dead region). Either one disappearing is what this catches.
    """
    admissible = {c: LANDSCAPE[c[0]][c[1]] for c in _cells() if sum(c) <= BUDGET}
    best = max(admissible, key=lambda c: admissible[c])

    local_optima = [
        cell
        for cell in admissible
        if all(LANDSCAPE[cell[0]][cell[1]] > LANDSCAPE[w][d] for w, d in _neighbours(*cell))
    ]
    assert [cell for cell in local_optima if cell != best], (
        f"landscape has no local optimum besides {best}: nothing for a climber to get stuck on"
    )

    raw_peak = max(_cells(), key=lambda c: LANDSCAPE[c[0]][c[1]])
    assert sum(raw_peak) > BUDGET, f"raw peak {raw_peak} is admissible: the budget rules nothing out"


@pytest.mark.parametrize(
    ("artifact", "evaluated", "correct", "score", "fail_class"),
    [
        (plan_source(1, 1), True, True, float(LANDSCAPE[1][1]), None),
        (plan_source(0, 0), True, True, float(LANDSCAPE[0][0]), None),
        # Over budget: the evaluation succeeded, the candidate is inadmissible
        # and measures nothing (§B.1 keeps those two verdicts apart).
        (plan_source(4, 4), True, False, 0.0, None),
        ("# thought about it, wrote nothing\n", False, False, None, MALFORMED),
        (plan_source(GRID, 0), False, False, None, OUT_OF_RANGE),
        # Off the grid the other way: a plan the evaluator can read perfectly
        # well, so it is out of range rather than unreadable.
        (plan_source(-1, 0), False, False, None, OUT_OF_RANGE),
        (plan_source(0, -1), False, False, None, OUT_OF_RANGE),
    ],
)
def test_evaluator_classifies_plans(
    artifact: str,
    evaluated: bool,
    correct: bool,
    score: float | None,
    fail_class: str | None,
) -> None:
    """Each kind of artifact lands in the right one of the three outcomes.

    Catches an over-budget plan being scored off the landscape anyway, and a
    plan the evaluator could not read being recorded as a successful zero.
    """
    result = ToySearchEvaluator().evaluate(artifact, WORKSPACE)

    assert result.evaluated is evaluated
    assert result.correct is correct
    assert result.score == score
    if fail_class is not None:
        assert result.fail_class == fail_class


def test_agent_answers_to_the_seed_alone() -> None:
    """The candidate depends on the seed and nothing else.

    That is what lets a recording place a specific candidate — a malformed one,
    say — at a specific attempt. An agent that keyed on the inherited history
    instead (as FakeAgent does) would still be deterministic but would make the
    recorded corpus shapes an accident, which is what this catches.
    """
    script = (plan_source(0, 0), plan_source(1, 1), "# nothing\n")
    agent = ToySearchAgent(script=script)
    root = Node(id="n000000")
    deeper = Node(id="n000001", parent_id="n000000", score=1.0)

    shallow_context = AgentContext(problem="p", workspace=WORKSPACE, history=(root,), seed=1)
    deep_context = AgentContext(
        problem="p", workspace=Path("elsewhere"), history=(root, deeper), seed=1
    )
    assert agent.propose(shallow_context).content == agent.propose(deep_context).content

    contents = {
        agent.propose(
            AgentContext(problem="p", workspace=WORKSPACE, history=(root,), seed=seed)
        ).content
        for seed in range(len(script))
    }
    assert contents == set(script)
