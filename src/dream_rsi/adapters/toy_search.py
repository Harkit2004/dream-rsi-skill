"""A toy discovery task with real structure, and the scripted agent that explores it.

``toy_evaluator`` measures how short a program is, which is a monotone gradient:
a policy that keeps whichever child scored best cannot go wrong on it, so a
replay world recorded from it tells you nothing about a policy. This task is the
one the fixtures in ``tests/fixtures/trees/`` are recorded from, and it is built
to be able to mislead a policy.

The task is to pick a plan ``(width, depth)`` on a fixed 5×5 throughput
landscape, subject to a cost budget:

* the landscape has a **local optimum** — a plan that beats all four of its
  neighbours without being the best available plan, so a hill-climber sticks;
* its **highest cell is over budget**, so following raw throughput uphill walks
  into a region where nothing is admissible;
* an artifact the evaluator cannot read a plan out of is a **hard failure**: no
  measurement, so the branch is dead rather than merely bad.

Everything is a table lookup on a string: no subprocess, no filesystem read, no
clock, and no model call. The evaluator ignores the workspace it is handed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dream_rsi.adapters.agent import AgentContext, Artifact
from dream_rsi.adapters.evaluator import EvalResult, ScoreDirection

__all__ = [
    "BUDGET",
    "GRID",
    "LANDSCAPE",
    "MALFORMED",
    "OUT_OF_RANGE",
    "ToySearchAgent",
    "ToySearchEvaluator",
    "plan_source",
]

# Throughput of plan ``(width, depth)`` as ``LANDSCAPE[width][depth]``. Hand-made
# rather than sampled from a function, because the properties above have to hold
# exactly and be checkable by reading it: the ridge climbing to 31 in the
# bottom-right corner is entirely over budget, the 18 at (1, 1) is a local
# optimum, and the 11 at (2, 2) is the valley separating them from the best
# admissible plan, the 19 at (3, 3).
LANDSCAPE = (
    (12, 14, 13, 9, 6),
    (14, 18, 15, 10, 7),
    (13, 15, 11, 12, 16),
    (9, 10, 12, 19, 24),
    (6, 7, 14, 22, 31),
)

GRID = len(LANDSCAPE)

# A plan costs ``width + depth`` and may spend no more than this. It is what
# makes a correct candidate and a successful evaluation different things here:
# an over-budget plan evaluates fine and is simply not admissible.
BUDGET = 6

# ``fail_class`` values for the two ways an artifact carries no plan. Both are
# task-defined, so neither is ``evaluator.CRASH``: the harness worked, the
# candidate was unreadable.
MALFORMED = "malformed"
OUT_OF_RANGE = "out_of_range"

_PLAN = re.compile(r"^PLAN\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$", re.MULTILINE)


def plan_source(width: int, depth: int) -> str:
    """The artifact a candidate plan is written as. The task's whole file format."""
    return f"PLAN = ({width}, {depth})\n"


@dataclass(frozen=True)
class ToySearchEvaluator:
    """Reads a plan out of an artifact and looks its throughput up."""

    direction: ScoreDirection = ScoreDirection.HIGHER_IS_BETTER

    # The throughput of the do-nothing plan (0, 0), which every attempt starts
    # from: the reference the paper's exploration prompt exposes as
    # ``question.baseline_score`` (§B.1).
    baseline_score: float | None = float(LANDSCAPE[0][0])

    def evaluate(self, artifact: str, workspace: Path) -> EvalResult:
        """Measure ``artifact``. ``workspace`` is unused: this task reads no files."""
        match = _PLAN.search(artifact)
        if match is None:
            return EvalResult.failed(
                "no 'PLAN = (width, depth)' line in the artifact", fail_class=MALFORMED
            )

        width, depth = int(match.group(1)), int(match.group(2))
        if width >= GRID or depth >= GRID:
            return EvalResult.failed(
                f"plan ({width}, {depth}) is off the {GRID}x{GRID} grid",
                fail_class=OUT_OF_RANGE,
            )

        cost = width + depth
        if cost > BUDGET:
            # Measured, not failed: the plan is well-formed, it ran, and it is
            # inadmissible. Scoring it 0.0 rather than off the landscape keeps
            # ``max_v(s_v)`` over a branch from being won by a plan that may not
            # be used.
            return EvalResult(
                score=0.0,
                correct=False,
                diagnostics=f"plan ({width}, {depth}) costs {cost} > budget {BUDGET}",
            )

        throughput = float(LANDSCAPE[width][depth])
        return EvalResult(
            score=throughput,
            correct=True,
            diagnostics=f"plan ({width}, {depth}) costs {cost}, throughput {throughput}",
        )


@dataclass(frozen=True)
class ToySearchAgent:
    """Returns ``script[seed % len(script)]``, and depends on nothing else.

    :class:`~dream_rsi.adapters.fake_agent.FakeAgent` picks its candidate from a
    fingerprint of the whole context, which is deterministic but not
    *arrangeable*: which attempt gets which candidate falls out of a hash. A
    recording that has to place a hard failure at a chosen attempt — the
    fixtures in ``tests/fixtures/trees/`` do — needs the candidate to be a
    function of the attempt alone, and the rollout numbers its attempts by
    handing each one ``config.seed + n`` (§3 leaves generation stochastic; see
    the PAPER-GAP on :attr:`AgentContext.seed`).

    Inert and stateless, so a batch of parallel workers may share one instance.
    """

    script: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.script:
            raise ValueError("script must hold at least one candidate")

    def propose(self, context: AgentContext) -> Artifact:
        """Produce the candidate this attempt's seed selects."""
        if context.seed is None:
            raise ValueError("ToySearchAgent needs a seeded context to pick a candidate")
        index = context.seed % len(self.script)
        return Artifact(
            content=self.script[index],
            proposal=(
                f"scripted candidate {index} of {len(self.script)}, "
                f"resuming {context.parent.id}"
            ),
        )
