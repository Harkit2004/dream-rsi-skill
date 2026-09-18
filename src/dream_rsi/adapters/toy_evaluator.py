"""A toy evaluator, so the protocol has a working implementation in the repo.

The toy task is "write the shortest program that solves it": an artifact is
correct when it defines the required entry point, and the measured quantity is
how many non-empty lines it takes — a metric that runs lower-is-better, which is
the direction that gets assumed away if nothing in the repo exercises it.

It is deliberately inert: no subprocess, no filesystem read, no clock. The real
discovery task and its recorded-tree fixtures are issue #6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dream_rsi.adapters.evaluator import EvalResult, ScoreDirection

__all__ = ["ToyEvaluator"]


@dataclass(frozen=True)
class ToyEvaluator:
    """Scores an artifact by its length, once it defines ``required_token``."""

    required_token: str = "def solve("
    direction: ScoreDirection = ScoreDirection.LOWER_IS_BETTER
    baseline_score: float | None = 10.0

    def evaluate(self, artifact: str, workspace: Path) -> EvalResult:
        """Measure ``artifact``. ``workspace`` is unused: the toy task reads no files."""
        lines = [line for line in artifact.splitlines() if line.strip()]
        correct = self.required_token in artifact
        verdict = "defines" if correct else "does not define"
        return EvalResult(
            score=float(len(lines)),
            correct=correct,
            diagnostics=f"{len(lines)} non-empty line(s); {verdict} {self.required_token!r}",
        )
