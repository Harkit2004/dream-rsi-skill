"""The evaluator / task adapter contract.

The paper's evaluator is task-specific: deterministic correctness checks plus a
performance measurement, producing the score ``s_v`` recorded on a node
(Dream-RSI §3, "Discovery trees and the shared decision interface"). The three
paper domains — algorithm engineering, mathematical optimization, GPU kernel
engineering — measure completely different quantities in completely different
units, so this module defines an interface and never an implementation.

Two things the paper does fix, and that this module therefore fixes too:

* ``s_v`` as stored on a node is **larger-is-better**: "Scores follow a fixed
  task-scoring protocol, with larger values indicating better quality" (§3).
* A *successful evaluation* is not the same thing as a *correct candidate*. An
  evaluation with ``error is None`` and ``fail_class == "ok"`` succeeded even
  when the candidate it measured is invalid (§B.1, "Success semantics").

The task's own metric need not run larger-is-better — §4 scores autocorrelation
and downstream runtime lower-is-better — so every task states its
:class:`ScoreDirection` and the conversion to ``s_v`` happens in one place,
:meth:`EvalResult.to_node_fields`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CRASH",
    "OK",
    "EvalResult",
    "ScoreDirection",
    "TaskEvaluator",
    "safe_evaluate",
]

# ``fail_class`` values. The paper names only "ok" (§B.1); the discovery agent
# reads the rest as free text, so tasks may add their own classes.
OK = "ok"

# PAPER-GAP: the paper lists no fail_class for an evaluator that crashes rather
# than returning a verdict — its prompts assume a score.json or an error.txt is
# always produced. We use "crash" so a harness failure is distinguishable from a
# task-defined failure of the candidate. Revisit if the authors' implementation
# lands (see references/method.md).
CRASH = "crash"


class ScoreDirection(str, Enum):
    """Which way a task's own metric runs. Part of the task contract, never assumed."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"

    def is_better(self, a: float | None, b: float | None) -> bool:
        """Is score ``a`` strictly better than score ``b`` under this direction?

        ``None`` is the score of an evaluation that produced no measurement, and
        is worse than every real score — including another ``None``, so that
        "strictly better" stays a strict order and two failures never displace
        each other.
        """
        if a is None:
            return False
        if b is None:
            return True
        return a > b if self is ScoreDirection.HIGHER_IS_BETTER else a < b

    def to_canonical(self, score: float | None) -> float | None:
        """Map a task metric onto ``s_v``, which is larger-is-better (§3)."""
        if score is None:
            return None
        # PAPER-GAP: the paper states that s_v is larger-is-better but not how a
        # lower-is-better task metric is mapped onto it. (For kernels it reports
        # inverse runtime, 1/ms, so some mapping happens.) We negate: it is
        # order-reversing for every real score, needs no special case at zero,
        # and is defined for negative metrics. Note that the choice is not
        # order-only — Equation 1 subtracts β₁·N from max_v(s_v), so the scale of
        # this mapping interacts with the β defaults chosen in scoring.py.
        return score if self is ScoreDirection.HIGHER_IS_BETTER else -score


@dataclass(frozen=True)
class EvalResult:
    """One evaluation of one artifact.

    ``score`` is in the task's own units and direction; it is ``None`` when no
    measurement was produced. ``correct`` is the deterministic correctness
    verdict, kept separate from ``fail_class``/``error`` because a successful
    evaluation may well report an incorrect candidate. ``diagnostics`` is the
    text the discovery agent reads when it resumes from this node.
    """

    score: float | None
    correct: bool
    diagnostics: str = ""
    fail_class: str = OK
    error: str | None = None

    @property
    def evaluated(self) -> bool:
        """Did the evaluation itself succeed? (§B.1 success semantics.)"""
        return self.error is None and self.fail_class == OK

    @classmethod
    def failed(cls, error: str, *, fail_class: str = CRASH, diagnostics: str = "") -> EvalResult:
        """A result standing in for an evaluation that produced no measurement."""
        return cls(
            score=None,
            correct=False,
            diagnostics=diagnostics or error,
            fail_class=fail_class,
            error=error,
        )

    def to_node_fields(self, direction: ScoreDirection) -> dict[str, Any]:
        """The ``tree.Node`` fields this result contributes.

        Used as ``tree.add_child(parent_id, artifact=..., **result.to_node_fields(d))``.
        The node's ``score`` is canonical ``s_v``; the task's own number survives
        in the diagnostics alongside the direction that produced it, so a
        recorded tree can be read back in the task's units.
        """
        return {
            "score": direction.to_canonical(self.score),
            "diagnostics": {
                "text": self.diagnostics,
                "correct": self.correct,
                "fail_class": self.fail_class,
                "error": self.error,
                "raw_score": self.score,
                "direction": direction.value,
            },
        }


@runtime_checkable
class TaskEvaluator(Protocol):
    """What a task must supply for its candidates to be scored.

    Implementations are structural: anything with these three members satisfies
    the protocol, no base class and no registration.
    """

    #: Which way :attr:`EvalResult.score` runs for this task.
    direction: ScoreDirection

    #: The task's reference score, in the same units and direction as
    #: :attr:`EvalResult.score`, or ``None`` where the task defines no
    #: reference. The paper's exploration prompt exposes it as
    #: ``question.baseline_score`` and the observation helpers compare probes
    #: against it (§B.1; issue #11).
    baseline_score: float | None

    def evaluate(self, artifact: str, workspace: Path) -> EvalResult:
        """Check and measure ``artifact`` as produced in ``workspace``."""
        ...


def safe_evaluate(evaluator: TaskEvaluator, artifact: str, workspace: Path) -> EvalResult:
    """Run ``evaluator``, turning a crash into a failed :class:`EvalResult`.

    One node whose evaluation raises must not take the rollout down with it: the
    node is recorded with no score and the rest of the batch continues.
    ``BaseException`` (a cancelled worker, an interrupt) is deliberately not
    caught — that is the rollout ending, not a node failing.
    """
    try:
        return evaluator.evaluate(artifact, workspace)
    except Exception as exc:  # noqa: BLE001 - the whole point is to not propagate
        return EvalResult.failed(f"{type(exc).__name__}: {exc}")
