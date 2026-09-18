"""Equation 1: the replay objective a policy version is scored by.

Dream-RSI §3, *Replay objective*, scores one policy version ``m`` on one
recorded world ``i``:

```
V_i^m = max_{v ∈ T_i^{m,k★}} s_v − β₁·N_i^m + β₂·N_i^m / max{1, k_i^{m,★}}
        └─ discovery quality ─┘   └─ cost ─┘   └── parallelism bonus ───┘
```

A policy is rewarded for what it found, charged for how much it had to open to
get there, and partly refunded for opening those nodes in parallel rather than
one at a time. The three inputs come off a finished replay
(:class:`~dream_rsi.replay.ReplayRun`), and the paper fixes what each is:

* **attainment** — ``max_v s_v`` over the revealed subtree. Node scores are
  canonical ``s_v``, larger-is-better (§3), so a lower-is-better task is
  already converted by the time it reaches here; see
  ``adapters.evaluator.ScoreDirection``. ``None`` where nothing scored was
  revealed.
* **revealed** — ``N_i^m = |T_i^{m,k★}| − 1``, "the number of revealed non-root
  nodes", standing in for the generation–evaluation requests the trajectory
  represents. That is ``len(run.revealed) - 1``.
* **rounds** — ``k_i^{m,★}``, "the number of completed rounds at termination".
  Not the width of a batch: the term "rewards the average number of attempts
  executed per decision round", so the same reveals batched into fewer rounds
  score higher (see issue #30). Only nonempty batches become rounds, so that is
  ``len(run.rounds)``.

This module is deliberately nothing but arithmetic: no I/O, no state, no import
of the simulator it scores or of any adapter. Pulling a ``ReplayRun`` in here
is how the objective would stop being reproducible from the numbers a run
reports, and ``tests/test_scoring.py`` asserts the imports stay out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "DEFAULT_COST_SWEEP",
    "DEFAULT_COST_WEIGHT",
    "DEFAULT_PARALLELISM_SWEEP",
    "DEFAULT_PARALLELISM_WEIGHT",
    "DEFAULT_WEIGHTS",
    "ReplayWeights",
    "replay_score",
    "sweep",
]

# PAPER-GAP: §3 requires only "fixed coefficients β₁, β₂ ≥ 0" and never gives
# their values; the appendix's "beta" is the policy's own tunable knob, not
# these. Any default is arbitrary against a task's score scale, which is
# task-specific and unbounded (inverse milliseconds, a packing radius, a
# negated runtime), so we fix defaults for a score of order 1 and make the
# sensitivity reportable with :func:`sweep` rather than pretend the numbers are
# the paper's. We charge a node 1% of a unit score and refund at most half of
# that for perfect batching, i.e. β₂ = β₁/2. The ordering matters more than the
# magnitude: with β₂ > β₁ the bonus exceeds the cost whenever a replay averages
# more than β₂/β₁ attempts per round, so revealing a node is rewarded on its
# own and a policy is paid to open the whole tree in one round regardless of
# what it finds. β₂ ≤ β₁ keeps the two terms a net cost, which is what "execution
# cost" and "bonus" are meant to be. Revisit if the authors' implementation
# lands (see references/method.md).
DEFAULT_COST_WEIGHT = 0.01
DEFAULT_PARALLELISM_WEIGHT = 0.005


@dataclass(frozen=True)
class ReplayWeights:
    """Equation 1's coefficients: ``cost`` is β₁, ``parallelism`` is β₂."""

    cost: float = DEFAULT_COST_WEIGHT
    parallelism: float = DEFAULT_PARALLELISM_WEIGHT

    def __post_init__(self) -> None:
        # §3: "For fixed coefficients β₁, β₂ ≥ 0". A negative coefficient
        # inverts the term it scales — paying a policy for execution cost, or
        # charging it for batching — so it is a different objective, not a
        # setting of this one.
        for name, value in (("cost", self.cost), ("parallelism", self.parallelism)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite coefficient >= 0, got {value!r}")


DEFAULT_WEIGHTS = ReplayWeights()

# PAPER-GAP: part of the same gap as the defaults above — the paper reports no β
# sensitivity, so the grid to report it over is ours. Each default scaled by
# {0, ½, 1, 2, 4}: it brackets the default by a factor of four either way and
# includes 0, which switches the term off entirely and shows what the objective
# would rank without it.
DEFAULT_COST_SWEEP = (0.0, 0.005, 0.01, 0.02, 0.04)
DEFAULT_PARALLELISM_SWEEP = (0.0, 0.0025, 0.005, 0.01, 0.02)


def _check_count(name: str, value: int) -> None:
    """Reject anything that is not a count of things that happened in a replay.

    ``N_i^m`` and ``k_i^{m,★}`` are cardinalities, so only a nonnegative integer
    describes a replay that could have run. A NaN would carry through the
    arithmetic and make the whole score NaN — which compares false against
    every rival, so the version neither wins nor loses a selection — an
    infinity would silently zero the parallelism term, a fraction counts
    attempts that cannot exist, and a bool is a caller mistake that would
    otherwise score as one node.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a count >= 0, got {value!r}")


def replay_score(
    attainment: float | None,
    *,
    revealed: int,
    rounds: int,
    weights: ReplayWeights = DEFAULT_WEIGHTS,
) -> float:
    """``V_i^m`` for one replay of one world (Equation 1).

    ``attainment`` is ``max_v s_v`` over the revealed subtree in canonical
    larger-is-better units, or ``None`` if no revealed node carries a score.
    ``revealed`` is ``N_i^m`` and ``rounds`` is ``k_i^{m,★}``; both are counts,
    and ``rounds`` may exceed or fall short of ``revealed`` because a nonempty
    batch can reveal several nodes or none.

    PAPER-GAP: §3 takes the maximum over a subtree that always contains the
    unscored root, and says nothing about a replay where that is the only node —
    or where every revealed node's evaluation failed, leaving ``s_v`` unset. We
    score attainment as ``-inf`` there, the same treatment
    ``ScoreDirection.is_better`` gives an unscored candidate: no attainment
    loses to any attainment. A finite convention cannot do that. Zero, in
    particular, would rank a policy that explored nothing above every policy
    that worked on a task whose canonical scores are negative, which is every
    lower-is-better task. Revisit if the authors' implementation lands (see
    references/method.md).
    """
    _check_count("revealed", revealed)
    _check_count("rounds", rounds)
    if attainment is not None and not math.isfinite(attainment):
        raise ValueError(f"attainment must be finite or None, got {attainment!r}")

    quality = -math.inf if attainment is None else float(attainment)
    cost = weights.cost * revealed
    bonus = weights.parallelism * revealed / max(1, rounds)
    return quality - cost + bonus


def sweep(
    attainment: float | None,
    *,
    revealed: int,
    rounds: int,
    cost_values: Sequence[float] = DEFAULT_COST_SWEEP,
    parallelism_values: Sequence[float] = DEFAULT_PARALLELISM_SWEEP,
) -> tuple[tuple[ReplayWeights, float], ...]:
    """One replay's score across a β grid, so a reported result can show its sensitivity.

    Returns ``(weights, score)`` pairs over the product of the two axes, in
    row-major order — ``cost_values`` outermost — so the order is the caller's
    and not a dict's or a set's (working rule 5). Scoring several replays under
    one grid is the caller's loop; this stays a function of one replay's
    numbers.
    """
    points: list[tuple[ReplayWeights, float]] = []
    for cost in cost_values:
        for parallelism in parallelism_values:
            weights = ReplayWeights(cost=cost, parallelism=parallelism)
            scored = replay_score(attainment, revealed=revealed, rounds=rounds, weights=weights)
            points.append((weights, scored))
    return tuple(points)
