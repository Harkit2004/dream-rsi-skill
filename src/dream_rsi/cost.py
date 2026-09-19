"""What a cycle spent, counted on each side of the loop (issue #18).

The paper's headline claim is quality "while substantially reducing discovery
cost" (§1), and §4 says what it prices that in: "The discovery cost is quantified
by the total cumulative number of discovery-agent calls." So the online half of a
cycle is counted in calls to the discovery agent, and against it stands what the
offline half did *instead* of calling one — replay, which §2 prices at nothing:
"a single costly online run enables thousands of rapid, zero-execution-cost
off-policy evaluations". The ratio between those two is the argument, which is
why the two halves are counted separately here and never added together.

Every number in this module is a count of something that happened, so a run under
one seed produces the same costs twice (working rule 5) and a cycle record can
carry them. How long a cycle *took* is not that: :class:`CycleTiming` measures the
machine it ran on rather than the run, which is why it is a separate type kept
out of the record.

Nothing here computes a number of its own — the counts are made where the work
is, in :mod:`dream_rsi.orchestrator`, :mod:`dream_rsi.develop` and
:mod:`dream_rsi.dream` — so this module imports nothing from the package and is
only the shape those counts are carried and read back in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CycleCost",
    "CycleTiming",
    "DreamCost",
    "OnlineCost",
    "total",
]


@dataclass(frozen=True)
class OnlineCost:
    """What one online rollout spent: §4's discovery cost.

    ``agent_calls`` is one per attempt, whatever came back from it — §4 counts
    "discovery-agent calls" and a call that raised was still made and still paid
    for. ``evaluations`` is what those calls bought: the attempts that reached
    the evaluator, which is fewer exactly when the agent raised. A node recorded
    with no score cost a call and measured nothing, and a cost report that could
    not say so would hide the one failure mode that is expensive.
    """

    agent_calls: int = 0
    evaluations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"agent_calls": self.agent_calls, "evaluations": self.evaluations}

    @classmethod
    def from_dict(cls, payload: Any) -> OnlineCost:
        return cls(
            agent_calls=int(payload["agent_calls"]),
            evaluations=int(payload["evaluations"]),
        )


@dataclass(frozen=True)
class DreamCost:
    """What one offline phase spent: model calls, replayed cells, revealed nodes.

    ``developer_calls`` is how many times the policy-development agent was asked
    for a revision, refusals included (``develop.Rejection``): an answer the
    harness could not use was still a call. ``cells`` is the ``(version, world)``
    replays the round ran and ``reveals`` is ``Σ N_i^m``, the nodes those replays
    revealed — the same ``N`` Equation 1 charges a version for.

    PAPER-GAP: §4 quantifies discovery cost as discovery-agent calls and prices
    dreaming at nothing — §2's "zero-execution-cost off-policy evaluations" — so
    the paper gives the offline half no unit at all. It is not free: each version
    costs a policy-development call, and each cell a pass over a recorded tree.
    We count those three quantities and keep them beside the online count rather
    than folding the two into one total the paper never defines, because a single
    number would have to price a replayed node against an agent call and the
    paper's whole claim is that they are not comparable. Revisit if the authors'
    implementation lands (see references/method.md).
    """

    developer_calls: int = 0
    cells: int = 0
    reveals: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "developer_calls": self.developer_calls,
            "cells": self.cells,
            "reveals": self.reveals,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DreamCost:
        return cls(
            developer_calls=int(payload["developer_calls"]),
            cells=int(payload["cells"]),
            reveals=int(payload["reveals"]),
        )


@dataclass(frozen=True)
class CycleCost:
    """One cycle's two halves, kept apart.

    The split is the structure and not a presentation choice: a reader — or a
    report — that wants the ratio the paper argues from has to reach one half or
    the other to get a number at all.
    """

    online: OnlineCost = OnlineCost()
    dreaming: DreamCost = DreamCost()

    @property
    def leverage(self) -> float | None:
        """Nodes replay revealed per discovery-agent call, or ``None`` with no calls.

        The ratio §2 claims is large: how much exploration one cycle re-examined
        for each attempt it actually paid an agent for.
        """
        if not self.online.agent_calls:
            return None
        return self.dreaming.reveals / self.online.agent_calls

    def to_dict(self) -> dict[str, Any]:
        return {"online": self.online.to_dict(), "dreaming": self.dreaming.to_dict()}

    @classmethod
    def from_dict(cls, payload: Any) -> CycleCost:
        return cls(
            online=OnlineCost.from_dict(payload["online"]),
            dreaming=DreamCost.from_dict(payload["dreaming"]),
        )


@dataclass(frozen=True)
class CycleTiming:
    """How long each half of one cycle took, on the machine that ran it.

    Deliberately not part of a cycle record: a record is a function of the run's
    seed and a clock is a function of the machine, so a wall clock in one would
    make two runs of the same seed disagree (working rule 5). It is written
    beside the record instead, and a report renders what it finds — a cycle whose
    timing is missing still has every count it spent.
    """

    online_seconds: float = 0.0
    dreaming_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "online_seconds": self.online_seconds,
            "dreaming_seconds": self.dreaming_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> CycleTiming:
        return cls(
            online_seconds=float(payload["online_seconds"]),
            dreaming_seconds=float(payload["dreaming_seconds"]),
        )


def total(costs: Iterable[CycleCost]) -> CycleCost:
    """What a run spent: its cycles' costs summed, half by half.

    Over no cycles this is a cost of zero rather than an error — a run that
    crashed before it finished one spent nothing this module can account for.
    """
    online = OnlineCost()
    dreaming = DreamCost()
    for cost in costs:
        online = OnlineCost(
            agent_calls=online.agent_calls + cost.online.agent_calls,
            evaluations=online.evaluations + cost.online.evaluations,
        )
        dreaming = DreamCost(
            developer_calls=dreaming.developer_calls + cost.dreaming.developer_calls,
            cells=dreaming.cells + cost.dreaming.cells,
            reveals=dreaming.reveals + cost.dreaming.reveals,
        )
    return CycleCost(online=online, dreaming=dreaming)
