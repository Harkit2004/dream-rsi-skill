"""A scripted, deterministic coding agent, so no test suite ever calls a real model.

The paper's discovery agent is an LLM and its transition is stochastic (§3):
the same starting workspace can yield different children. Nothing downstream of
this repo's adapters can be tested against that. :class:`FakeAgent` keeps the
shape — one attempt per call, resumed from a parent — and drops the randomness:
the candidate it returns is a pure function of the context it was given, so a
recorded rollout is reproducible (AGENTS.md rule 5).

It is deliberately inert: no network, no subprocess, no clock, and no state
across calls, so a batch of parallel workers may share one instance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from dream_rsi.adapters.agent import AgentContext, Artifact

__all__ = ["DEFAULT_SCRIPT", "FakeAgent"]

# Candidates for the toy task in ``toy_evaluator`` — each defines ``solve``, and
# they differ in length, which is what that task measures. Callers with a
# different task pass their own script.
DEFAULT_SCRIPT = (
    "def solve(values):\n    return sum(values)\n",
    (
        "def solve(values):\n    total = 0\n    for value in values:\n"
        "        total += value\n    return total\n"
    ),
    "def solve(values):\n    return sum(v for v in values if v)\n",
    (
        "def solve(values):\n    acc = 0\n    for value in values:\n"
        "        acc = acc + value\n    return acc\n"
    ),
)


@dataclass(frozen=True)
class FakeAgent:
    """Returns one of ``script``, picked by a fingerprint of the context."""

    script: tuple[str, ...] = DEFAULT_SCRIPT

    def __post_init__(self) -> None:
        if not self.script:
            raise ValueError("script must hold at least one candidate")

    def propose(self, context: AgentContext) -> Artifact:
        """Produce the candidate this context always produces."""
        digest = _fingerprint(context)
        body = self.script[int(digest, 16) % len(self.script)]
        return Artifact(
            content=f"# attempt {digest}\n{body}",
            proposal=(
                f"scripted attempt {digest}, resuming {context.parent.id} "
                f"after {len(context.observations)} inherited observation(s)"
            ),
        )


def _fingerprint(context: AgentContext) -> str:
    """A stable digest of everything the agent was given.

    Hashed rather than ``hash()``-ed because that is salted per process, and
    two runs of the same rollout have to agree.
    """
    payload = json.dumps(
        {
            "problem": context.problem,
            "workspace": str(context.workspace),
            "history": [node.to_dict() for node in context.history],
            "observations": list(context.observations),
            "seed": context.seed,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()
