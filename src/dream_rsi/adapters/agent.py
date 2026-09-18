"""The coding agent adapter contract.

Dream-RSI's claim is that all improvement lives in the orchestration layer:
"Only the exploration-policy code changes; the underlying models, evaluator, and
execution interfaces remain fixed" (§3). That holds only if the agent sits
behind an interface narrow enough that there is nothing to reach around — hence
one method, :meth:`CodingAgent.propose`, and no way to set weights, prompt
internals, or decoding beyond what the adapter's own constructor chooses to
expose to a normal caller.

The paper's agent is prompted with the task and the discovery history and writes
two files: ``proposal.md``, the mechanism and the evidence for it, and
``$eval_program``, the candidate itself (§B.1). The node it produces "resumes
the parent's saved workspace and uses its accumulated observations as context"
(§3) — that inheritance is what :class:`AgentContext` carries.

The paper used Gemini models through a CLI. Nothing here names a provider: this
module defines an interface and never an implementation, and the only adapter
shipped in the repo is the deterministic
:class:`~dream_rsi.adapters.fake_agent.FakeAgent` the test suite runs against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from dream_rsi.tree import Node

__all__ = ["AgentContext", "Artifact", "CodingAgent"]


@dataclass(frozen=True)
class Artifact:
    """One attempt produced by the discovery agent.

    ``content`` is the candidate program: the string a
    :class:`~dream_rsi.adapters.evaluator.TaskEvaluator` is handed and the tree
    records as ``Node.artifact``.
    """

    content: str

    # PAPER-GAP: §3 says a node records "the generated artifact", singular, while
    # the exploration prompt in §B.1 has the agent write both a proposal and the
    # program; the paper never says which one the node stores. We keep them
    # apart and treat the program as the artifact, because that is the string
    # the evaluator measures — the proposal is rationale, recorded alongside it
    # so a later attempt can read why this one was tried. Revisit if the
    # authors' implementation lands (see references/method.md).
    proposal: str = ""


@dataclass(frozen=True)
class AgentContext:
    """What the agent is given to produce one attempt.

    ``history`` is the inherited chain of attempts, root first, ending at the
    primary parent this attempt continues from; ``workspace`` is that parent's
    saved workspace as resumed for this attempt, and ``observations`` are the
    observations accumulated along the chain (§3).
    """

    problem: str
    workspace: Path
    history: tuple[Node, ...] = field(default_factory=tuple)
    observations: tuple[str, ...] = ()

    # PAPER-GAP: the paper treats generation as stochastic (§3, online rollout)
    # and never says how that randomness is controlled. We put an explicit seed
    # on the request, because a rollout that cannot be reproduced cannot be
    # replayed (AGENTS.md rule 5); each adapter maps it onto whatever its
    # provider exposes, and ``None`` means "the provider's default".
    seed: int | None = None

    def __post_init__(self) -> None:
        # §3: every attempt begins at exactly one primary parent — the root, or
        # a previously created node. A context with no parent names no starting
        # workspace and no inherited observations, so there is nothing to resume.
        if not self.history:
            raise ValueError("history must name the parent this attempt resumes from")

    @property
    def parent(self) -> Node:
        """The primary parent: the node whose workspace and observations are inherited."""
        return self.history[-1]


@runtime_checkable
class CodingAgent(Protocol):
    """What an adapter must supply for the orchestrator to run a rollout.

    Implementations are structural: anything with this one method satisfies the
    protocol, no base class and no registration. Adapters wrap agents; they do
    not change them (AGENTS.md rule 6).
    """

    def propose(self, context: AgentContext) -> Artifact:
        """Produce one candidate, resuming from ``context.parent``."""
        ...
