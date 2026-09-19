"""A scripted, deterministic policy-development agent, so no run calls a model.

The paper's policy-development agent is an LLM that rewrites the exploration
policy from replay feedback (§3, §B.2). :class:`FakeDeveloper` keeps the shape —
one revision per call, handed the previous version's feedback — and drops the
model: it answers from a fixed list, so the loop in :mod:`dream_rsi.run` is
runnable end to end without a provider, which is what issue #16's "done when"
asks for.

It is deliberately inert: no network, no subprocess, no clock, and no state
across calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from dream_rsi.develop import RevisionContext

__all__ = ["DEFAULT_SCRIPT", "FakeDeveloper"]


def _beta_revision(beta: float) -> str:
    """The baseline at a different baked-in ``beta`` — §B.2's single knob.

    §B.2 has a version expose exactly one behavioural parameter and "route every
    behavioral threshold through one ``_schedule(beta) -> dict``", so turning it
    is the smallest revision that is still a real one: the policies in
    :mod:`dream_rsi.policy` read ``beta`` off their config and change how long
    they stay with a branch.
    """
    return (
        "from dream_rsi.policy import GreedyBestFirstPolicy\n"
        "\n"
        "\n"
        "class OptimalPolicy(GreedyBestFirstPolicy):\n"
        "    def __init__(self, config=None):\n"
        f"        super().__init__({{'beta': {beta}}})\n"
    )


# Revisions for the toy task, differing in the one knob §B.2 gives a version.
# Callers with something else to try pass their own script.
DEFAULT_SCRIPT = (
    _beta_revision(0.5),
    _beta_revision(1.5),
    _beta_revision(2.5),
)


@dataclass(frozen=True)
class FakeDeveloper:
    """Answers with ``script[n]``, where ``n`` counts the revisions asked for."""

    script: tuple[str, ...] = DEFAULT_SCRIPT

    def __post_init__(self) -> None:
        if not self.script:
            raise ValueError("script must hold at least one revision")

    def revise(self, context: RevisionContext) -> str:
        """Produce the revision this context always produces.

        The position is read off the context rather than kept here, so the agent
        stays stateless: ``history`` is the versions already developed this round
        and ``rejected`` the output already refused for this one, so a refusal
        moves on to the next entry instead of offering the same text again.
        """
        asked = len(context.history) + len(context.rejected)
        return self.script[asked % len(self.script)]
