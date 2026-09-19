"""The policy-development agent: revising policy code from replay feedback.

Step 4 of Dream-RSI's outer loop (§3). Dreaming (issue #12) scores ``M`` versions
over the history; this is where the versions after ``π_t^0`` come from: "the
policy-development agent examines the replay trajectories and scores of
``π_t^m``, together with feedback from earlier revisions, to identify successful
decisions and recurring failures. It then revises the executable policy code to
produce ``π_t^{m+1}``, which is evaluated on the same ``t`` replay worlds."

So :func:`develop` is that chain rather than a sweep: score the incumbent, hand
the agent what it did, take back a module, score it, hand *that* back. The
versions are developed in sequence because each one is written from the last
one's replay feedback — which is also why ``M`` is a serial chain and not a
search width (see ``dream.DEFAULT_VERSIONS``).

Three things this module owes its caller.

**Every candidate goes through the sandbox.** A revision is model-written Python
and the only thing built from one here is
:func:`~dream_rsi.sandbox.sandboxed_candidate`, so a version's code is compiled
and run in a child process under :class:`~dream_rsi.sandbox.SandboxLimits` and
never in this one. Nothing in this module compiles, imports or executes a
revision: :func:`validate_source` reads it with :mod:`ast`, which parses without
running, and everything else about a version is learned by replaying it behind
the boundary.

**A rejection is information.** Output that cannot be a policy — it does not
parse, or it defines no policy class — is not silently dropped and not quietly
scored as a failure either. It comes back in :attr:`DevelopmentReport.rejected`,
and the next attempt at the same version is given it, with the reason, so the
agent can see that it was refused rather than writing the same thing again.
Everything the check cannot see statically is caught behind the boundary instead
and arrives as a failed cell, whose error text is fed to the next revision the
same way (``dream.WorldReplay.error``).

**The prompt is one file.** ``prompts/policy_development.md``, rendered by
:meth:`RevisionContext.prompt`, so what the agent is told is reviewable in one
place instead of being assembled out of fragments scattered through here.

This module is not on the replay path — it is what produces the code that runs
there — and it still imports no adapter: a development agent satisfies
:class:`PolicyDeveloper` structurally, and the only one in the repository is the
scripted stub in ``tests/test_develop.py``.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Protocol, runtime_checkable

from dream_rsi.dream import (
    DEFAULT_VERSIONS,
    DreamConfig,
    DreamReport,
    ReplayWorld,
    VersionReport,
    WorldReplay,
    dream,
)
from dream_rsi.replay import ReplayRound
from dream_rsi.sandbox import (
    DEFAULT_LIMITS,
    DEFAULT_POLICY_NAME,
    SandboxLimits,
    sandboxed_candidate,
)

__all__ = [
    "DEFAULT_ATTEMPTS",
    "PROMPT_PATH",
    "DevelopedVersion",
    "DevelopmentReport",
    "PolicyDeveloper",
    "Rejection",
    "RevisionContext",
    "develop",
    "validate_source",
]

# PAPER-GAP: §B.2 prints the improvement prompt the authors used, but against
# their own harness — ``question.observed()``, ``probe_batch``, ``see.policy.api``
# and a ``pareto.reward`` objective — none of which is this repository's
# interface, and their policy module is not released. The prompt here re-expresses
# §B.2's own instructions (prefix-only decisions, the single beta knob, the batch
# and termination rules, the refusal to treat one failure as closure) against the
# interface in ``dream_rsi.policy`` and against Equation 1, which is what this
# harness actually scores. What is left unspecified either way — what the agent is
# shown of a version's replay, and what happens to output that cannot be used —
# is answered by :class:`RevisionContext` and :data:`DEFAULT_ATTEMPTS`. It is one
# file so the whole of it can be diffed against the authors' version when their
# code lands (see references/method.md).
PROMPT_PATH = Path(__file__).parent / "prompts" / "policy_development.md"

# PAPER-GAP: the paper never mentions a revision that cannot be used, so it says
# nothing about asking again. A model that answers with prose, a fenced block or
# a module defining no policy is common enough that dropping the version would
# lose an ``M`` to a formatting slip, so the agent is asked again with the
# refusal in front of it — three times in all, after which the round stops with
# the versions it has. Stopping rather than moving to the next slot, because the
# next slot would be the same base, the same feedback and the same refusal.
# Revisit if the authors' implementation lands (see references/method.md).
DEFAULT_ATTEMPTS = 3


@dataclass(frozen=True)
class Rejection:
    """One output the harness refused, and why.

    Kept whole rather than summarised: what the agent wrote is half of what a
    reader — or the next attempt — needs to see to understand the refusal.
    """

    source: str
    reason: str


@dataclass(frozen=True)
class RevisionContext:
    """What the development agent is given to produce ``π_t^{m+1}`` (§3).

    ``source`` is ``π_t^m``'s code and ``report`` is what it did: its ``V^m``,
    its ``V_i^m`` on each world, the trajectory each of those was computed from,
    and the error text of any world it raised on. ``history`` is the earlier
    versions of this round — §3's "feedback from earlier revisions" — and
    ``rejected`` is the output already refused for *this* version, which is the
    feedback the paper does not have because its agent's output is assumed
    usable.
    """

    source: str
    report: VersionReport
    history: tuple[VersionReport, ...] = ()
    rejected: tuple[Rejection, ...] = ()

    def prompt(self) -> str:
        """This context rendered into :data:`PROMPT_PATH`.

        ``Template.substitute`` and not ``safe_substitute``: a placeholder in the
        prompt file that nothing here fills is a mistake in one of the two, and
        it should stop the round rather than reach a model verbatim.
        """
        return Template(PROMPT_PATH.read_text(encoding="utf-8")).substitute(
            policy_name=DEFAULT_POLICY_NAME,
            version=self.report.name,
            score=_number(self.report.score),
            source=self.source.strip("\n"),
            replays=_replays_text(self.report),
            history=_history_text(self.history),
            rejected=_rejected_text(self.rejected),
        )


@runtime_checkable
class PolicyDeveloper(Protocol):
    """What an adapter must supply to be the policy-development agent.

    One method, for the reason ``adapters.agent.CodingAgent`` has one: the claim
    the paper is making is that improvement lives in the orchestration layer, and
    an interface with nothing to reach around is what keeps a provider out of
    everything downstream of it.
    """

    def revise(self, context: RevisionContext) -> str:
        """The next version's complete module source, revised from ``context``."""
        ...


@dataclass(frozen=True)
class DevelopedVersion:
    """One version of a round: the code, and what replaying it scored.

    The code is what the next revision starts from and what an accepted version
    is deployed as (issue #15 selects one of these), so a round has to hand back
    the source alongside the number it earned.
    """

    name: str
    source: str
    report: VersionReport


@dataclass(frozen=True)
class DevelopmentReport:
    """One offline phase: the versions it developed and what it refused.

    ``comparison`` is the same grid :func:`~dream_rsi.dream.dream` reports, over
    exactly these versions, so ranking, the comparison table and the JSON a
    reader inspects the round from are the dreaming harness's and not a second
    implementation of them.
    """

    versions: tuple[DevelopedVersion, ...]
    rejected: tuple[Rejection, ...]
    comparison: DreamReport


def validate_source(source: str, *, policy_name: str = DEFAULT_POLICY_NAME) -> str | None:
    """Why this output cannot be a candidate, or ``None`` if it can.

    Two refusals, both static — :func:`ast.parse` builds a tree and runs nothing:
    output that is not Python, and output that defines no policy class for the
    sandbox to instantiate. §B.2's contract is "keep ``NAME = "OptimalPolicy"``
    and implement ``class OptimalPolicy(...)``", so a module may rename the class
    by assigning ``NAME`` and this follows it there.

    Everything else a policy owes — that the class is constructible, that its
    instances answer ``select``, that a decision returns node ids — cannot be
    known without running the code, and running it is the sandbox's job
    (``_sandbox_child._load_policy`` raises where a version fails any of them).
    This check exists to turn the two faults that *are* visible into feedback
    before ``M`` child processes are spent discovering them.
    """
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        # ValueError as well: ``ast.parse`` raises it, not SyntaxError, for text
        # it cannot even scan — a null byte in the middle of an answer — and to
        # a round that is the same refusal rather than a crash.
        return f"the revision does not parse as Python: {exc}"
    name = _declared_name(module, policy_name)
    if name is None:
        # ``NAME`` is computed, so which class the sandbox will look for is not
        # a static fact. Let it decide rather than refuse a legal version.
        return None
    if not any(isinstance(node, ast.ClassDef) and node.name == name for node in module.body):
        return (
            f"the revision defines no class {name} at module level, so there is no "
            f'policy to run; §B.2: keep NAME = "{name}" and implement class {name}(...)'
        )
    return None


def develop(
    source: str,
    worlds: Sequence[ReplayWorld],
    developer: PolicyDeveloper,
    *,
    versions: int = DEFAULT_VERSIONS,
    attempts: int = DEFAULT_ATTEMPTS,
    config: DreamConfig | None = None,
    limits: SandboxLimits = DEFAULT_LIMITS,
    scratch_root: str | Path | None = None,
) -> DevelopmentReport:
    """Develop and score ``M`` policy versions over the history ``ℋ_t`` (§3).

    ``source`` is the incumbent ``π_t^0`` — §3 begins the offline phase by
    evaluating it — and ``worlds`` is the history every version is replayed over.
    Each later version is what ``developer`` wrote from the previous one's replay
    feedback, sandboxed under ``limits`` and scored on the same worlds under the
    same ``config``, because versions offered different widths or weights were
    not compared with each other.

    The round comes back with however many versions it got: ``versions`` of them
    when the agent's output is usable, fewer when a version could not be filled
    in ``attempts`` tries, with every refusal in
    :attr:`DevelopmentReport.rejected`.
    """
    if versions < 1:
        # §3: "M ≥ 1 policy versions", the first of which is the incumbent.
        raise ValueError(f"a development round needs at least one version, got {versions}")
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    config = DreamConfig() if config is None else config

    developed: list[DevelopedVersion] = []
    rejected: list[Rejection] = []
    revision: str | None = source
    while revision is not None:
        name = f"v{len(developed)}"
        developed.append(
            DevelopedVersion(
                name=name,
                source=revision,
                report=_score(
                    name,
                    revision,
                    worlds,
                    config=config,
                    limits=limits,
                    scratch_root=scratch_root,
                ),
            )
        )
        if len(developed) == versions:
            break
        revision, refused = _revise(developer, tuple(developed), attempts=attempts)
        rejected.extend(refused)

    return DevelopmentReport(
        versions=tuple(developed),
        rejected=tuple(rejected),
        comparison=DreamReport(
            versions=tuple(version.report for version in developed),
            worlds=tuple(world.name for world in worlds),
            config=config,
        ),
    )


def _score(
    name: str,
    source: str,
    worlds: Sequence[ReplayWorld],
    *,
    config: DreamConfig,
    limits: SandboxLimits,
    scratch_root: str | Path | None,
) -> VersionReport:
    """Replay one version over the whole history, behind the sandbox boundary.

    The one place a version's source becomes something that runs, and it builds
    a :func:`~dream_rsi.sandbox.sandboxed_candidate` to do it — issue #14: "every
    generated candidate goes through #13. No exceptions." A version that cannot
    be loaded, oversteps its limits or answers with something that is not a batch
    fails there, and :func:`~dream_rsi.dream.dream` records that as this version's
    failed cells rather than ending the round.
    """
    candidate = sandboxed_candidate(
        name,
        source,
        limits=limits,
        scratch_root=scratch_root,
    )
    return dream([candidate], worlds, config=config).versions[0]


def _revise(
    developer: PolicyDeveloper,
    developed: tuple[DevelopedVersion, ...],
    *,
    attempts: int,
) -> tuple[str | None, tuple[Rejection, ...]]:
    """Ask for the next version until it is usable, or until the tries run out.

    Every attempt is given the refusals the earlier ones earned, so an agent that
    was refused can see it; an agent that cannot produce a usable module ends the
    round rather than being asked a fourth time (see :data:`DEFAULT_ATTEMPTS`).
    """
    latest = developed[-1]
    refused: list[Rejection] = []
    for _ in range(attempts):
        context = RevisionContext(
            source=latest.source,
            report=latest.report,
            history=tuple(version.report for version in developed[:-1]),
            rejected=tuple(refused),
        )
        revision = developer.revise(context)
        reason = validate_source(revision)
        if reason is None:
            return revision, tuple(refused)
        refused.append(Rejection(source=revision, reason=reason))
    return None, tuple(refused)


def _declared_name(module: ast.Module, default: str) -> str | None:
    """The class the sandbox will look for, or ``None`` where that is not static.

    §B.2's ``NAME = "OptimalPolicy"`` is the paper's hook for renaming the policy
    class, and ``_sandbox_child._load_policy`` reads it. A module that computes
    it instead is legal and unreadable from here.
    """
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "NAME" for target in targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
        return None
    return default


def _replays_text(report: VersionReport) -> str:
    """What the version did on each world, round by round (§3's trajectories)."""
    return "\n".join(_replay_text(replay) for replay in report.replays)


def _replay_text(replay: WorldReplay) -> str:
    """One world: the score, the shape of the replay, and every decision in it."""
    if replay.result is None:
        return f"- {replay.world}: no score — the version raised {replay.error}"
    result = replay.result
    scores = {point.node_id: point.score for point in result.curve}
    lines = [
        f"- {replay.world}: V_i={_number(replay.score)} "
        f"attainment={_number(result.attainment)} revealed={result.revealed} "
        f"rounds={result.round_count} stopped={result.stop_reason}"
    ]
    lines += [
        f"    round {round_.index}: "
        + "; ".join(
            _selection_text(round_, index, scores) for index in range(len(round_.selected))
        )
        for round_ in result.rounds
    ]
    return "\n".join(lines)


def _selection_text(round_: ReplayRound, index: int, scores: dict[str, float | None]) -> str:
    """One selection of one round: what was opened, and what that revealed.

    Indexed defensively because ``ReplayRound.observations`` defaults to empty:
    a trajectory read back from a log written before they were recorded still
    has its decisions, and they are the half of the feedback that matters here.
    """
    selected = round_.selected[index]
    node_id = round_.revealed[index] if index < len(round_.revealed) else None
    if node_id is None:
        return f"{selected} -> nothing left to reveal"
    seen = round_.observations[index] if index < len(round_.observations) else ()
    text = f"{selected} -> {node_id} scored {_number(scores.get(node_id))}"
    return f"{text} [{'; '.join(seen)}]" if seen else text


def _history_text(history: tuple[VersionReport, ...]) -> str:
    """The earlier versions of this round: §3's "feedback from earlier revisions"."""
    if not history:
        return "(none — this is the first revision of the round)"
    return "\n".join(_version_text(version) for version in history)


def _version_text(version: VersionReport) -> str:
    """One earlier version: what it averaged, and where it fell over."""
    text = f"- {version.name}: average replay score {_number(version.score)}"
    failed = ", ".join(replay.world for replay in version.failures)
    return f"{text}, failed on {failed}" if failed else text


def _rejected_text(rejected: tuple[Rejection, ...]) -> str:
    """Output already refused for this version, whole, so it is not written again."""
    if not rejected:
        return "(none)"
    return "\n\n".join(
        f"- refused: {rejection.reason}\n\n```python\n{rejection.source.strip()}\n```"
        for rejection in rejected
    )


def _number(value: float | None) -> str:
    """A score as the prompt states it; ``none`` where there is not one."""
    return "none" if value is None else f"{value:.4f}"
