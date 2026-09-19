"""The dreaming harness: evaluate ``M`` policy versions over the whole history.

Step 3 of Dream-RSI's outer loop (§3, *Offline evaluation*). The history
``ℋ_t`` is fixed, and the method "constructs and evaluates ``M ≥ 1`` policy
versions ``π_t^0, …, π_t^{M-1}``, starting with ``π_t^0 = π_t``. Each version is
evaluated separately on every historical tree ``𝒯_i``". This module is that
grid: every candidate against every recorded world, Equation 1 per cell, and one
report over the whole thing.

The aggregate is the paper's own: "the evaluation score of policy version
``π_t^m`` is its average replay score across the fixed history,
``V^m = (1/t) Σ_{i=1}^{t} V_i^m``". A mean, not a min and not a sum — which is
what :func:`dream` computes and ``tests/test_dream.py`` pins, because it is the
number selection reads (issue #15).

Two things the harness owes its callers beyond the arithmetic.

**Nothing may depend on scheduling.** The ``(version, world)`` cells are
independent — replay generates nothing, and a world is immutable and reusable —
so :class:`DreamConfig` can run them several at a time. What comes back is
assembled in the order the caller gave, never the order the cells finished, so
one report is a function of its inputs alone (working rule 5). ``workers`` is
therefore not part of the record: it cannot change a number in it, and
``tests/test_dream.py`` asserts the serialised report is identical at every
setting.

**A candidate that raises is one cell's outcome, not the round's.** Candidates
are LLM-written Python (issues #13, #14), so a version that throws is a normal
event rather than a bug in the harness. It is recorded as a failure against the
world it fell over on, the rest of the sweep runs, and the version scores no
aggregate at all — see :class:`VersionReport`. Bounding what such code may *do*
(time, memory, the network, the filesystem) belongs to
:mod:`dream_rsi.sandbox`, which raises where a candidate oversteps; this module
only contains the exception.

**Selection reads a grid, it does not build one.** :func:`select` is step 5 —
"the next online policy is selected from all ``M`` evaluated versions" — and it
takes one :class:`DreamReport`, because that is what makes the incumbent's floor
mean anything: every version in a report was replayed over the same history
under the same ``DreamConfig``, ``π_t^0`` among them. It re-derives nothing and
accepts no score from anywhere else, so the ``V^0`` it holds a candidate to is
this round's, not the one the incumbent earned when the history was shorter.

Dreaming is replay, so this module reaches the simulator, the objective and the
tree, and no adapter: nothing on the replay path may call a discovery agent or
an evaluator, and ``tests/test_dream.py`` asserts the import stays out.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from dream_rsi.replay import DEFAULT_SEED, ReplayPolicy, ReplaySimulator, SimResult
from dream_rsi.scoring import DEFAULT_WEIGHTS, ReplayWeights, replay_score

__all__ = [
    "DEFAULT_VERSIONS",
    "DreamConfig",
    "DreamReport",
    "PolicyCandidate",
    "ReplayWorld",
    "Selection",
    "VersionReport",
    "WorldReplay",
    "dream",
    "select",
]

# PAPER-GAP: §3 requires only "M ≥ 1" policy versions and never says what M is;
# §4's ablations report no sweep over it either. M is the number of revisions the
# policy-development agent makes in one offline phase (issue #14), so each one
# costs a model call and a full pass over the history, and the paper's own
# structure bounds what it can buy: versions are developed in sequence, each from
# the last one's replay feedback, so a large M is a long serial chain inside one
# outer iteration rather than a wider search. We default to 8 — enough revisions
# for the development agent to act on feedback and back out of a bad one, cheap
# enough that a dreaming round costs single-digit model calls, and the whole grid
# over the fixtures in ``tests/fixtures/trees/`` replays in well under a second.
# It is the caller's number: :func:`dream` scores whatever candidate list it is
# handed. Revisit if the authors' implementation lands (see references/method.md).
DEFAULT_VERSIONS = 8


@dataclass(frozen=True)
class PolicyCandidate:
    """One policy version to evaluate: ``π_t^m``, and a name to report it under.

    ``factory`` is called once per world and must hand back a *fresh* policy.
    Not one instance reused, for two reasons: a policy carries per-rollout state
    (§3: replay "resets the policy's per-rollout state" before each policy-world
    pair), and the cells of one version's row may be replayed at the same time,
    where a shared instance would have two worlds writing over each other's
    state. It is also where model-written code enters:
    :func:`dream_rsi.sandbox.sandboxed_candidate` builds exactly this, with a
    factory that starts a child process per world and takes the version's
    decisions there. That is the only way source becomes a candidate — a policy
    an LLM wrote is never called in this process.
    """

    name: str
    factory: Callable[[], ReplayPolicy]


@dataclass(frozen=True)
class ReplayWorld:
    """One recorded world in the history: ``𝒯_i``, and a name to report it under.

    The name is how a per-world score is read back in the report; the fixtures
    use their directory name, and a live pool (issue #17) would use whatever
    identifies the rollout that recorded it.
    """

    name: str
    simulator: ReplaySimulator


@dataclass(frozen=True)
class DreamConfig:
    """How one dreaming round replays its grid.

    ``width`` is the paper's ``W`` offered to the policy inside each replay and
    ``max_rounds`` is ``K₂``; both are passed straight to
    :meth:`~dream_rsi.replay.ReplaySimulator.replay`, whose own defaults apply
    when they are left alone. A dreaming round should offer the ``W`` the online
    rollout will actually run at, since Equation 1's parallelism bonus is what
    makes batching worth anything to a version.

    ``workers`` is not that: it is how many ``(version, world)`` cells this
    harness replays at once, a property of the machine and of nothing in the
    paper. It cannot change a reported number — see the module docstring — so it
    is deliberately absent from :meth:`DreamReport.to_dict`.
    """

    width: int = 1
    max_rounds: int | None = None
    seed: int = DEFAULT_SEED
    weights: ReplayWeights = DEFAULT_WEIGHTS
    workers: int = 1

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"width must be at least 1 (§3: W ≥ 1), got {self.width}")
        if self.max_rounds is not None and self.max_rounds < 0:
            raise ValueError(f"max_rounds must not be negative, got {self.max_rounds}")
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}")


@dataclass(frozen=True)
class WorldReplay:
    """One cell of the grid: what version ``m`` did on world ``i``.

    ``score`` is ``V_i^m`` and ``result`` the trajectory it was computed from,
    or — where the version raised — ``score`` is ``None``, ``result`` is ``None``
    and ``error`` says what was raised.

    ``score`` is ``-inf`` for a replay that revealed nothing carrying an ``s_v``:
    that is what :func:`~dream_rsi.scoring.replay_score` makes of no attainment,
    so that no attainment loses to any attainment. JSON has no token for it, so
    :meth:`DreamReport.to_dict` writes it as ``null`` and ``error`` is what tells
    the two ``null``s apart.
    """

    world: str
    score: float | None
    result: SimResult | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "world": self.world,
            "score": _json_score(self.score),
            "error": self.error,
            "result": None if self.result is None else self.result.to_dict(),
        }


@dataclass(frozen=True)
class VersionReport:
    """One row of the grid: version ``m`` over the whole history.

    ``score`` is §3's ``V^m = (1/t) Σ_i V_i^m``, the average replay score across
    the fixed history — or ``None`` where any world failed.

    PAPER-GAP: §3's ``V^m`` assumes every version scores on every tree and says
    nothing about one that cannot be evaluated on part of the history. We leave
    the aggregate undefined rather than average the worlds it survived: a partial
    mean is not ``V^m``, and it would flatter exactly the version that fell over
    on the worlds that were hardest to navigate — a candidate could raise its
    own score by crashing. A version with no aggregate cannot win a selection
    (issue #15), which is the same treatment the incumbent floor gives anything
    that fails to beat ``V^0``. Revisit if the authors' implementation lands (see
    references/method.md).
    """

    name: str
    score: float | None
    replays: tuple[WorldReplay, ...]

    @property
    def failures(self) -> tuple[WorldReplay, ...]:
        """The worlds this version raised on, in history order."""
        return tuple(replay for replay in self.replays if replay.error is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": _json_score(self.score),
            "failed_worlds": [replay.world for replay in self.failures],
            "replays": [replay.to_dict() for replay in self.replays],
        }


@dataclass(frozen=True)
class DreamReport:
    """One dreaming round's comparison of ``M`` versions over ``t`` worlds.

    Both audiences of issue #12 read this: :meth:`to_text` is the table a human
    compares versions in, and :meth:`to_dict` / :meth:`to_json` is what the
    policy-development agent is given, carrying each cell's full trajectory
    because that is what §3 has it examine — "the policy-development agent
    examines the replay trajectories and scores of ``π_t^m``".

    It is a report rather than a stored format: nothing reads one back, so unlike
    ``tree.DiscoveryTree`` and ``replay.SimResult`` there is no loader and no
    schema version to check against. A round that needs to be re-read after the
    fact keeps the ``SimResult`` logs, which have both.
    """

    versions: tuple[VersionReport, ...]
    worlds: tuple[str, ...]
    config: DreamConfig

    @property
    def ranking(self) -> tuple[str, ...]:
        """Version names, best average replay score first.

        PAPER-GAP: §3 selects ``m⋆ ∈ argmax_m V^m`` and gives no tie-break, and
        no ``V^m`` at all for a version that failed. This is the report's display
        order, not a selection (issue #15): a tie keeps the order the versions
        were given in, which is the order they were developed and so puts the
        incumbent ``π_t^0`` first, and versions with no aggregate come last in
        that same order. Revisit if the authors' implementation lands (see
        references/method.md).
        """
        ordered = sorted(enumerate(self.versions), key=_rank_key)
        return tuple(version.name for _, version in ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": {
                "cost_weight": self.config.weights.cost,
                "parallelism_weight": self.config.weights.parallelism,
            },
            "replay": {
                "width": self.config.width,
                "max_rounds": self.config.max_rounds,
                "seed": self.config.seed,
            },
            "worlds": list(self.worlds),
            "ranking": list(self.ranking),
            "versions": [version.to_dict() for version in self.versions],
        }

    def to_json(self) -> str:
        """The report as JSON, byte-identical for the same candidates and worlds.

        ``allow_nan=False`` for the reason ``replay.SimResult.to_json`` gives:
        Python writes a non-finite number as a bare token no other JSON reader
        accepts. Scores are already put through :func:`_json_score`, so the only
        way to trip it is a bug here rather than a legitimate ``-inf``.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    def to_text(self) -> str:
        """The comparison table, for whoever is reading the round afterwards.

        One row per version, best first, with its aggregate and its score on each
        world; a cell the version raised on reads ``failed``, and every failure
        is spelled out underneath, because a version named without its error is
        how someone comes to look for a version that is not there.
        """
        by_name = {version.name: version for version in self.versions}
        rows = [
            [
                "-" if by_name[name].score is None else str(rank + 1),
                name,
                _cell(by_name[name].score),
                *(_cell(replay.score, replay.error) for replay in by_name[name].replays),
            ]
            for rank, name in enumerate(self.ranking)
        ]
        header = ["rank", "version", "V^m", *self.worlds]
        widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]

        weights = self.config.weights
        # ``None`` is not "no cap": it is replay's own K₂ default, which depends
        # on the world (see ``ReplaySimulator.replay``), so it has no one number.
        cap = "default" if self.config.max_rounds is None else self.config.max_rounds
        lines = [
            f"dreaming round: {len(self.versions)} version(s) x {len(self.worlds)} world(s)",
            (
                f"objective: β₁={weights.cost} β₂={weights.parallelism} | "
                f"W={self.config.width} | K₂={cap} | seed={self.config.seed}"
            ),
            "",
            *(
                "  ".join(cell.rjust(width) for cell, width in zip(row, widths))
                for row in (header, *rows)
            ),
        ]
        failures = [
            f"  {version.name} on {replay.world}: {replay.error}"
            for version in self.versions
            for replay in version.failures
        ]
        if failures:
            lines += ["", "failures:", *failures]
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Selection:
    """Which version is deployed next, and the line that says why (§3, issue #15).

    ``winner`` is ``m⋆`` and ``incumbent`` is ``π_t^0``; :attr:`retained` is the
    two being the same version, which §3 makes a normal outcome rather than a
    failed round — "because the candidate set includes the current policy, this
    selection satisfies ``V^{m⋆} ≥ V^0``". A round that improves nothing keeps
    what it has, and the history it collected is still what the next round
    dreams over.

    ``score`` and ``incumbent_score`` are those two versions' ``V^m`` on the
    round that was selected from, so the floor can be read back afterwards
    rather than taken on trust; :attr:`margin` is the improvement they bound.
    The names are how the winner is mapped back to code — a
    ``develop.DevelopedVersion`` carries the source under the same name.
    """

    winner: str
    incumbent: str
    score: float | None
    incumbent_score: float | None
    rationale: str

    @property
    def retained(self) -> bool:
        """Whether ``π_{t+1} = π_t``: no version beat the incumbent (§3)."""
        return self.winner == self.incumbent

    @property
    def margin(self) -> float | None:
        """``V^{m⋆} − V^0``: zero on a retention, ``None`` where either has no score."""
        if self.winner == self.incumbent:
            # Not a subtraction: where the incumbent won on ``-inf`` both sides
            # are the same infinity and the difference of those is not a number.
            return None if self.score is None else 0.0
        if self.score is None or self.incumbent_score is None:
            return None
        return self.score - self.incumbent_score

    def to_dict(self) -> dict[str, Any]:
        """The selection as the run report (issue #18) carries it."""
        return {
            "winner": self.winner,
            "incumbent": self.incumbent,
            "retained": self.retained,
            "score": _json_score(self.score),
            "incumbent_score": _json_score(self.incumbent_score),
            "margin": _json_score(self.margin),
            "rationale": self.rationale,
        }


def dream(
    candidates: Sequence[PolicyCandidate],
    worlds: Sequence[ReplayWorld],
    *,
    config: DreamConfig | None = None,
) -> DreamReport:
    """Replay every candidate over every world and report the comparison (§3).

    ``candidates`` are the ``M`` versions, in the order they were developed —
    ``π_t^0 = π_t`` first, as §3 has it — and ``worlds`` is the history ``ℋ_t``.
    Every cell is replayed under ``config``: the same ``W``, ``K₂``, seed and
    ``β`` for all of them, because a version that was offered a different width
    from its rivals was not compared with them.
    """
    config = DreamConfig() if config is None else config
    _check_unique("version name", (candidate.name for candidate in candidates))
    _check_unique("world name", (world.name for world in worlds))
    if not candidates:
        # §3: "M ≥ 1 policy versions".
        raise ValueError("a dreaming round needs at least one policy version to evaluate")
    if not worlds:
        # §3's V^m averages over t ≥ 1 trees; over none it is not a number.
        raise ValueError("a dreaming round needs at least one replay world to evaluate on")

    # One flat list of cells in row-major order — versions outermost — so what
    # comes back lines up with the grid however it was scheduled.
    cells = [(candidate, world) for candidate in candidates for world in worlds]
    if config.workers == 1:
        replayed = [_replay(candidate, world, config) for candidate, world in cells]
    else:
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            # ``map`` yields in submission order, not completion order, which is
            # what keeps the report a function of its inputs (working rule 5).
            replayed = list(pool.map(lambda cell: _replay(*cell, config), cells))

    # One row of the grid per version, ``len(worlds)`` cells wide.
    stride = len(worlds)
    versions = tuple(
        _aggregate(candidate.name, tuple(replayed[index * stride : (index + 1) * stride]))
        for index, candidate in enumerate(candidates)
    )
    return DreamReport(
        versions=versions,
        worlds=tuple(world.name for world in worlds),
        config=config,
    )


def select(report: DreamReport, *, incumbent: str | None = None) -> Selection:
    """Name ``π_{t+1}`` from one dreaming round, no worse than ``π_t`` (§3).

    §3's step 5: "the next online policy is selected from all ``M`` evaluated
    versions as ``π_{t+1} = π_t^{m⋆}``, where ``m⋆ ∈ argmax_m V^m``. Because the
    candidate set includes the current policy, this selection satisfies
    ``V^{m⋆} ≥ V^0``." The floor is therefore not a rule applied on top of the
    argmax — it *is* the argmax, as long as the incumbent is one of the versions
    being compared and every version was scored on the same history under the
    same conditions. Both are checked here rather than assumed: a grid is one
    pool and one :class:`DreamConfig` by construction (:func:`dream`), but
    ``develop.DevelopmentReport.comparison`` assembles one out of per-version
    reports, so a row scored over an earlier, smaller pool is an assembly away
    — and comparing today's candidates against yesterday's ``V^0`` would leave
    the guarantee holding over a history nobody replays any more.

    ``incumbent`` names ``π_t^0``; it defaults to the first version of the
    report, which is the order §3 develops them in and the order
    :func:`~dream_rsi.develop.develop` produces them in.
    """
    by_name = {version.name: version for version in report.versions}
    name = report.versions[0].name if incumbent is None else incumbent
    if name not in by_name:
        raise ValueError(
            f"incumbent {name!r} is not one of the versions this round scored "
            f"({', '.join(by_name)}); §3's floor needs π_t^0 among them"
        )
    for version in report.versions:
        scored = tuple(replay.world for replay in version.replays)
        if scored != report.worlds:
            raise ValueError(
                f"version {version.name!r} was scored on {list(scored)}, not on this round's "
                f"pool {list(report.worlds)}; a V^m from another history is not comparable"
            )

    floor = by_name[name]
    best = _best(report.versions)
    # PAPER-GAP: §3's argmax has no tie-break and assumes every version has a
    # V^m. Two choices follow. A version that only *matches* V^0 does not
    # displace the incumbent — the floor is read strictly, since swapping in an
    # equally-scoring policy spends an online rollout to learn nothing — and
    # among versions that tie above it the earliest developed wins, which is the
    # one its rivals were revised from and the one this report already ranks
    # first. Where the incumbent itself scored no aggregate there is no V^0 to
    # be no worse than, so the best scoring version wins and the round is an
    # improvement by default; retaining a policy that raised on the history
    # would keep every later round stuck on it. Revisit if the authors'
    # implementation lands (see references/method.md).
    if best is None or (floor.score is not None and best.score <= floor.score):
        winner = floor
    else:
        winner = best
    return Selection(
        winner=winner.name,
        incumbent=floor.name,
        score=winner.score,
        incumbent_score=floor.score,
        rationale=_rationale(report, winner, floor),
    )


def _best(versions: Sequence[VersionReport]) -> VersionReport | None:
    """The highest-scoring version, earliest first on a tie, or ``None`` if none scored.

    A version with no aggregate is not a low score, it is no score: it cannot
    win, however badly the ones that ran did (:class:`VersionReport`).
    """
    scored = [version for version in versions if version.score is not None]
    if not scored:
        return None
    # ``max`` keeps the first of equal keys, which is the tie-break this module
    # documents: the earliest version developed, the one its rivals came from.
    return max(scored, key=lambda version: version.score)


def _rationale(report: DreamReport, winner: VersionReport, floor: VersionReport) -> str:
    """One line saying why this version is being deployed, for the run log (issue #18)."""
    grid = f"{len(report.versions)} version(s) over {len(report.worlds)} world(s)"
    if winner.name != floor.name:
        against = (
            f"beats V^0={_number(floor.score)} by {_gap(winner.score, floor.score)}"
            if floor.score is not None
            else f"{floor.name} scored no aggregate, so there was no floor to hold"
        )
        return (
            f"selected {winner.name} at V^m={_number(winner.score)} in place of the "
            f"incumbent {floor.name}: {against}; {grid}"
        )
    if floor.score is None:
        return (
            f"retained the incumbent {floor.name}: no version of this round scored an "
            f"aggregate, {floor.name} included; {grid}"
        )
    runners = [version for version in report.versions if version.name != floor.name]
    runner_up = _best(runners)
    if runner_up is None:
        best = "no other version scored an aggregate"
    else:
        best = (
            f"best other version {runner_up.name} scored {_number(runner_up.score)} "
            f"({_gap(runner_up.score, floor.score)})"
        )
    return (
        f"retained the incumbent {floor.name} at V^0={_number(floor.score)}: "
        f"no version beat it, {best}; {grid}"
    )


def _number(score: float | None) -> str:
    """A ``V^m`` as the log states it; ``none`` where the version scored none."""
    return "none" if score is None else f"{score:.4f}"


def _gap(score: float, floor: float) -> str:
    """The distance between two ``V^m``, for the log.

    ``no difference`` where both revealed nothing carrying an ``s_v``: two
    versions on ``-inf`` are equally far from having attained anything, and the
    subtraction of one from the other is a ``nan`` that means nothing to a
    reader (see :class:`WorldReplay`).
    """
    difference = score - floor
    return "no difference" if math.isnan(difference) else f"{difference:+.4f}"


def _replay(candidate: PolicyCandidate, world: ReplayWorld, config: DreamConfig) -> WorldReplay:
    """One cell: replay one version on one world and score it, or record its crash.

    ``Exception`` and not ``BaseException``: a candidate's own failure is what
    this contains, while a ``KeyboardInterrupt`` or a ``SystemExit`` is someone
    stopping the round and has to keep travelling. The message is the exception's
    type and text and deliberately not a traceback — a traceback carries the file
    paths the policy was compiled under, which would make one round's report
    differ from the next's on a different machine.
    """
    try:
        run = world.simulator.replay(
            candidate.factory(),
            width=config.width,
            max_rounds=config.max_rounds,
            seed=config.seed,
        )
        result = run.result()
    except Exception as exc:  # noqa: BLE001 - the whole point is to not propagate
        return WorldReplay(
            world=world.name,
            score=None,
            result=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    # Outside the guard: Equation 1 is arithmetic over the numbers the run
    # reports, so a refusal here is a bug in this harness and not the
    # candidate's failure to navigate.
    score = replay_score(
        result.attainment,
        revealed=result.revealed,
        rounds=result.round_count,
        weights=config.weights,
    )
    return WorldReplay(world=world.name, score=score, result=result)


def _aggregate(name: str, replays: tuple[WorldReplay, ...]) -> VersionReport:
    """``V^m = (1/t) Σ_i V_i^m`` over one version's row, or nothing if a world failed."""
    scores = [replay.score for replay in replays]
    if any(score is None for score in scores):
        return VersionReport(name=name, score=None, replays=replays)
    # Summed in history order, so the floating-point result is the same every
    # run whatever order the cells were replayed in.
    return VersionReport(name=name, score=math.fsum(scores) / len(scores), replays=replays)


def _rank_key(item: tuple[int, VersionReport]) -> tuple[int, float, int]:
    """Best score first, versions with no aggregate last, ties in the given order."""
    index, version = item
    if version.score is None:
        return (1, 0.0, index)
    return (0, -version.score, index)


def _json_score(score: float | None) -> float | None:
    """A score as JSON can hold it: ``-inf`` becomes ``null`` (see :class:`WorldReplay`)."""
    if score is None or not math.isfinite(score):
        return None
    return score


def _cell(score: float | None, error: str | None = None) -> str:
    """One table cell: a score to four places, or ``failed`` where there is none.

    A revealed-nothing replay has a score of ``-inf`` and prints as such; only a
    crash leaves a cell, or a version's aggregate, without one at all.
    """
    if error is not None or score is None:
        return "failed"
    return f"{score:.4f}"


def _check_unique(what: str, names: Iterable[str]) -> None:
    """Refuse a grid whose rows or columns cannot be told apart in the report."""
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"{what} {name!r} appears twice; each one has to be tellable apart")
        seen.add(name)
