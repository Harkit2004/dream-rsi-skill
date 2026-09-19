"""The outer loop: deploy, record, dream, select, redeploy — ``T`` times (issue #16).

§3's recursive self-improvement loop, and the point at which the modules built
so far become one thing. At outer iteration ``t`` the driver:

1. deploys ``π_t`` online and collects a tree (:mod:`dream_rsi.orchestrator`);
2. appends it to the history — §3: "its final tree is recorded as ``𝒯_t`` and
   appended to the history, giving ``ℋ_t = ℋ_{t-1} ∪ {𝒯_t}``";
3. dreams ``M`` versions over every tree in ``ℋ_t`` (:mod:`dream_rsi.dream`);
4. has the policy-development agent refine the code (:mod:`dream_rsi.develop`);
5. selects ``π_{t+1}`` and goes round again.

Steps 3 and 4 are one call: §3 develops the versions in sequence, each from the
last one's replay feedback, which is what :func:`~dream_rsi.develop.develop`
does.

**This file is a wiring diagram.** Every number it reports is computed
somewhere else, and anything here that grew a rule of its own would belong in
the module it was reaching into. What is genuinely the driver's, and so is
here, is three things.

**The deployed policy is model-written too.** Dreaming already runs candidates
behind :mod:`dream_rsi.sandbox`; the winner of a cycle is then *deployed*, and
this is the only place that happens. It goes behind the same boundary, so there
is no path anywhere in the package by which policy source runs in the harness
process.

**A cycle is atomic.** Each one writes its tree, the policy it deployed and the
policy it selected into its own directory, and a record last of all, by rename.
So the record existing means the cycle finished, a run resumes by reading the
records it finds, and a cycle interrupted half-way is redone from the top rather
than patched up. That is what makes a crash in cycle 5 cost cycle 5.

**The cost report is the driver's.** Each half of a cycle counts what it spent
where it spends it — agent calls in the rollout, model calls and revealed nodes
in the offline phase — but the driver is the only place both halves are in
view, so it is where they are put on one record and reported against each other
(issue #18). §4 prices discovery in discovery-agent calls and §2 prices replay at
nothing, and the ratio of the two is the paper's claim, so a run says what each
half cost rather than what the cycle cost.

**The pool is a store beside the cycles.** A cycle records its tree in its own
directory, as the evidence for the record it writes, and adds it to
:class:`~dream_rsi.pool.SimulatorPool` under the cycle's name — so what a later
session dreams over is the pool, not a walk over whatever cycle directories it
finds, and a run resumed tomorrow inherits every tree the runs before it
recorded (issue #17). The cycle records stay the authority on which cycles this
run may trust: resuming re-adds the tree of a finished cycle the pool is missing
rather than dreaming over a history shorter than its own records claim.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import accumulate
from pathlib import Path
from typing import Any

from dream_rsi.adapters.agent import CodingAgent
from dream_rsi.adapters.evaluator import TaskEvaluator
from dream_rsi.adapters.fake_developer import FakeDeveloper
from dream_rsi.adapters.toy_search import ToySearchAgent, ToySearchEvaluator, plan_source
from dream_rsi.cost import CycleCost, CycleTiming, DreamCost, total
from dream_rsi.develop import DEFAULT_ATTEMPTS, PolicyDeveloper, develop
from dream_rsi.dream import DEFAULT_VERSIONS, DreamConfig, Selection, select
from dream_rsi.orchestrator import (
    STORE_DIRNAME,
    TREE_FILENAME,
    RolloutConfig,
    run_rollout,
)
from dream_rsi.pool import PoolConfig, PoolStats, SimulatorPool, subsample
from dream_rsi.sandbox import DEFAULT_LIMITS, SandboxedPolicy, SandboxLimits
from dream_rsi.tree import DiscoveryTree
from dream_rsi.workspace import SnapshotStore

__all__ = [
    "CYCLES_DIRNAME",
    "CYCLE_TEMPLATE",
    "DEFAULT_POLICY_SOURCE",
    "NEXT_POLICY_FILENAME",
    "POLICY_FILENAME",
    "POOL_DIRNAME",
    "RECORD_FILENAME",
    "TIMING_FILENAME",
    "CycleRecord",
    "Run",
    "RunConfig",
    "RunError",
    "main",
    "run_cycles",
]

CYCLES_DIRNAME = "cycles"
CYCLE_TEMPLATE = "cycle_{:03d}"

# Where the run keeps ℋ: one tree per finished cycle, under the cycle's name.
POOL_DIRNAME = "pool"

# What one cycle leaves behind, beside the tree and round log
# ``orchestrator.Rollout.save`` writes. ``RECORD_FILENAME`` is written last and
# by rename, so its presence is what marks a cycle finished.
POLICY_FILENAME = "policy.py"
NEXT_POLICY_FILENAME = "next_policy.py"
RECORD_FILENAME = "cycle.json"
WORKSPACE_DIRNAME = "workspace"

# How long the cycle took, beside the record rather than in it: the record is a
# function of the run's seed and a clock is a function of the machine (see
# ``cost.CycleTiming``). A cycle whose timing is missing is still a finished
# cycle — only the record says that — so this file is read where it is there and
# nothing fails where it is not.
TIMING_FILENAME = "timing.json"

# The toy wiring ``main`` runs: candidates spanning the landscape in
# ``adapters/toy_search.py``, so a rollout over them is choosing between real
# alternatives rather than walking a gradient.
TOY_PROBLEM = "pick the plan (width, depth) with the best throughput within the cost budget"
TOY_SCRIPT = (
    plan_source(0, 1),
    plan_source(1, 1),
    plan_source(2, 2),
    plan_source(3, 3),
    plan_source(2, 1),
    plan_source(1, 2),
    plan_source(0, 2),
    plan_source(4, 0),
)

# π_1 for a run that is not handed one: the §B.2 shape, "keep NAME =
# "OptimalPolicy" and implement class OptimalPolicy(...)", over a baseline the
# package already ships. A real run supplies its own.
DEFAULT_POLICY_SOURCE = """\
from dream_rsi.policy import GreedyBestFirstPolicy


class OptimalPolicy(GreedyBestFirstPolicy):
    pass
"""


class RunError(ValueError):
    """A run directory holds something this driver cannot read back."""


@dataclass(frozen=True)
class RunConfig:
    """How many cycles a run does, and what each phase of one runs under.

    ``rollout`` is the online half and ``dreaming`` the offline one; ``versions``
    is §3's ``M`` and ``attempts`` how many times a refused revision is asked for
    again. ``limits`` caps every process a policy's code runs in, online
    deployment included. ``pool`` bounds how much of the history one offline
    phase replays, and by default bounds it not at all.
    """

    cycles: int = 3
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    dreaming: DreamConfig = field(default_factory=DreamConfig)
    versions: int = DEFAULT_VERSIONS
    attempts: int = DEFAULT_ATTEMPTS
    limits: SandboxLimits = DEFAULT_LIMITS
    pool: PoolConfig = field(default_factory=PoolConfig)

    def __post_init__(self) -> None:
        if self.cycles < 1:
            # §3 indexes the outer iterations from t = 1, so a run is at least
            # one deploy-and-dream. Zero of them is a caller's mistake, not an
            # empty run.
            raise ValueError(f"a run needs at least one cycle, got {self.cycles}")
        # Checked here as well as in ``develop``, which is only reached once this
        # cycle's rollout has run and been saved: a configuration that can never
        # finish a cycle should not first spend the online half of one.
        if self.versions < 1:
            raise ValueError(f"a cycle needs at least one policy version, got {self.versions}")
        if self.attempts < 1:
            raise ValueError(f"a revision needs at least one attempt, got {self.attempts}")


@dataclass(frozen=True)
class CycleRecord:
    """One completed outer iteration, as its ``cycle.json`` carries it.

    ``world`` is the name this cycle's tree joined the pool under and ``pool`` is
    the worlds the offline phase ran over — ``ℋ_t``, this cycle's own last, since
    §3 appends before it dreams, or the subsequence of it that
    :func:`~dream_rsi.pool.subsample` kept when the run set a
    :class:`~dream_rsi.pool.PoolConfig` limit. It is what this cycle dreamed
    over, which is why a limit is measured rather than assumed: the store still
    holds every tree. ``versions`` are the ``M`` the round developed
    and ``rejected`` the count of outputs the harness refused along the way
    (``develop.Rejection``). ``selection`` names ``π_{t+1}``.

    ``best_score`` is the best ``s_v`` this cycle's own rollout recorded — the
    discovery quality §4 plots against cumulative agent calls — in the canonical
    direction node scores are stored in, so it is comparable across cycles and
    across tasks that do not run larger-is-better. It is ``None`` where the cycle
    scored nothing at all.

    ``cost`` is what the cycle spent on each side of the loop (issue #18).
    ``timing`` is how long that took, and is the one thing here that is not a
    function of the run's seed: it is read from ``timing.json`` beside the record
    and deliberately absent from :meth:`to_dict`, so two runs under one seed
    write identical records (working rule 5). It is ``None`` for a cycle whose
    timing file is missing.
    """

    index: int
    world: str
    attempts: int
    stop_reason: str
    pool: tuple[str, ...]
    versions: tuple[str, ...]
    rejected: int
    selection: Selection
    best_score: float | None = None
    cost: CycleCost = CycleCost()
    timing: CycleTiming | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "world": self.world,
            "attempts": self.attempts,
            "stop_reason": self.stop_reason,
            "pool": list(self.pool),
            "versions": list(self.versions),
            "rejected": self.rejected,
            "selection": self.selection.to_dict(),
            "best_score": self.best_score,
            "cost": self.cost.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> CycleRecord:
        """Read a record back, as a resumed run reads the cycles it already did.

        A ``V^m`` of ``-inf`` comes back as ``None``, because JSON has no token
        for it and ``Selection.to_dict`` already writes it that way. Nothing
        downstream re-runs a selection off a record — the next cycle deploys the
        source in ``next_policy.py``, not a score — so the record is a log here
        and not an input.
        """
        selection = payload["selection"]
        return cls(
            index=int(payload["index"]),
            world=str(payload["world"]),
            attempts=int(payload["attempts"]),
            stop_reason=str(payload["stop_reason"]),
            pool=tuple(str(name) for name in payload["pool"]),
            versions=tuple(str(name) for name in payload["versions"]),
            rejected=int(payload["rejected"]),
            selection=Selection(
                winner=str(selection["winner"]),
                incumbent=str(selection["incumbent"]),
                score=selection["score"],
                incumbent_score=selection["incumbent_score"],
                rationale=str(selection["rationale"]),
            ),
            best_score=None if payload["best_score"] is None else float(payload["best_score"]),
            cost=CycleCost.from_dict(payload["cost"]),
        )


@dataclass(frozen=True)
class Run:
    """A finished run: what each cycle did, and the policy the next one would deploy.

    ``pool`` is what the simulator pool holds at the end and ``stats`` what it
    costs to dream over (issue #17). They describe the store, not the round: a
    cycle that subsampled dreamed over the worlds its own record names, while
    the pool keeps every tree for the runs that come after.
    """

    cycles: tuple[CycleRecord, ...]
    policy: str
    pool: tuple[str, ...]
    stats: PoolStats

    @classmethod
    def load(cls, directory: str | Path) -> Run:
        """Read a run back out of its directory, with nothing live behind it.

        What :func:`run_cycles` would have returned, assembled from the cycle
        records, the policy the last of them selected and the pool on disk — no
        agent, no evaluator and no developer, because a report is read after the
        process that produced it is gone. The cycles are those with a record, in
        order and stopped at the first gap, for the reason :func:`_resume` gives.
        """
        directory = Path(directory)
        cycles = directory / CYCLES_DIRNAME
        records = tuple(record for _, record in _finished(cycles))
        pool = SimulatorPool(directory / POOL_DIRNAME)
        # A run with no finished cycle never selected anything, so there is no π
        # it would deploy next; π_1 was the caller's and is not on disk.
        policy = (
            _read(_cycle_dir(cycles, records[-1].index) / NEXT_POLICY_FILENAME)
            if records
            else ""
        )
        return cls(
            cycles=records,
            policy=policy,
            pool=pool.names(),
            stats=pool.stats(),
        )

    @property
    def totals(self) -> CycleCost:
        """What the whole run spent, half by half — §4's *cumulative* cost.

        Summed from the records, so a resumed run totals the cycles it read back
        as well as the ones it ran: the cost of a cycle is the cycle's, not the
        session's.
        """
        return total(record.cost for record in self.cycles)

    @property
    def best_score(self) -> float | None:
        """The best ``s_v`` any cycle of this run recorded, or ``None`` if none did."""
        scored = [record.best_score for record in self.cycles if record.best_score is not None]
        return max(scored) if scored else None

    def to_text(self) -> str:
        """The run report: quality against cost, cycle by cycle (issue #18).

        One row per cycle, carrying what the cycle discovered, what each half of
        it spent and which version it selected, under a header that states the
        split the paper's claim rests on. The rationale lines beneath say why
        each of those versions was the one deployed next, in the selection's own
        words (``dream.Selection``), because a run that changed policy five times
        and cannot say why is not a report.
        """
        # §4 plots discovery quality against the *cumulative* number of
        # discovery-agent calls, so the report carries the running total beside
        # each cycle's own rather than leaving the reader to add them up.
        cumulative = list(accumulate(record.cost.online.agent_calls for record in self.cycles))
        rows = [
            [
                str(record.index),
                _quality(record.best_score),
                str(record.cost.online.agent_calls),
                str(spent),
                str(record.cost.online.evaluations),
                _seconds(None if record.timing is None else record.timing.online_seconds),
                str(record.cost.dreaming.developer_calls),
                str(record.cost.dreaming.cells),
                str(record.cost.dreaming.reveals),
                _seconds(None if record.timing is None else record.timing.dreaming_seconds),
                record.selection.winner,
            ]
            for record, spent in zip(self.cycles, cumulative, strict=True)
        ]
        header = [
            "cycle",
            "best",
            "online.calls",
            "online.total",
            "online.evals",
            "online.s",
            "dream.calls",
            "dream.cells",
            "dream.reveals",
            "dream.s",
            "selected",
        ]
        widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]

        totals = self.totals
        lines = [
            f"run: {len(self.cycles)} cycle(s), best score {_quality(self.best_score)}",
            # The two halves side by side and never summed: §4 counts discovery
            # in agent calls, §2 prices replay at nothing, and what a reader
            # needs is the ratio between them rather than a total across them.
            (
                f"cost split: {totals.online.agent_calls} online agent call(s) "
                f"against {totals.dreaming.developer_calls} policy-development call(s) "
                f"and {totals.dreaming.cells} replay cell(s) revealing "
                f"{totals.dreaming.reveals} node(s)"
            ),
            f"  {_leverage(totals.leverage)} replayed node(s) per online agent call",
            (
                f"pool: {self.stats.trees} tree(s), {self.stats.nodes} node(s), "
                f"{self.stats.bytes} byte(s) on disk"
            ),
            "",
            *(
                "  ".join(cell.rjust(width) for cell, width in zip(row, widths))
                for row in (header, *rows)
            ),
        ]
        if self.cycles:
            lines += [
                "",
                "selection:",
                *(
                    f"  cycle {record.index}: {record.selection.rationale}"
                    for record in self.cycles
                ),
            ]
        return "\n".join(lines) + "\n"


def run_cycles(
    *,
    agent: CodingAgent,
    evaluator: TaskEvaluator,
    developer: PolicyDeveloper,
    policy: str,
    problem: str,
    directory: str | Path,
    config: RunConfig | None = None,
    scratch_root: str | Path | None = None,
) -> Run:
    """Run ``config.cycles`` outer iterations under ``directory``, and report them.

    ``policy`` is ``π_1``'s module source — it is only read on a run that has no
    completed cycles to resume, since after that ``π_t`` is whatever cycle
    ``t - 1`` selected. ``scratch_root`` is where the sandboxed processes put
    their scratch directories, and is passed to every one of them.

    Resuming is what calling this twice on the same directory does: the cycles
    whose records are there are read back rather than redone, and the first one
    without a record is run from the top. Nothing else distinguishes a resumed
    run — there is no flag, so a crashed run is restarted with the command that
    started it.

    A policy that fails online — it oversteps its limits, or answers with
    something that is not a batch — raises out of the cycle it was deployed in
    (:class:`~dream_rsi.sandbox.SandboxError`). That is deliberate: it is the
    version this run's own selection committed to, and swallowing it would leave
    every later cycle exploring under a policy nobody chose.
    """
    config = RunConfig() if config is None else config
    cycles = Path(directory) / CYCLES_DIRNAME
    cycles.mkdir(parents=True, exist_ok=True)
    pool = SimulatorPool(Path(directory) / POOL_DIRNAME)

    records, source = _resume(cycles, config.cycles, policy, pool)
    history = [record.world for record in records]
    for index in range(len(records), config.cycles):
        record, source = _cycle(
            index,
            agent=agent,
            evaluator=evaluator,
            developer=developer,
            source=source,
            problem=problem,
            cycle=cycles / CYCLE_TEMPLATE.format(index),
            pool=pool,
            history=tuple(history),
            config=config,
            scratch_root=scratch_root,
        )
        records.append(record)
        history.append(record.world)
    return Run(cycles=tuple(records), policy=source, pool=pool.names(), stats=pool.stats())


def _resume(
    cycles: Path, wanted: int, policy: str, pool: SimulatorPool
) -> tuple[list[CycleRecord], str]:
    """The cycles already finished under ``cycles``, and what the next one deploys.

    Read in order and stopped at the first gap: a cycle with no record did not
    finish, and every cycle after it was run under a policy that cycle was
    supposed to pick, so whatever is on disk beyond the gap describes a history
    this run no longer has. They are redone rather than trusted.

    A finished cycle's tree is put back into the pool if the pool cannot read it
    back, from the copy the cycle itself recorded. The records say which worlds
    this run has; dreaming over fewer of them because a pool directory was
    cleaned, half-copied, or never written by an older run would shorten ``ℋ_t``
    without saying so — and failing on a tree the run has a good copy of would
    be worse still.
    """
    records: list[CycleRecord] = []
    source = policy
    for directory, record in _finished(cycles, wanted):
        records.append(record)
        if not pool.holds(record.world):
            pool.add(record.world, _tree(directory / TREE_FILENAME))
        source = _read(directory / NEXT_POLICY_FILENAME)
    return records, source


def _finished(cycles: Path, wanted: int | None = None) -> list[tuple[Path, CycleRecord]]:
    """The finished cycles under ``cycles``, in order, stopped at the first gap.

    ``wanted`` bounds how many are looked for; without one the walk runs until it
    finds a cycle directory with no record, which is what reading a run back
    needs (:meth:`Run.load`) and what resuming one needs bounded.
    """
    found: list[tuple[Path, CycleRecord]] = []
    index = 0
    while wanted is None or index < wanted:
        directory = _cycle_dir(cycles, index)
        if not (directory / RECORD_FILENAME).is_file():
            break
        found.append((directory, _read_record(directory)))
        index += 1
    return found


def _cycle_dir(cycles: Path, index: int) -> Path:
    return cycles / CYCLE_TEMPLATE.format(index)


def _cycle(
    index: int,
    *,
    agent: CodingAgent,
    evaluator: TaskEvaluator,
    developer: PolicyDeveloper,
    source: str,
    problem: str,
    cycle: Path,
    pool: SimulatorPool,
    history: tuple[str, ...],
    config: RunConfig,
    scratch_root: str | Path | None,
) -> tuple[CycleRecord, str]:
    """One outer iteration: §3's five steps, and the directory that records them."""
    # From the top, not from what is there: a directory without a record is a
    # cycle that was interrupted, and its tree, round log and next policy may
    # each be from a different moment. Rerunning over them would leave a cycle
    # whose files disagree with each other.
    shutil.rmtree(cycle, ignore_errors=True)
    workspace = cycle / WORKSPACE_DIRNAME
    workspace.mkdir(parents=True)
    _write(cycle / POLICY_FILENAME, source)

    # 1. Deploy π_t online. Behind the sandbox because the source is whatever the
    # development agent wrote and this run selected.
    started = time.monotonic()
    with SandboxedPolicy(
        source, limits=config.limits, scratch_root=scratch_root
    ) as deployed:
        rollout = run_rollout(
            agent=agent,
            evaluator=evaluator,
            policy=deployed,
            problem=problem,
            workspace=workspace,
            snapshots=SnapshotStore(cycle / STORE_DIRNAME),
            # PAPER-GAP: §3 has the online transition be stochastic — "the
            # discovery agent may generate different outcomes from the same
            # starting workspace" — and never says how the rollouts of
            # successive outer iterations are seeded. Left alone, a deterministic
            # agent would hand a retained policy the identical tree every cycle
            # and the pool would grow in name only. We offset the run's seed by
            # the cycle index, so each cycle draws its own outcomes and the whole
            # run is still a function of one number. Revisit if the authors'
            # implementation lands (see references/method.md).
            config=replace(config.rollout, seed=config.rollout.seed + index),
        )
    online_seconds = time.monotonic() - started
    rollout.save(cycle)

    # 2. Append 𝒯_t to the history: ℋ_t = ℋ_{t-1} ∪ {𝒯_t} (§3). Into the pool
    # before the record that vouches for this cycle is written, so a cycle a
    # later run trusts is a cycle whose tree that run can dream over.
    name = CYCLE_TEMPLATE.format(index)
    pool.add(name, rollout.tree)
    dreamed = subsample((*history, name), config.pool)

    # 3 and 4. Dream M versions over ℋ_t, each revised from the last one's replay
    # feedback, then 5: select π_{t+1}, which §3 guarantees is no worse than π_t.
    started = time.monotonic()
    development = develop(
        source,
        pool.worlds(dreamed),
        developer,
        versions=config.versions,
        attempts=config.attempts,
        config=config.dreaming,
        limits=config.limits,
        scratch_root=scratch_root,
    )
    selection = select(development.comparison)
    dreaming_seconds = time.monotonic() - started
    chosen = {version.name: version.source for version in development.versions}[selection.winner]
    _write(cycle / NEXT_POLICY_FILENAME, chosen)

    # What the cycle cost, each half counted by the half that spent it (issue
    # #18): the rollout knows its calls and its evaluations, and the offline
    # phase its model calls and the grid it replayed.
    timing = CycleTiming(online_seconds=online_seconds, dreaming_seconds=dreaming_seconds)
    record = CycleRecord(
        index=index,
        world=name,
        attempts=len(rollout.tree) - 1,
        stop_reason=rollout.stop_reason,
        pool=dreamed,
        versions=tuple(version.name for version in development.versions),
        rejected=len(development.rejected),
        selection=selection,
        best_score=_best_score(rollout.tree),
        cost=CycleCost(
            online=rollout.cost,
            dreaming=DreamCost(
                developer_calls=development.calls,
                cells=development.comparison.cells,
                reveals=development.comparison.reveals,
            ),
        ),
        timing=timing,
    )
    # Before the record, which is what marks the cycle finished: a cycle the
    # resumed run trusts is one whose timing is already beside it.
    _write(cycle / TIMING_FILENAME, json.dumps(timing.to_dict(), indent=2, sort_keys=True) + "\n")
    _write_record(cycle / RECORD_FILENAME, record)
    return record, chosen


def _best_score(tree: DiscoveryTree) -> float | None:
    """The best ``s_v`` in a recorded tree, or ``None`` where nothing scored.

    Node scores are canonical — ``EvalResult.to_node_fields`` has already turned
    the task's own direction round — so the best is the largest whichever way the
    task's metric runs, and the root, which has no score, is excluded with every
    attempt that failed to produce one.
    """
    scores = [node.score for node in tree.iter_nodes() if node.score is not None]
    return max(scores) if scores else None


def _tree(path: Path) -> DiscoveryTree:
    """A finished cycle's own copy of the tree it recorded."""
    try:
        return DiscoveryTree.load(path)
    except (OSError, ValueError) as exc:
        raise RunError(f"{path} is not a readable tree: {exc}") from exc


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunError(f"a finished cycle is missing {path}: {exc}") from exc


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _read_record(directory: Path) -> CycleRecord:
    """One finished cycle's record, with the timing that was written beside it."""
    path = directory / RECORD_FILENAME
    try:
        record = CycleRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RunError(f"{path} is not a readable cycle record: {exc}") from exc
    return replace(record, timing=_read_timing(directory / TIMING_FILENAME))


def _read_timing(path: Path) -> CycleTiming | None:
    """How long the cycle took, or ``None`` where that was not recorded.

    Missing is not broken: the record is what says a cycle finished, and a run
    whose report is a clock short still spent every call it spent. A file that is
    there and unreadable is a different matter and says so.
    """
    if not path.is_file():
        return None
    try:
        return CycleTiming.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RunError(f"{path} is not a readable cycle timing: {exc}") from exc


def _quality(score: float | None) -> str:
    """A best score as the report states it; ``none`` where the cycle scored nothing."""
    return "none" if score is None else f"{score:.4f}"


def _seconds(elapsed: float | None) -> str:
    """A measured duration for the report, or ``-`` where none was recorded."""
    return "-" if elapsed is None else f"{elapsed:.2f}"


def _leverage(ratio: float | None) -> str:
    """The replay-per-call ratio, or ``none`` for a run that made no calls."""
    return "none" if ratio is None else f"{ratio:.2f}"


def _write_record(path: Path, record: CycleRecord) -> None:
    """Write the record by rename, so it appears whole or not at all.

    The cycle is finished when this file is there, and every other file it
    promises was written before it. A half-written record would be a cycle a
    resumed run trusts and cannot read.
    """
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the loop on the toy task, with no model on either side of it."""
    parser = argparse.ArgumentParser(description="Run the Dream-RSI loop on the toy task.")
    parser.add_argument(
        "directory",
        nargs="?",
        default="runs/loop",
        type=Path,
        help=f"where to write {CYCLES_DIRNAME}/ (default: runs/loop); "
        "running again resumes it",
    )
    parser.add_argument("--cycles", type=int, default=3, help="outer iterations T (default: 3)")
    parser.add_argument("--workers", type=int, default=2, help="parallel workers W (default: 2)")
    parser.add_argument(
        "--rounds", type=int, default=4, help="decision rounds per rollout (default: 4)"
    )
    parser.add_argument(
        "--versions",
        type=int,
        default=DEFAULT_VERSIONS,
        help=f"policy versions M per cycle (default: {DEFAULT_VERSIONS})",
    )
    parser.add_argument("--seed", type=int, default=0, help="the run's seed (default: 0)")
    parser.add_argument(
        "--pool-limit",
        type=int,
        default=None,
        help="dream over at most this many worlds a cycle (default: the whole pool)",
    )
    args = parser.parse_args(argv)

    run = run_cycles(
        agent=ToySearchAgent(script=TOY_SCRIPT),
        evaluator=ToySearchEvaluator(),
        developer=FakeDeveloper(),
        policy=DEFAULT_POLICY_SOURCE,
        problem=TOY_PROBLEM,
        directory=args.directory,
        config=RunConfig(
            cycles=args.cycles,
            rollout=RolloutConfig(workers=args.workers, max_rounds=args.rounds, seed=args.seed),
            # The width a version is offered while dreaming is the width the next
            # rollout will actually run at, or Equation 1's parallelism term
            # rewards batching the online loop cannot spend (``dream.DreamConfig``).
            # The seed is the run's, and unlike the rollout's it is not offset per
            # cycle: replay reseeds each policy-world pair from it (§3), and two
            # cycles dreaming over the same world have to score it the same way
            # for their V^m to be comparable at all.
            dreaming=DreamConfig(width=args.workers, seed=args.seed),
            versions=args.versions,
            pool=PoolConfig(limit=args.pool_limit, seed=args.seed),
        ),
    )
    print(run.to_text(), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
