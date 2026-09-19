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

**The pool is the cycle directories.** One tree per cycle, dreamed over by name.
Keeping it across sessions, measuring it, and deciding what to do once it is too
large to replay in full is issue #17; this grows it honestly and reads all of it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dream_rsi.adapters.agent import CodingAgent
from dream_rsi.adapters.evaluator import TaskEvaluator
from dream_rsi.adapters.fake_developer import FakeDeveloper
from dream_rsi.adapters.toy_search import ToySearchAgent, ToySearchEvaluator, plan_source
from dream_rsi.develop import DEFAULT_ATTEMPTS, PolicyDeveloper, develop
from dream_rsi.dream import DEFAULT_VERSIONS, DreamConfig, ReplayWorld, Selection, select
from dream_rsi.orchestrator import (
    STORE_DIRNAME,
    TREE_FILENAME,
    RolloutConfig,
    run_rollout,
)
from dream_rsi.replay import ReplaySimulator
from dream_rsi.sandbox import DEFAULT_LIMITS, SandboxedPolicy, SandboxLimits
from dream_rsi.tree import DiscoveryTree
from dream_rsi.workspace import SnapshotStore

__all__ = [
    "CYCLES_DIRNAME",
    "CYCLE_TEMPLATE",
    "DEFAULT_POLICY_SOURCE",
    "NEXT_POLICY_FILENAME",
    "POLICY_FILENAME",
    "RECORD_FILENAME",
    "CycleRecord",
    "Run",
    "RunConfig",
    "RunError",
    "main",
    "run_cycles",
]

CYCLES_DIRNAME = "cycles"
CYCLE_TEMPLATE = "cycle_{:03d}"

# What one cycle leaves behind, beside the tree and round log
# ``orchestrator.Rollout.save`` writes. ``RECORD_FILENAME`` is written last and
# by rename, so its presence is what marks a cycle finished.
POLICY_FILENAME = "policy.py"
NEXT_POLICY_FILENAME = "next_policy.py"
RECORD_FILENAME = "cycle.json"
WORKSPACE_DIRNAME = "workspace"

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
    deployment included.
    """

    cycles: int = 3
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    dreaming: DreamConfig = field(default_factory=DreamConfig)
    versions: int = DEFAULT_VERSIONS
    attempts: int = DEFAULT_ATTEMPTS
    limits: SandboxLimits = DEFAULT_LIMITS

    def __post_init__(self) -> None:
        if self.cycles < 1:
            # §3 indexes the outer iterations from t = 1, so a run is at least
            # one deploy-and-dream. Zero of them is a caller's mistake, not an
            # empty run.
            raise ValueError(f"a run needs at least one cycle, got {self.cycles}")


@dataclass(frozen=True)
class CycleRecord:
    """One completed outer iteration, as its ``cycle.json`` carries it.

    ``world`` is the name this cycle's tree joined the pool under and ``pool`` is
    ``ℋ_t``, the worlds the offline phase ran over — this cycle's own last, since
    §3 appends before it dreams. ``versions`` are the ``M`` the round developed
    and ``rejected`` the count of outputs the harness refused along the way
    (``develop.Rejection``). ``selection`` names ``π_{t+1}``.

    Counting what a cycle *cost* — agent calls, node reveals, wall clock, and the
    online-versus-dreaming split — is issue #18, and deliberately not here.
    """

    index: int
    world: str
    attempts: int
    stop_reason: str
    pool: tuple[str, ...]
    versions: tuple[str, ...]
    rejected: int
    selection: Selection

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
        )


@dataclass(frozen=True)
class Run:
    """A finished run: what each cycle did, and the policy the next one would deploy."""

    cycles: tuple[CycleRecord, ...]
    policy: str


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

    records, pool, source = _resume(cycles, config.cycles, policy)
    for index in range(len(records), config.cycles):
        record, world, source = _cycle(
            index,
            agent=agent,
            evaluator=evaluator,
            developer=developer,
            source=source,
            problem=problem,
            cycle=cycles / CYCLE_TEMPLATE.format(index),
            pool=tuple(pool),
            config=config,
            scratch_root=scratch_root,
        )
        records.append(record)
        pool.append(world)
    return Run(cycles=tuple(records), policy=source)


def _resume(
    cycles: Path, wanted: int, policy: str
) -> tuple[list[CycleRecord], list[ReplayWorld], str]:
    """The cycles already finished under ``cycles``, and what the next one deploys.

    Read in order and stopped at the first gap: a cycle with no record did not
    finish, and every cycle after it was run under a policy that cycle was
    supposed to pick, so whatever is on disk beyond the gap describes a history
    this run no longer has. They are redone rather than trusted.
    """
    records: list[CycleRecord] = []
    pool: list[ReplayWorld] = []
    source = policy
    for index in range(wanted):
        directory = cycles / CYCLE_TEMPLATE.format(index)
        if not (directory / RECORD_FILENAME).is_file():
            break
        record = _read_record(directory / RECORD_FILENAME)
        records.append(record)
        pool.append(_world(record.world, directory))
        source = _read(directory / NEXT_POLICY_FILENAME)
    return records, pool, source


def _cycle(
    index: int,
    *,
    agent: CodingAgent,
    evaluator: TaskEvaluator,
    developer: PolicyDeveloper,
    source: str,
    problem: str,
    cycle: Path,
    pool: tuple[ReplayWorld, ...],
    config: RunConfig,
    scratch_root: str | Path | None,
) -> tuple[CycleRecord, ReplayWorld, str]:
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
    rollout.save(cycle)

    # 2. Append 𝒯_t to the history: ℋ_t = ℋ_{t-1} ∪ {𝒯_t} (§3).
    world = ReplayWorld(name=CYCLE_TEMPLATE.format(index), simulator=ReplaySimulator(rollout.tree))
    history = (*pool, world)

    # 3 and 4. Dream M versions over ℋ_t, each revised from the last one's replay
    # feedback, then 5: select π_{t+1}, which §3 guarantees is no worse than π_t.
    development = develop(
        source,
        history,
        developer,
        versions=config.versions,
        attempts=config.attempts,
        config=config.dreaming,
        limits=config.limits,
        scratch_root=scratch_root,
    )
    selection = select(development.comparison)
    chosen = {version.name: version.source for version in development.versions}[selection.winner]
    _write(cycle / NEXT_POLICY_FILENAME, chosen)

    record = CycleRecord(
        index=index,
        world=world.name,
        attempts=len(rollout.tree) - 1,
        stop_reason=rollout.stop_reason,
        pool=tuple(entry.name for entry in history),
        versions=tuple(version.name for version in development.versions),
        rejected=len(development.rejected),
        selection=selection,
    )
    _write_record(cycle / RECORD_FILENAME, record)
    return record, world, chosen


def _world(name: str, directory: Path) -> ReplayWorld:
    """The replay world a finished cycle's tree makes, read back off disk."""
    return ReplayWorld(
        name=name,
        simulator=ReplaySimulator(DiscoveryTree.load(directory / TREE_FILENAME)),
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunError(f"a finished cycle is missing {path}: {exc}") from exc


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _read_record(path: Path) -> CycleRecord:
    try:
        return CycleRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RunError(f"{path} is not a readable cycle record: {exc}") from exc


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
            dreaming=DreamConfig(width=args.workers),
            versions=args.versions,
        ),
    )
    for record in run.cycles:
        print(
            f"cycle {record.index}: {record.attempts} attempt(s), stopped on "
            f"{record.stop_reason}, dreamed {len(record.versions)} version(s) over "
            f"{len(record.pool)} world(s) -> {record.selection.rationale}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
