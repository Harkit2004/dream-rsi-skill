"""The full RSI driver: T cycles of deploy, record, dream, select, redeploy (issue #16).

§3's outer loop, wired out of pieces that are already tested on their own. What
matters here is therefore the wiring and nothing else: that each cycle's tree
joins the history its own dreaming phase runs over, that the version a cycle
selects is the one the next cycle actually deploys, that a run interrupted
half-way picks up where it stopped without losing or repeating a tree, and that
the whole thing is a function of its seed.

Nothing in this file calls a model: the discovery agent is the scripted toy one
and the development agent is a stub that answers from a list.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dream_rsi.adapters.toy_search import ToySearchAgent, ToySearchEvaluator, plan_source
from dream_rsi.cost import DreamCost, OnlineCost
from dream_rsi.develop import RevisionContext
from dream_rsi.dream import DreamConfig
from dream_rsi.orchestrator import (
    STOP_EMPTY_BATCH,
    STOP_MAX_ROUNDS,
    TREE_FILENAME,
    RolloutConfig,
)
from dream_rsi.pool import TREE_SUFFIX, PoolConfig
from dream_rsi.run import (
    CYCLE_TEMPLATE,
    CYCLES_DIRNAME,
    MANIFEST_FILENAME,
    NEXT_POLICY_FILENAME,
    POLICY_FILENAME,
    POOL_DIRNAME,
    RECORD_FILENAME,
    TIMING_FILENAME,
    Run,
    RunConfig,
    RunError,
    run_cycles,
)
from dream_rsi.sandbox import SandboxError, SandboxLimits
from dream_rsi.tree import DiscoveryTree

REPO = Path(__file__).resolve().parents[1]

# Generous next to what these policies spend — a rollout of eight attempts over
# the toy landscape is milliseconds of child CPU — so nothing here is measuring
# the machine it runs on. No policy in this file writes, bar the one that is
# supposed to be stopped from writing.
TEST_LIMITS = SandboxLimits(
    wall_seconds=5.0,
    cpu_seconds=2,
    memory_bytes=512 * 1024 * 1024,
    disk_bytes=1024 * 1024,
)

PROBLEM = "pick the plan (width, depth) with the best throughput within the cost budget"

# Candidates spanning the toy landscape, so a rollout over them has real
# alternatives to choose between. The agent answers to the attempt's seed, so
# which attempt proposes which plan is fixed (``adapters/toy_search.py``).
SCRIPT = (
    plan_source(0, 1),
    plan_source(1, 1),
    plan_source(2, 2),
    plan_source(3, 3),
    plan_source(2, 1),
    plan_source(1, 2),
    plan_source(0, 2),
    plan_source(4, 0),
)

# π_1: refines level by level and never stops itself, so it runs to the round
# cap and records a tree eight attempts wide and four deep.
BREADTH_SOURCE = """\
from dream_rsi.policy import BreadthFirstPolicy


class OptimalPolicy(BreadthFirstPolicy):
    pass
"""

# The revision the stub answers with: the same strategy that stops after two
# rounds. On the world the incumbent recorded it reaches the same best score
# having revealed half the nodes, so Equation 1's cost term puts it ahead and
# selection has to prefer it — and deployed online it ends on its own empty
# batch instead of the round cap, which is how a test can see which of the two
# drove a rollout.
THRIFTY_SOURCE = """\
from dream_rsi.policy import BreadthFirstPolicy


class OptimalPolicy(BreadthFirstPolicy):
    def reset(self, rng=None):
        super().reset(rng)
        self.spent = 0

    def choose(self, tree, live, width):
        self.spent += 1
        if self.spent > 2:
            return ()
        return super().choose(tree, live, width)
"""


def _escape_source(escape: Path) -> str:
    """A policy that writes outside its sandbox, which the boundary refuses.

    Where a test needs a cycle to fail for a reason that is not the run's
    configuration, this is the deterministic way: the sandbox refuses the write
    and the cycle raises with it, leaving whatever the driver wrote before the
    cycle behind.
    """
    return (
        "from pathlib import Path\n"
        "\n"
        "\n"
        "class OptimalPolicy:\n"
        "    def __init__(self, config=None):\n"
        "        self.config = dict(config or {})\n"
        "\n"
        "    def select(self, tree, eligible, width):\n"
        f"        Path({str(escape)!r}).write_text('escaped')\n"
        "        return (tree.root_id,)\n"
    )


@dataclass
class ScriptedDeveloper:
    """A development agent that answers from a list and keeps what it was asked.

    Deterministic and inert: no model, no network, no clock. The number of
    contexts it collected is how a test sees how many versions a run actually
    developed, which is what tells a resumed run from one that started over.
    """

    script: tuple[str, ...] = (THRIFTY_SOURCE,)
    calls: list[RevisionContext] = field(default_factory=list)

    def revise(self, context: RevisionContext) -> str:
        self.calls.append(context)
        return self.script[(len(self.calls) - 1) % len(self.script)]


def _run(
    directory: Path,
    developer: ScriptedDeveloper,
    *,
    cycles: int = 3,
    policy: str = BREADTH_SOURCE,
    scratch_root: Path | None = None,
    pool: PoolConfig | None = None,
    workers: int = 2,
    rounds: int = 4,
    seed: int = 0,
    dreaming: DreamConfig | None = None,
) -> Run:
    """A toy run in ``directory``: two versions a cycle, over the toy landscape."""
    return run_cycles(
        agent=ToySearchAgent(script=SCRIPT),
        evaluator=ToySearchEvaluator(),
        developer=developer,
        policy=policy,
        problem=PROBLEM,
        directory=directory,
        config=RunConfig(
            cycles=cycles,
            rollout=RolloutConfig(workers=workers, max_rounds=rounds, max_nodes=16, seed=seed),
            dreaming=DreamConfig(width=workers) if dreaming is None else dreaming,
            # Two: the incumbent and one revision, which is the smallest round
            # that can select anything other than what it started with.
            versions=2,
            limits=TEST_LIMITS,
            pool=PoolConfig() if pool is None else pool,
        ),
        scratch_root=scratch_root,
    )


def _cycle_dir(directory: Path, index: int) -> Path:
    return directory / CYCLES_DIRNAME / CYCLE_TEMPLATE.format(index)


def _records(directory: Path) -> list[str]:
    """Every cycle record a run left behind, as the bytes it wrote them as."""
    return [
        path.read_text(encoding="utf-8")
        for path in sorted((directory / CYCLES_DIRNAME).glob(f"*/{RECORD_FILENAME}"))
    ]


def _trees(directory: Path) -> list[str]:
    return [
        path.read_text(encoding="utf-8")
        for path in sorted((directory / CYCLES_DIRNAME).glob(f"*/{TREE_FILENAME}"))
    ]


@pytest.mark.parametrize("field", ["cycles", "versions", "attempts"])
def test_a_run_that_could_not_finish_a_cycle_is_rejected_before_it_starts(field: str) -> None:
    """A count below one is refused when the config is built, not mid-cycle.

    ``develop`` already rejects a round of no versions and a revision of no
    attempts, but it is called after the online rollout has run and been saved —
    the expensive half, and the one that costs real agent calls. A driver that
    left the check to ``develop`` therefore burns a rollout to find out its
    configuration was never runnable, and leaves behind a cycle directory
    holding a tree no record vouches for.
    """
    with pytest.raises(ValueError, match="at least one"):
        RunConfig(**{field: 0})


def test_a_three_cycle_run_grows_the_pool_by_one_tree_per_cycle(tmp_path: Path) -> None:
    """Issue #16's first "tests first": three cycles, one new replay world each.

    §3: after the rollout of iteration ``t`` its tree is recorded as ``𝒯_t`` and
    appended, "giving ``ℋ_t = ℋ_{t-1} ∪ {𝒯_t}``", and the offline phase then
    runs over that expanded collection. So cycle ``t`` dreams over ``t + 1``
    worlds and its own is the last of them — a driver that dreamed over the
    history *before* its rollout, or only over its own tree, fails here.
    """
    run = _run(tmp_path, ScriptedDeveloper())

    assert [record.index for record in run.cycles] == [0, 1, 2]
    assert [record.pool for record in run.cycles] == [
        ("cycle_000",),
        ("cycle_000", "cycle_001"),
        ("cycle_000", "cycle_001", "cycle_002"),
    ]
    assert [record.world for record in run.cycles] == ["cycle_000", "cycle_001", "cycle_002"]

    # Every world named is a tree on disk holding the attempts its record claims,
    # so the pool is trees and not just names in a log.
    for record in run.cycles:
        tree = DiscoveryTree.load(_cycle_dir(tmp_path, record.index) / TREE_FILENAME)
        assert record.attempts > 0
        assert len(tree) - 1 == record.attempts


def test_the_next_cycle_deploys_the_version_the_last_one_selected(tmp_path: Path) -> None:
    """§3's step 5 closing into step 1: ``π_{t+1}`` is what cycle ``t`` selected.

    The revision beats the incumbent on the world the incumbent recorded, so the
    selection is a real switch rather than a retention, and the two policies are
    distinguishable by what they do online: the incumbent runs to the round cap,
    the revision stops itself. A driver that redeployed ``π_t`` regardless — the
    easy bug, since the incumbent's source is right there in the loop — records
    the incumbent's stop reason in cycle 1 and fails.
    """
    run = _run(tmp_path, ScriptedDeveloper())
    first, second = run.cycles[0], run.cycles[1]

    assert not first.selection.retained
    assert (_cycle_dir(tmp_path, 1) / POLICY_FILENAME).read_text(encoding="utf-8") == (
        THRIFTY_SOURCE
    )
    assert first.stop_reason == STOP_MAX_ROUNDS
    assert second.stop_reason == STOP_EMPTY_BATCH
    assert second.attempts < first.attempts


def test_a_run_resumed_after_a_crash_keeps_the_cycles_it_finished(tmp_path: Path) -> None:
    """A crash in a later cycle costs that cycle and nothing before it.

    The interruption is the one that actually happens: a cycle whose rollout is
    on disk but whose record was never written. The tree it left behind is
    vouched for by nothing, so the resumed run redoes that cycle — and only that
    cycle. What it ends with has to be what an uninterrupted run ends with: the
    same three worlds, none dropped and none counted twice.
    """
    whole = _run(tmp_path / "whole", ScriptedDeveloper())

    resumed_dir = tmp_path / "resumed"
    _run(resumed_dir, ScriptedDeveloper(), cycles=2)
    (_cycle_dir(resumed_dir, 1) / RECORD_FILENAME).unlink()

    developer = ScriptedDeveloper()
    resumed = _run(resumed_dir, developer)

    # Cycle 0 was read back, not redone: only cycles 1 and 2 asked for a
    # revision. A driver that started from scratch asks three times.
    assert len(developer.calls) == 2
    assert [record.pool for record in resumed.cycles] == [
        record.pool for record in whole.cycles
    ]
    assert _records(resumed_dir) == _records(tmp_path / "whole")
    assert _trees(resumed_dir) == _trees(tmp_path / "whole")


@pytest.mark.parametrize(
    ("change", "named"),
    [
        pytest.param({"seed": 1}, "rollout.seed", id="rollout-seed"),
        pytest.param({"policy": THRIFTY_SOURCE}, "policy", id="policy"),
        pytest.param({"dreaming": DreamConfig(width=1)}, "dreaming.width", id="replay-width"),
    ],
)
def test_a_resume_under_a_changed_configuration_is_refused(
    tmp_path: Path, change: dict[str, object], named: str
) -> None:
    """Issue #45's first "tests first": a history is one experiment or it is not.

    A cycle's tree is what a later cycle's ``V^m`` averages over, so the cycles
    under one directory have to have been produced under the same run: the same
    starting policy, the same problem, the same online conditions and the same
    replay conditions. The manifest written before the first cycle says which,
    and a resume handed something else is refused rather than appended to it —
    naming the field, so a caller can tell a changed seed from a changed policy
    without reading the manifest themselves.
    """
    _run(tmp_path, ScriptedDeveloper(), cycles=1)

    with pytest.raises(RunError, match=named):
        _run(tmp_path, ScriptedDeveloper(), cycles=1, **change)


def test_a_resume_under_a_different_worker_count_still_resumes(tmp_path: Path) -> None:
    """Issue #45's third "tests first": the machine's parallelism is not the run.

    ``RolloutConfig.workers`` changes how long a rollout takes and nothing about
    the tree it records, and ``DreamConfig.workers`` how many ``(version, world)``
    cells replay at once, which :class:`~dream_rsi.dream.DreamConfig` documents as
    unable to move a reported number. Both are properties of the machine a
    session happens to run on, so a manifest that compared them would refuse a
    resume that changes nothing about the history. The width, which does change
    it, is held fixed here, and the resumed run goes on to a third cycle so the
    new workers actually run rather than the refusal merely not firing.
    """
    _run(tmp_path, ScriptedDeveloper(), cycles=2, workers=1, dreaming=DreamConfig(width=1))

    resumed = _run(
        tmp_path, ScriptedDeveloper(), cycles=3, workers=4, dreaming=DreamConfig(width=1, workers=4)
    )

    assert len(resumed.cycles) == 3


def test_the_manifest_is_written_before_the_first_cycle(tmp_path: Path) -> None:
    """Issue #45's "persist a run manifest before the first cycle".

    The manifest is what vouches for the cycles under the directory, so it has
    to be there before there is a cycle to vouch for: an implementation that
    wrote it after cycle 0 completed would leave a run that crashed in its first
    cycle with no manifest at all. The failed cycle here is a policy the sandbox
    refuses, so nothing was recorded and the manifest is all the run left.
    """
    source = _escape_source(tmp_path / "escaped.txt")

    with pytest.raises(SandboxError):
        _run(tmp_path / "run", ScriptedDeveloper(), cycles=1, policy=source)

    manifest = json.loads((tmp_path / "run" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["policy"] == source
    assert manifest["problem"] == PROBLEM


def test_a_run_directory_without_a_manifest_is_not_resumed(tmp_path: Path) -> None:
    """A finished cycle nothing vouches for is not accepted.

    The records say which cycles this run has; the manifest says what they were
    produced under. A directory that has one and not the other cannot be checked
    against the configuration it is being resumed with, and accepting the cycles
    anyway is the mixing issue #45 exists to prevent.
    """
    _run(tmp_path, ScriptedDeveloper(), cycles=1)
    (tmp_path / MANIFEST_FILENAME).unlink()

    with pytest.raises(RunError, match="manifest"):
        _run(tmp_path, ScriptedDeveloper(), cycles=1)


def test_two_runs_under_the_same_seed_record_the_same_cycles(tmp_path: Path) -> None:
    """End-to-end determinism (AGENTS.md rule 5), across two directories.

    Everything a cycle writes is compared, so a driver that let the thread pool's
    completion order, a wall clock, or the run directory's path reach a tree or a
    record fails here rather than in whatever consumes them later.
    """
    _run(tmp_path / "one", ScriptedDeveloper(), cycles=2)
    _run(tmp_path / "two", ScriptedDeveloper(), cycles=2)

    assert _trees(tmp_path / "one") == _trees(tmp_path / "two")
    assert _records(tmp_path / "one") == _records(tmp_path / "two")


def test_the_deployed_policy_is_sandboxed_online_too(tmp_path: Path) -> None:
    """Online deployment runs model-written code, so it goes behind the boundary.

    Dreaming already runs candidates in a child process (issue #13), but the
    winner of a cycle is *deployed*, and the driver is the one place that happens.
    A driver that executed the selected source in the harness — imported it,
    ``exec``-ed it, built the class itself — lets a policy that writes outside
    its scratch directory do exactly that, which is the one thing CLAUDE.md says
    never to relax. Here the write is refused, the cycle fails with it, and the
    file never appears.
    """
    escape = tmp_path / "escaped.txt"
    source = _escape_source(escape)

    with pytest.raises(SandboxError):
        _run(tmp_path / "run", ScriptedDeveloper(), cycles=1, policy=source)

    assert not escape.exists()


def test_the_toy_loop_runs_three_cycles_from_the_command_line(tmp_path: Path) -> None:
    """Issue #16's "done when": ``python -m dream_rsi.run --cycles 3``, no model.

    A subprocess rather than :func:`~dream_rsi.run.main` in-process, because what
    is being checked is that the module is runnable as the issue writes it — the
    default wiring included, since that is where a driver would otherwise need a
    provider to be runnable at all.
    """
    target = tmp_path / "run"
    completed = subprocess.run(
        [sys.executable, "-m", "dream_rsi.run", "--cycles", "3", str(target)],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for index in range(3):
        record = json.loads(
            (_cycle_dir(target, index) / RECORD_FILENAME).read_text(encoding="utf-8")
        )
        assert record["index"] == index
        assert len(record["pool"]) == index + 1


def test_a_resumed_run_reports_the_pool_it_had_at_shutdown(tmp_path: Path) -> None:
    """Issue #17's "done when": the simulator pool outlives the session.

    Two invocations against one directory, the second inheriting everything the
    first left: the same trees, under the same names, holding the same nodes and
    the same bytes. A driver whose pool was only the cycle directories it
    happened to walk this session — or one that re-recorded a finished cycle's
    tree on the way past and grew the pool by doing so — reports something else
    the second time round.
    """
    first = _run(tmp_path, ScriptedDeveloper(), cycles=2)

    resumed = _run(tmp_path, ScriptedDeveloper(), cycles=2)

    assert first.pool == ("cycle_000", "cycle_001")
    assert resumed.pool == first.pool
    assert resumed.stats == first.stats
    assert resumed.stats.trees == 2
    assert resumed.stats.nodes == sum(record.attempts + 1 for record in first.cycles)
    assert resumed.stats.bytes > 0


def test_a_resumed_run_restores_a_pool_tree_that_went_missing(tmp_path: Path) -> None:
    """A finished cycle is in the pool, whatever happened to the pool directory.

    The cycle records are what say which cycles this run may trust, and each one
    names the world it recorded; the pool is where dreaming reads those worlds
    from. A resumed run whose pool had been cleared out from under it — a
    half-copied run directory, a cleaner, an older run that predates the pool —
    would otherwise dream over a history shorter than the one its own records
    claim, silently, which is the one thing issue #17 says not to do.
    """
    whole = _run(tmp_path, ScriptedDeveloper(), cycles=2)
    shutil.rmtree(tmp_path / POOL_DIRNAME)

    resumed = _run(tmp_path, ScriptedDeveloper(), cycles=2)

    assert resumed.pool == whole.pool
    assert resumed.stats == whole.stats


def test_a_resumed_run_restores_a_pool_tree_that_will_not_load(tmp_path: Path) -> None:
    """A pool entry that is a name and no readable tree is a tree gone missing.

    A file left half-written by an interrupted copy still lists, so a resume
    that asked only which names the directory held would leave it in place —
    and then fail counting the pool, with a good copy of that tree sitting in
    the cycle directory the whole time. The name is not the tree; whether it
    loads is.
    """
    whole = _run(tmp_path, ScriptedDeveloper(), cycles=2)
    (tmp_path / POOL_DIRNAME / "cycle_000.json").write_text('{"nodes": [', encoding="utf-8")

    resumed = _run(tmp_path, ScriptedDeveloper(), cycles=2)

    assert resumed.pool == whole.pool
    assert resumed.stats == whole.stats


# Every file a finished cycle publishes, as a path under the run directory, and
# what a run resumed with that file gone has to do: carry on with the history it
# had, or refuse with a RunError naming what is missing. The last entry is the
# state the publish order exists to rule out — a record whose tree is gone from
# the pool *and* from the cycle's own copy — which nothing can recover and so
# has to be the refusal rather than a history nobody chose.
_PUBLISHED = (
    pytest.param((MANIFEST_FILENAME,), "refused", id="manifest"),
    pytest.param((f"{CYCLES_DIRNAME}/cycle_000/{RECORD_FILENAME}",), "resumed", id="record"),
    pytest.param((f"{CYCLES_DIRNAME}/cycle_000/{TREE_FILENAME}",), "resumed", id="cycle-tree"),
    pytest.param((f"{CYCLES_DIRNAME}/cycle_000/rounds.json",), "resumed", id="round-log"),
    pytest.param((f"{CYCLES_DIRNAME}/cycle_000/{POLICY_FILENAME}",), "refused", id="policy"),
    pytest.param(
        (f"{CYCLES_DIRNAME}/cycle_000/{NEXT_POLICY_FILENAME}",), "refused", id="next-policy"
    ),
    pytest.param((f"{CYCLES_DIRNAME}/cycle_000/{TIMING_FILENAME}",), "resumed", id="timing"),
    pytest.param((f"{POOL_DIRNAME}/cycle_000{TREE_SUFFIX}",), "resumed", id="pool-entry"),
    pytest.param(
        (f"{CYCLES_DIRNAME}/cycle_000/{TREE_FILENAME}", f"{POOL_DIRNAME}/cycle_000{TREE_SUFFIX}"),
        "refused",
        id="tree-and-pool-entry",
    ),
)


@pytest.mark.parametrize(("missing", "outcome"), _PUBLISHED)
def test_a_run_directory_missing_a_published_file_resumes_or_says_why(
    tmp_path: Path, missing: tuple[str, ...], outcome: str
) -> None:
    """Issue #47's "tests first": what a crash leaves behind is still readable.

    Power loss is the one failure a test cannot stage, so what is pinned here is
    the state it leaves: take a published file away and a resumed run either
    carries on with the history it had or refuses with a :class:`RunError`
    naming what is gone. A traceback out of a loader is neither — and a shorter
    history nobody chose is worse than either, which is what the tree-and-
    pool-entry case checks: a record is trusted only while the tree it vouches
    for can be found, and when it cannot the run says so.
    """
    whole = _run(tmp_path / "whole", ScriptedDeveloper(), cycles=1)
    damaged = tmp_path / "damaged"
    shutil.copytree(tmp_path / "whole", damaged)
    for name in missing:
        (damaged / name).unlink()

    if outcome == "refused":
        with pytest.raises(RunError) as refusal:
            _run(damaged, ScriptedDeveloper(), cycles=1)
        assert Path(missing[0]).name in str(refusal.value)
        return

    resumed = _run(damaged, ScriptedDeveloper(), cycles=1)
    assert resumed.pool == whole.pool
    assert resumed.stats == whole.stats


def test_a_pool_limit_bounds_the_history_a_cycle_dreams_over(tmp_path: Path) -> None:
    """Subsampling is opt-in, and what it dropped is on the record.

    Dreaming costs one replay per tree per version, so a pool that grows every
    cycle eventually costs more than the online evaluations it replaced. A run
    given a limit dreams over that many worlds and its record says which —
    a driver that took the limit and went on replaying the whole history, or one
    that dropped the trees from the store instead of from this round, fails
    here. The store keeps everything either way: the limit is a budget for one
    cycle, not a retention policy.
    """
    run = _run(tmp_path, ScriptedDeveloper(), cycles=3, pool=PoolConfig(limit=1, seed=0))

    # Each cycle dreams over one world, and §3 appends before it dreams, so the
    # one world left is the cycle's own.
    assert [record.pool for record in run.cycles] == [
        ("cycle_000",),
        ("cycle_001",),
        ("cycle_002",),
    ]
    assert run.pool == ("cycle_000", "cycle_001", "cycle_002")
    assert run.stats.trees == 3


# π_1 for the hand count: every round opens one more branch from the root and
# does nothing else, online and in replay alike. So what it records is a star —
# one round, one attempt, one child of the root — and a replay of that star
# reveals one node per round until there are none left. Both halves of a cycle
# are then countable from the config alone.
STAR_SOURCE = """\
class OptimalPolicy:
    def __init__(self, config=None):
        self.config = dict(config or {})

    def select(self, tree, eligible, width):
        return (tree.root_id,) * width
"""

# The revision the hand count's developer answers with: one reveal and then the
# empty batch, so its row of the grid is a different number from the incumbent's
# and a harness that reported one of them twice is visible.
ONE_REVEAL_SOURCE = """\
class OptimalPolicy:
    def __init__(self, config=None):
        self.config = dict(config or {})
        self.spent = 0

    def reset(self, rng=None):
        self.spent = 0

    def select(self, tree, eligible, width):
        self.spent += 1
        return (tree.root_id,) if self.spent == 1 else ()
"""


def test_a_cycle_costs_exactly_what_the_hand_count_says(tmp_path: Path) -> None:
    """Issue #18's first "tests first": the counters against a run counted by hand.

    One cycle at ``W = 1`` over three rounds, under a policy that opens one
    branch from the root per round. Every number follows from that:

    * three attempts, so three discovery-agent calls (§4's discovery cost) and
      three evaluations, since the toy agent raises on none of them;
    * two versions — the incumbent and the one revision the developer answers
      with — over the one world this cycle appended, so two replayed cells and
      one policy-development call;
    * the incumbent replaying its own star reveals all three recorded attempts,
      the revision stops after one, so four revealed nodes in all.

    A driver that counted rounds instead of attempts, cells instead of reveals,
    or the whole pool instead of this cycle's replays fails on one of them.
    """
    developer = ScriptedDeveloper(script=(ONE_REVEAL_SOURCE,))

    run = _run(tmp_path, developer, cycles=1, policy=STAR_SOURCE, workers=1, rounds=3)

    cost = run.cycles[0].cost
    assert cost.online == OnlineCost(agent_calls=3, evaluations=3)
    assert cost.dreaming == DreamCost(developer_calls=1, cells=2, reveals=4)

    # The cycle's best score is the best ``s_v`` in the tree that cycle recorded,
    # not the last one attempted and not one from another cycle.
    tree = DiscoveryTree.load(_cycle_dir(tmp_path, 0) / TREE_FILENAME)
    assert run.cycles[0].best_score == max(
        node.score for node in tree.iter_nodes() if node.score is not None
    )


def test_a_resumed_run_does_not_count_the_cycles_it_already_did_twice(tmp_path: Path) -> None:
    """Issue #18's second "tests first": resuming re-reads costs, it does not re-spend them.

    Three invocations against one directory — two cycles, then the third, then
    one with nothing left to do — against an uninterrupted run of the same three
    cycles. The totals have to agree with it every time. A driver that added what
    it spent this session to what the records it read back already said doubles
    the cycles it redid; one that counted only this session's work loses the
    cycles it resumed.
    """
    whole = _run(tmp_path / "whole", ScriptedDeveloper(), cycles=3)

    resumed_dir = tmp_path / "resumed"
    _run(resumed_dir, ScriptedDeveloper(), cycles=2)
    resumed = _run(resumed_dir, ScriptedDeveloper(), cycles=3)
    again = _run(resumed_dir, ScriptedDeveloper(), cycles=3)

    assert [record.cost for record in resumed.cycles] == [
        record.cost for record in whole.cycles
    ]
    assert resumed.totals == whole.totals
    assert again.totals == whole.totals
    # Against the records rather than against another total, so a run that
    # totalled some of its cycles agrees with itself and still fails here.
    assert again.totals.online.agent_calls == sum(
        record.cost.online.agent_calls for record in whole.cycles
    )
    assert again.totals.dreaming.reveals == sum(
        record.cost.dreaming.reveals for record in whole.cycles
    )


def test_the_report_renders_from_a_run_read_back_off_disk(tmp_path: Path) -> None:
    """Issue #18's third "tests first": the report needs the directory and nothing else.

    :meth:`Run.load` builds a run out of the cycle records, the pool and the
    policy on disk, with no agent, no evaluator and no developer behind it — the
    state a run is in once the process that made it is gone. What it renders has
    to be what the run itself rendered, including the line saying why each
    cycle's version was the one deployed next. A report that reached for
    anything the cycles did not record cannot be produced here at all.
    """
    run = _run(tmp_path, ScriptedDeveloper(), cycles=2)

    loaded = Run.load(tmp_path)

    assert loaded.cycles == run.cycles
    assert loaded.policy == run.policy
    assert loaded.to_text() == run.to_text()
    for record in run.cycles:
        assert record.selection.rationale in loaded.to_text()


def test_the_report_diffs_the_policy_each_cycle_deployed(tmp_path: Path) -> None:
    """Issue #20: the report shows the policy's *code* changing, cycle by cycle.

    The revision wins cycle 0, so cycle 1 deploys different source and the report
    has to show that as a diff of the two. Cycle 1 then selects the same source
    again — the stub answers from a one-entry script — so there is nothing to
    show and the report has to say so rather than print an empty hunk or repeat
    the previous cycle's.

    Asserted on the lines the two sources differ by, so a report that diffed a
    cycle against itself (always unchanged), against the *selected* source
    instead of the *deployed* one (always changed), or one cycle out of step
    fails here.
    """
    run = _run(tmp_path, ScriptedDeveloper(), cycles=2)

    text = run.to_text()
    printed = [line.strip() for line in text.splitlines()]
    before, after = BREADTH_SOURCE.splitlines(), THRIFTY_SOURCE.splitlines()
    for line in set(after) - set(before):
        assert f"+{line}" in printed
    for line in set(before) - set(after):
        assert f"-{line}" in printed
    # Cycle 1 redeploys what cycle 0 already deployed a revision of, and the
    # revision offered is the same text again: no diff, and the report says it.
    assert "cycle 1: unchanged" in printed


def test_the_toy_loop_reports_the_cost_split_from_the_command_line(tmp_path: Path) -> None:
    """Issue #18's "done when": a 3-cycle toy run emits a report showing the split.

    The ratio between the two halves is the paper's whole argument, so the run
    has to say what each of them spent without the reader adding anything up.
    The numbers printed are checked against the records on disk rather than
    against constants, so this fails on a report that prints one half twice or
    totals only the cycles of the session that happened to print it.
    """
    target = tmp_path / "run"
    completed = subprocess.run(
        [sys.executable, "-m", "dream_rsi.run", "--cycles", "3", str(target)],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    records = [
        json.loads((_cycle_dir(target, index) / RECORD_FILENAME).read_text(encoding="utf-8"))
        for index in range(3)
    ]
    calls = sum(record["cost"]["online"]["agent_calls"] for record in records)
    reveals = sum(record["cost"]["dreaming"]["reveals"] for record in records)
    assert calls > 0 and reveals > 0
    assert f"{calls} online agent call(s)" in completed.stdout
    assert f"{reveals} node(s)" in completed.stdout


def test_a_cycle_whose_timing_went_missing_is_still_a_finished_cycle(tmp_path: Path) -> None:
    """The record vouches for a cycle; the clock written beside it does not.

    Wall clock is kept out of the record because a record is a function of the
    run's seed and a clock is a function of the machine — so it is a second file,
    and a second file is one a cleaner, a half-copied run directory or a run
    older than this report can leave behind. Everything the cycle spent is on the
    record either way, so a driver that insisted on the timing would redo a cycle
    it already has and pay for it a second time.
    """
    run = _run(tmp_path, ScriptedDeveloper(), cycles=1)
    (_cycle_dir(tmp_path, 0) / TIMING_FILENAME).unlink()

    developer = ScriptedDeveloper()
    resumed = _run(tmp_path, developer, cycles=1)

    assert developer.calls == []
    assert resumed.totals == run.totals
    assert resumed.cycles[0].timing is None
    assert Run.load(tmp_path).to_text() == resumed.to_text()


def test_a_run_with_no_finished_cycle_reports_that_and_not_a_traceback(tmp_path: Path) -> None:
    """The report of a run that crashed in cycle 0 is a report with nothing in it.

    That directory is exactly what a report is read from — someone is looking
    because the run stopped — so rendering it must not divide the reveals by no
    agent calls or reach for a policy no cycle ever selected.
    """
    (tmp_path / CYCLES_DIRNAME).mkdir(parents=True)

    run = Run.load(tmp_path)

    assert run.cycles == ()
    assert run.policy == ""
    assert "0 cycle(s)" in run.to_text()
